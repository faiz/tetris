"""Masked-Action PPO trainer, deterministic evaluator, and training curve plotter."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tetris.pieces import BOARD_HEIGHT, BOARD_WIDTH
from tetris.rl.env import MAX_CANDIDATES, TetrisRLEnv, VecTetrisEnv
from tetris.rl.features import NUM_FEATURES
from tetris.rl.network import TetrisActorCritic, save_checkpoint


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve 'auto' to Apple Silicon MPS when available, else CPU."""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested, but torch.backends.mps.is_available() is False.")
    return torch.device(device_arg)


@dataclass
class PPOConfig:
    """Hyperparameters for Masked-Action PPO training on Tetris."""

    num_envs: int = 16
    rollout_steps: int = 128
    updates: int = 150
    lr: float = 3e-4
    final_lr: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 128
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    final_ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    max_pieces_per_episode: int = 2500
    eval_interval: int = 25
    eval_episodes: int = 8
    eval_max_pieces: int = 3000
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    plot_path: str = "assets/training_curve.png"


@torch.no_grad()
def evaluate_policy(
    model: TetrisActorCritic,
    num_episodes: int = 20,
    max_pieces: int = 5000,
    base_seed: int = 10000,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run deterministic evaluation episodes across seeded environments."""
    was_training = model.training
    model.eval()
    dev = torch.device(device)

    envs = [
        TetrisRLEnv(seed=base_seed + i, gamma=0.99, max_pieces_per_episode=max_pieces)
        for i in range(num_episodes)
    ]
    obs_list = [env.reset(seed=base_seed + i)[0] for i, env in enumerate(envs)]
    active = list(range(num_episodes))
    results: list[dict[str, Any]] = [{} for _ in range(num_episodes)]

    while active:
        grids = np.stack([obs_list[i][0] for i in active], axis=0)
        feats = np.stack([obs_list[i][1] for i in active], axis=0)
        masks = np.stack([obs_list[i][2] for i in active], axis=0)

        g_t = torch.from_numpy(grids).to(dev)
        f_t = torch.from_numpy(feats).to(dev)
        m_t = torch.from_numpy(masks).to(dev)
        actions, _ = model.select_action(g_t, f_t, m_t, deterministic=True)
        act_np = actions.cpu().numpy()

        next_active: list[int] = []
        for idx_in_batch, env_idx in enumerate(active):
            obs, _, term, trunc, info = envs[env_idx].step(int(act_np[idx_in_batch]))
            if term or trunc:
                ep = dict(info["episode"])
                ep["seed"] = base_seed + env_idx
                ep["truncated"] = bool(trunc)
                results[env_idx] = ep
            else:
                obs_list[env_idx] = obs
                next_active.append(env_idx)
        active = next_active

    if was_training:
        model.train()

    lines_arr = np.array([r["lines"] for r in results], dtype=np.float64)
    score_arr = np.array([r["score"] for r in results], dtype=np.float64)
    pieces_arr = np.array([r["pieces"] for r in results], dtype=np.float64)
    tetris_arr = np.array([r["tetris_rate"] for r in results], dtype=np.float64)

    return {
        "episodes": results,
        "num_episodes": num_episodes,
        "mean_lines": float(lines_arr.mean()),
        "median_lines": float(np.median(lines_arr)),
        "std_lines": float(lines_arr.std()),
        "min_lines": int(lines_arr.min()),
        "max_lines": int(lines_arr.max()),
        "mean_score": float(score_arr.mean()),
        "median_score": float(np.median(score_arr)),
        "max_score": int(score_arr.max()),
        "mean_pieces": float(pieces_arr.mean()),
        "mean_tetris_rate": float(tetris_arr.mean()),
    }


def _rolling_mean(values: list[float], window: int = 30) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def plot_training_curves(history: dict[str, Any], plot_path: str | Path) -> Path:
    """Render 3-panel training and evaluation progression plot to PNG."""
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.grid(True, color="#30363d", alpha=0.55, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    ep_steps = history["ep_steps"]
    ep_lines = history["ep_lines"]
    ep_scores = history["ep_scores"]
    eval_steps = history["eval_steps"]
    eval_lines = history["eval_mean_lines"]
    eval_scores = history["eval_mean_scores"]
    upd_steps = history["update_steps"]
    entropies = history["entropy"]
    value_losses = history["value_loss"]

    # Panel 1: Lines cleared
    ax1 = axes[0]
    if ep_steps:
        ax1.plot(ep_steps, ep_lines, color="#38bdf8", alpha=0.22, linewidth=0.9, label="Train Episode Lines")
        ax1.plot(ep_steps, _rolling_mean(ep_lines, 30), color="#38bdf8", linewidth=2.0, label="Train Rolling Mean (30)")
    if eval_steps:
        ax1.plot(eval_steps, eval_lines, color="#38d972", marker="o", markersize=5, linewidth=2.2, label="Deterministic Eval Mean")
    ax1.set_ylabel("Lines Cleared", color="#f0f6fc", fontweight="bold")
    ax1.set_title("Tetris RL Training Progression (Masked-Action PPO on Apple Silicon)", color="#f0f6fc", fontsize=13, fontweight="bold", pad=10)
    if ep_steps or eval_steps:
        ax1.legend(loc="lower right", frameon=True, facecolor="#0d1117", edgecolor="#30363d")

    # Panel 2: Official Score
    ax2 = axes[1]
    if ep_steps:
        ax2.plot(ep_steps, ep_scores, color="#ffd500", alpha=0.20, linewidth=0.9, label="Train Episode Score")
        ax2.plot(ep_steps, _rolling_mean(ep_scores, 30), color="#ffd500", linewidth=2.0, label="Train Rolling Mean (30)")
    if eval_steps:
        ax2.plot(eval_steps, eval_scores, color="#ff922b", marker="s", markersize=5, linewidth=2.2, label="Deterministic Eval Mean")
    ax2.set_ylabel("Official Score", color="#f0f6fc", fontweight="bold")
    if ep_steps or eval_steps:
        ax2.legend(loc="lower right", frameon=True, facecolor="#0d1117", edgecolor="#30363d")

    # Panel 3: Entropy & Value Loss
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3_r.spines["top"].set_color("#30363d")
    ax3_r.spines["right"].set_color("#30363d")
    lines_legend = []
    if upd_steps:
        l1 = ax3.plot(upd_steps, entropies, color="#b24bf3", linewidth=1.8, label="Policy Entropy")
        l2 = ax3_r.plot(upd_steps, value_losses, color="#ff4757", linewidth=1.8, linestyle="--", label="Value Loss")
        lines_legend = l1 + l2
    ax3.set_ylabel("Policy Entropy", color="#b24bf3", fontweight="bold")
    ax3_r.set_ylabel("Value Loss", color="#ff4757", fontweight="bold")
    ax3.set_xlabel("Environment Steps (Piece Placements)", color="#f0f6fc", fontweight="bold")
    if lines_legend:
        labels = [l.get_label() for l in lines_legend]
        ax3.legend(lines_legend, labels, loc="upper right", frameon=True, facecolor="#0d1117", edgecolor="#30363d")

    plt.tight_layout()
    fig.savefig(out, dpi=150, facecolor="#0d1117")
    plt.close(fig)
    return out


def train_ppo(config: PPOConfig, device: torch.device) -> dict[str, Any]:
    """Train TetrisActorCritic using Masked-Action PPO with GAE and deterministic eval."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    vec_env = VecTetrisEnv(
        num_envs=config.num_envs,
        base_seed=config.seed,
        gamma=config.gamma,
        max_pieces_per_episode=config.max_pieces_per_episode,
    )
    model = TetrisActorCritic(num_features=NUM_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    t_steps = config.rollout_steps
    n_envs = config.num_envs
    batch_size = t_steps * n_envs

    obs_grids = torch.zeros((t_steps, n_envs, MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.uint8)
    obs_feats = torch.zeros((t_steps, n_envs, MAX_CANDIDATES, NUM_FEATURES), dtype=torch.float32)
    obs_masks = torch.zeros((t_steps, n_envs, MAX_CANDIDATES), dtype=torch.bool)
    actions_buf = torch.zeros((t_steps, n_envs), dtype=torch.int64)
    logprobs_buf = torch.zeros((t_steps, n_envs), dtype=torch.float32)
    rewards_buf = torch.zeros((t_steps, n_envs), dtype=torch.float32)
    values_buf = torch.zeros((t_steps, n_envs), dtype=torch.float32)
    dones_buf = torch.zeros((t_steps, n_envs), dtype=torch.bool)

    next_g, next_f, next_m = vec_env.reset()
    global_step = 0
    best_eval_lines = -1.0
    best_eval_rank = (-1, -1.0)
    best_checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
    latest_checkpoint_path = Path(config.checkpoint_dir) / "latest_model.pt"

    recent_lines: deque[int] = deque(maxlen=50)
    recent_scores: deque[int] = deque(maxlen=50)
    history: dict[str, list[float]] = {
        "ep_steps": [],
        "ep_lines": [],
        "ep_scores": [],
        "eval_steps": [],
        "eval_mean_lines": [],
        "eval_mean_scores": [],
        "update_steps": [],
        "entropy": [],
        "value_loss": [],
        "policy_loss": [],
    }

    start_time = time.perf_counter()

    for update in range(1, config.updates + 1):
        frac = 1.0 - (update - 1) / max(1, config.updates - 1) if config.updates > 1 else 0.0
        cur_lr = config.final_lr + (config.lr - config.final_lr) * frac
        cur_ent = config.final_ent_coef + (config.ent_coef - config.final_ent_coef) * frac
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # 1. Rollout collection
        model.eval()
        for step in range(t_steps):
            global_step += n_envs
            g_cpu = torch.from_numpy(next_g)
            f_cpu = torch.from_numpy(next_f)
            m_cpu = torch.from_numpy(next_m)

            obs_grids[step] = g_cpu
            obs_feats[step] = f_cpu
            obs_masks[step] = m_cpu

            with torch.no_grad():
                act, logp, _, val = model.get_action_and_value(
                    g_cpu.to(device), f_cpu.to(device), m_cpu.to(device)
                )

            act_cpu = act.cpu()
            actions_buf[step] = act_cpu
            logprobs_buf[step] = logp.cpu()
            values_buf[step] = val.cpu()

            (next_g, next_f, next_m), rew_np, term_np, trunc_np, infos = vec_env.step(
                act_cpu.numpy()
            )
            rewards_buf[step] = torch.from_numpy(rew_np)
            dones_buf[step] = torch.from_numpy(term_np | trunc_np)

            # Bootstrap truncated episodes with terminal_observation value
            trunc_indices: list[int] = []
            term_obs_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for env_i, info in enumerate(infos):
                if "episode" in info:
                    ep = info["episode"]
                    recent_lines.append(ep["lines"])
                    recent_scores.append(ep["score"])
                    history["ep_steps"].append(float(global_step))
                    history["ep_lines"].append(float(ep["lines"]))
                    history["ep_scores"].append(float(ep["score"]))
                if trunc_np[env_i] and not term_np[env_i] and "terminal_observation" in info:
                    trunc_indices.append(env_i)
                    term_obs_list.append(info["terminal_observation"])

            if trunc_indices:
                tg = torch.from_numpy(np.stack([o[0] for o in term_obs_list], axis=0)).to(device)
                tf = torch.from_numpy(np.stack([o[1] for o in term_obs_list], axis=0)).to(device)
                tm = torch.from_numpy(np.stack([o[2] for o in term_obs_list], axis=0)).to(device)
                with torch.no_grad():
                    term_vals = model.get_value(tg, tf, tm).cpu()
                for idx_pos, env_i in enumerate(trunc_indices):
                    rewards_buf[step, env_i] += config.gamma * term_vals[idx_pos]

        # 2. GAE(lambda)
        with torch.no_grad():
            next_val = model.get_value(
                torch.from_numpy(next_g).to(device),
                torch.from_numpy(next_f).to(device),
                torch.from_numpy(next_m).to(device),
            ).cpu()

        advantages = torch.zeros_like(rewards_buf)
        last_gae = torch.zeros((n_envs,), dtype=torch.float32)
        for t in reversed(range(t_steps)):
            next_non_terminal = 1.0 - dones_buf[t].float()
            next_values = next_val if t == t_steps - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + config.gamma * next_values * next_non_terminal - values_buf[t]
            last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_buf

        # 3. PPO policy & value updates
        model.train()
        b_grids = obs_grids.reshape(batch_size, MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH)
        b_feats = obs_feats.reshape(batch_size, MAX_CANDIDATES, NUM_FEATURES)
        b_masks = obs_masks.reshape(batch_size, MAX_CANDIDATES)
        b_actions = actions_buf.reshape(batch_size)
        b_logprobs = logprobs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)

        pg_losses: list[float] = []
        v_losses: list[float] = []
        ent_vals: list[float] = []

        indices = np.arange(batch_size)
        mb_size = min(config.minibatch_size, batch_size)
        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, mb_size):
                mb_idx = indices[start : start + mb_size]
                mb_idx_t = torch.from_numpy(mb_idx)

                mb_g = b_grids[mb_idx_t].to(device)
                mb_f = b_feats[mb_idx_t].to(device)
                mb_m = b_masks[mb_idx_t].to(device)
                mb_a = b_actions[mb_idx_t].to(device)
                mb_old_lp = b_logprobs[mb_idx_t].to(device)
                mb_adv = b_advantages[mb_idx_t].to(device)
                mb_ret = b_returns[mb_idx_t].to(device)

                _, new_lp, entropy, new_val = model.get_action_and_value(
                    mb_g, mb_f, mb_m, action=mb_a
                )
                log_ratio = new_lp - mb_old_lp
                ratio = log_ratio.exp()

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + config.vf_coef * v_loss - cur_ent * ent_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_vals.append(float(ent_loss.item()))

        mean_ent = float(np.mean(ent_vals))
        mean_vloss = float(np.mean(v_losses))
        mean_pgloss = float(np.mean(pg_losses))
        history["update_steps"].append(float(global_step))
        history["entropy"].append(mean_ent)
        history["value_loss"].append(mean_vloss)
        history["policy_loss"].append(mean_pgloss)

        should_eval = (
            update == 1
            or update == config.updates
            or (config.eval_interval > 0 and update % config.eval_interval == 0)
        )
        if should_eval:
            eval_stats = evaluate_policy(
                model=model,
                num_episodes=config.eval_episodes,
                max_pieces=config.eval_max_pieces,
                base_seed=10000,
                device=device,
            )
            eval_lines = eval_stats["mean_lines"]
            eval_score = eval_stats["mean_score"]
            history["eval_steps"].append(float(global_step))
            history["eval_mean_lines"].append(float(eval_lines))
            history["eval_mean_scores"].append(float(eval_score))

            metadata = {
                "update": update,
                "global_step": global_step,
                "eval_mean_lines": eval_lines,
                "eval_mean_score": eval_score,
                "eval_mean_tetris_rate": eval_stats["mean_tetris_rate"],
                "device": str(device),
                "config": {k: v for k, v in asdict(config).items()},
            }
            save_checkpoint(latest_checkpoint_path, model, metadata=metadata)
            eval_rank = (int(eval_lines // 25), float(eval_score))
            if eval_rank >= best_eval_rank:
                best_eval_rank = eval_rank
                best_eval_lines = max(best_eval_lines, eval_lines)
                save_checkpoint(best_checkpoint_path, model, metadata=metadata)

            elapsed = time.perf_counter() - start_time
            sps = global_step / max(elapsed, 1e-6)
            tr_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
            print(
                f"[Update {update:3d}/{config.updates}] steps={global_step:7d} ({sps:6.0f} sps) | "
                f"train_lines={tr_lines:6.1f} | eval_lines={eval_lines:7.1f} | "
                f"eval_score={eval_score:9.0f} | tetris={eval_stats['mean_tetris_rate']*100:4.1f}% | "
                f"ent={mean_ent:.3f} vloss={mean_vloss:.3f} best={best_eval_lines:.1f}"
            )
        elif update % 10 == 0:
            elapsed = time.perf_counter() - start_time
            sps = global_step / max(elapsed, 1e-6)
            tr_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
            print(
                f"[Update {update:3d}/{config.updates}] steps={global_step:7d} ({sps:6.0f} sps) | "
                f"train_lines={tr_lines:6.1f} | ent={mean_ent:.3f} vloss={mean_vloss:.3f}"
            )

    plot_training_curves(history, config.plot_path)
    return {
        "best_eval_lines": best_eval_lines,
        "best_checkpoint": str(best_checkpoint_path),
        "plot_path": config.plot_path,
        "history": history,
    }
