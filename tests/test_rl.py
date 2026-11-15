"""Unit and integration tests for Part 2: RL features, environment, network, and PPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from evaluate import main as evaluate_main
from tetris.pieces import BOARD_HEIGHT, BOARD_WIDTH
from tetris.rl.env import MAX_CANDIDATES, TetrisRLEnv, VecTetrisEnv
from tetris.rl.features import (
    NUM_FEATURES,
    compute_board_potential,
    extract_candidate_features,
)
from tetris.rl.network import TetrisActorCritic, load_checkpoint, save_checkpoint
from tetris.rl.ppo import PPOConfig, resolve_device, train_ppo


def _reference_features_single(
    board: np.ndarray, lines_cleared: int, landing_height: float
) -> np.ndarray:
    """Loop-based scalar reference implementation for verifying vectorized features."""
    occ = board != 0
    col_heights = []
    holes = 0
    covered_weight = 0

    for c in range(BOARD_WIDTH):
        top_row = BOARD_HEIGHT
        blocks_seen = 0
        for r in range(BOARD_HEIGHT):
            if occ[r, c]:
                if top_row == BOARD_HEIGHT:
                    top_row = r
                blocks_seen += 1
            elif blocks_seen > 0:
                holes += 1
                covered_weight += blocks_seen
        col_heights.append(BOARD_HEIGHT - top_row)

    bumpiness = sum(abs(col_heights[c] - col_heights[c + 1]) for c in range(BOARD_WIDTH - 1))
    sum_h = sum(col_heights)
    max_h = max(col_heights)
    min_h = min(col_heights)

    row_trans = 0
    for r in range(BOARD_HEIGHT):
        prev = True
        for c in range(BOARD_WIDTH):
            cur = bool(occ[r, c])
            if cur != prev:
                row_trans += 1
            prev = cur
        if not prev:
            row_trans += 1

    col_trans = 0
    for c in range(BOARD_WIDTH):
        prev = False
        for r in range(BOARD_HEIGHT):
            cur = bool(occ[r, c])
            if cur != prev:
                col_trans += 1
            prev = cur
        if not prev:
            col_trans += 1

    well_sum = 0
    for c in range(BOARD_WIDTH):
        left_h = BOARD_HEIGHT if c == 0 else col_heights[c - 1]
        right_h = BOARD_HEIGHT if c == BOARD_WIDTH - 1 else col_heights[c + 1]
        depth = max(0, min(left_h, right_h) - col_heights[c])
        well_sum += depth * (depth + 1) // 2

    return np.array(
        [
            lines_cleared / 4.0,
            1.0 if lines_cleared == 4 else 0.0,
            holes / 40.0,
            min(covered_weight, 100.0) / 100.0,
            bumpiness / 40.0,
            sum_h / 200.0,
            max_h / 20.0,
            (max_h - min_h) / 20.0,
            row_trans / 100.0,
            col_trans / 100.0,
            min(well_sum, 60.0) / 60.0,
            landing_height / 20.0,
        ],
        dtype=np.float32,
    )


def test_vectorized_features_match_reference() -> None:
    rng = np.random.default_rng(77)
    boards = (rng.random((8, BOARD_HEIGHT, BOARD_WIDTH)) > 0.68).astype(np.uint8)
    boards[0].fill(0)  # Empty board edge case
    boards[1, 16:, :9] = 1  # Deep well on column 9
    lines = np.array([0, 4, 1, 2, 3, 0, 1, 4], dtype=np.int32)
    landing_h = np.array([1.0, 2.5, 6.0, 10.0, 14.5, 3.0, 7.0, 19.0], dtype=np.float32)

    got = extract_candidate_features(boards, lines, landing_h)
    expected = np.stack(
        [_reference_features_single(boards[i], int(lines[i]), float(landing_h[i])) for i in range(8)],
        axis=0,
    )
    assert got.shape == (8, NUM_FEATURES)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    assert compute_board_potential(boards[0]) == 0.0
    assert compute_board_potential(boards[2]) < 0.0


def test_rl_env_invariants_and_truncation() -> None:
    env = TetrisRLEnv(seed=3, gamma=0.99, max_pieces_per_episode=5)
    (grids, feats, mask), info = env.reset(seed=3)
    assert grids.shape == (MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH)
    assert feats.shape == (MAX_CANDIDATES, NUM_FEATURES)
    assert mask.shape == (MAX_CANDIDATES,)
    assert mask.any()
    assert info == {}

    saw_truncation = False
    for _ in range(5):
        valid = np.flatnonzero(mask)
        (grids, feats, mask), reward, term, trunc, info = env.step(int(valid[0]))
        assert isinstance(reward, float)
        if trunc:
            saw_truncation = True
            assert not term
            assert "episode" in info
            assert "terminal_observation" in info
            tg, tf, tm = info["terminal_observation"]
            assert tm.any()
            break
    assert saw_truncation

    # Vectorized env auto-resets cleanly after truncation/termination
    vec = VecTetrisEnv(num_envs=3, base_seed=10, max_pieces_per_episode=4)
    vg, vf, vm = vec.reset()
    assert vg.shape == (3, MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH)
    for _ in range(6):
        acts = np.array([int(np.flatnonzero(vm[i])[0]) for i in range(3)])
        (vg, vf, vm), _, _, _, _ = vec.step(acts)
        assert vm.any(axis=1).all()


def test_actor_critic_masking_gradients_and_checkpoint(tmp_path: Path) -> None:
    model = TetrisActorCritic()
    grids = torch.randint(0, 2, (4, MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.uint8)
    feats = torch.randn(4, MAX_CANDIDATES, NUM_FEATURES)
    mask = torch.zeros(4, MAX_CANDIDATES, dtype=torch.bool)
    mask[:, :5] = True

    for _ in range(15):
        action, logp, ent, val = model.get_action_and_value(grids, feats, mask)
        assert (action < 5).all()
        assert logp.shape == (4,)
        assert ent.shape == (4,)
        assert val.shape == (4,)

    loss = -(logp.mean() - 0.5 * (val ** 2).mean() + 0.01 * ent.mean())
    loss.backward()
    has_cnn_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.spatial_conv.parameters())
    has_mlp_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.struct_mlp.parameters())
    assert has_cnn_grad and has_mlp_grad

    ckpt = save_checkpoint(tmp_path / "model.pt", model, metadata={"update": 1})
    loaded, meta = load_checkpoint(ckpt, device="cpu")
    assert meta["update"] == 1
    model.eval()
    loaded.eval()
    with torch.no_grad():
        l1, v1 = model(grids, feats, mask)
        l2, v2 = loaded(grids, feats, mask)
        torch.testing.assert_close(l1, l2)
        torch.testing.assert_close(v1, v2)


def test_ppo_smoke_train_and_evaluate(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    plot_file = tmp_path / "curve.png"
    cfg = PPOConfig(
        num_envs=2,
        rollout_steps=8,
        updates=2,
        minibatch_size=8,
        ppo_epochs=2,
        eval_interval=1,
        eval_episodes=2,
        eval_max_pieces=20,
        max_pieces_per_episode=15,
        checkpoint_dir=str(ckpt_dir),
        plot_path=str(plot_file),
    )
    device = resolve_device("auto")
    summary = train_ppo(cfg, device=device)
    assert Path(summary["best_checkpoint"]).exists()
    assert plot_file.exists() and plot_file.stat().st_size > 1000

    # Verify evaluate CLI entrypoint works with the saved checkpoint
    evaluate_main(
        [
            "--checkpoint",
            str(ckpt_dir / "best_model.pt"),
            "--episodes",
            "2",
            "--max-pieces",
            "20",
            "--device",
            "cpu",
        ]
    )
