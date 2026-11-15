"""CLI entrypoint for training the Tetris RL agent on Apple Silicon MPS or CPU."""

from __future__ import annotations

import argparse
from typing import Optional

from tetris.rl.ppo import PPOConfig, resolve_device, train_ppo


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Masked-Action PPO Dual-Stream CNN+MLP Tetris agent."
    )
    parser.add_argument("--updates", type=int, default=150, help="Number of PPO updates.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per env per update.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial Adam learning rate.")
    parser.add_argument("--eval-interval", type=int, default=25, help="Updates between eval runs.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Deterministic eval episodes.")
    parser.add_argument("--max-pieces", type=int, default=2500, help="Max pieces per train episode.")
    parser.add_argument("--eval-max-pieces", type=int, default=3000, help="Max pieces per eval episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, or cpu.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--plot-path", type=str, default="assets/training_curve.png", help="Output PNG curve path.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Training Tetris RL agent on device: {device}")
    config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        lr=args.lr,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_pieces_per_episode=args.max_pieces,
        eval_max_pieces=args.eval_max_pieces,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        plot_path=args.plot_path,
    )
    summary = train_ppo(config, device=device)
    print(
        f"Training complete. Best eval mean lines: {summary['best_eval_lines']:.1f} | "
        f"Checkpoint: {summary['best_checkpoint']} | Plot: {summary['plot_path']}"
    )


if __name__ == "__main__":
    main()
