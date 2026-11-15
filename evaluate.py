"""CLI entrypoint for benchmarking a trained Tetris RL checkpoint."""

from __future__ import annotations

import argparse
from typing import Optional

from tetris.rl.network import load_checkpoint
from tetris.rl.ppo import evaluate_policy, resolve_device


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a trained Tetris RL checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of seeded eval episodes.")
    parser.add_argument("--max-pieces", type=int, default=5000, help="Max pieces per episode.")
    parser.add_argument("--seed", type=int, default=10000, help="Base seed for eval episodes.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, mps, or cpu.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    stats = evaluate_policy(
        model=model,
        num_episodes=args.episodes,
        max_pieces=args.max_pieces,
        base_seed=args.seed,
        device=device,
    )

    print(f"Checkpoint : {args.checkpoint}")
    if metadata:
        print(
            f"Trained at : update={metadata.get('update', '?')}, "
            f"steps={metadata.get('global_step', '?')}, device={metadata.get('device', '?')}"
        )
    print(f"Eval Device: {device} | Episodes: {stats['num_episodes']} (max_pieces={args.max_pieces})")
    print("-" * 64)
    print(
        f"Lines Cleared : mean={stats['mean_lines']:.1f}, median={stats['median_lines']:.1f}, "
        f"std={stats['std_lines']:.1f}, min={stats['min_lines']}, max={stats['max_lines']}"
    )
    print(
        f"Official Score: mean={stats['mean_score']:.0f}, median={stats['median_score']:.0f}, "
        f"max={stats['max_score']}"
    )
    print(f"Pieces Placed : mean={stats['mean_pieces']:.1f}")
    print(f"Tetris Rate   : {stats['mean_tetris_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
