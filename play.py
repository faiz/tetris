"""CLI entrypoint for playing Tetris as a human, watching the RL agent, or playing Versus."""

from __future__ import annotations

import argparse
from typing import Optional

from tetris.ui_pygame import render_preview_image, run_game_loop


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Guideline Tetris or watch/challenge the trained RL agent."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ai",
        action="store_true",
        help="Watch the trained RL agent play Tetris live.",
    )
    mode_group.add_argument(
        "--versus",
        action="store_true",
        help="Split-screen Human (left) vs Trained RL Agent (right).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained RL checkpoint (default: checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=8.0,
        help="AI placement speed in pieces/second (default: 8.0).",
    )
    parser.add_argument(
        "--instant",
        action="store_true",
        help="Drop AI pieces immediately without step-by-step animation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for piece generation.",
    )
    parser.add_argument(
        "--render-preview",
        type=str,
        default=None,
        help="Render a single headless preview PNG and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.render_preview:
        out = render_preview_image(
            output_path=args.render_preview,
            checkpoint_path=args.checkpoint,
            seed=7 if args.seed is None else args.seed,
        )
        print(f"Saved preview image to {out}")
        return

    mode = "ai" if args.ai else ("versus" if args.versus else "human")
    run_game_loop(
        mode=mode,
        checkpoint_path=args.checkpoint,
        ai_speed=args.speed,
        instant=args.instant,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
