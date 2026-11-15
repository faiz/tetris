"""Unit and headless UI tests for Part 1: Human-Playable Tetris."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from play import main as play_main
from tetris.engine import LOCK_DELAY, TetrisGame, fits
from tetris.pieces import BOARD_HEIGHT, BOARD_WIDTH, SHAPES, BagRandomizer


def test_seven_bag_randomizer_fairness_and_peek() -> None:
    rng = BagRandomizer(seed=123)
    peeked = rng.peek(3)
    drawn = [rng.next_piece() for _ in range(14)]
    assert drawn[:3] == peeked
    assert sorted(drawn[:7]) == list(range(7))
    assert sorted(drawn[7:14]) == list(range(7))


def test_piece_movement_and_wall_kicks() -> None:
    game = TetrisGame(seed=0)
    # Move all the way to the left wall
    while game.move(-1):
        pass
    assert not game.move(-1)
    # Rotating against left/right wall should remain legal via kicks
    for _ in range(10):
        game.move(1)
    assert game.rotate(1)
    assert fits(game.board, game.current_cells(), game.piece_x, game.piece_y)


def test_line_clears_scoring_and_combos() -> None:
    game = TetrisGame(seed=5)
    # Force an O-piece on an almost-full bottom 2 rows to test double clear
    game.board[18:, :8] = 1
    game.current_piece = 1  # O piece
    game.rotation = 0
    game.piece_x = 8
    game.piece_y = 0
    cleared = game.hard_drop()
    assert cleared == 2
    assert game.lines == 2
    assert game.combo == 0
    assert game.score >= 300  # Double clear + hard drop points

    # Setup a Tetris (4-line clear) with an I-piece vertically in column 9
    game.board[16:20, :9] = 2
    game.current_piece = 0  # I piece
    game.rotation = 1  # vertical at dc=2
    game.piece_x = 7  # 7 + 2 = 9
    game.piece_y = 0
    cleared2 = game.hard_drop()
    assert cleared2 == 4
    assert game.tetris_count == 1
    assert game.combo == 1  # Back-to-back line clear increments combo
    assert game.lines == 6


def test_hold_lockout_per_piece() -> None:
    game = TetrisGame(seed=9)
    first = game.current_piece
    second = game.next_queue(1)[0]
    assert game.hold() is True
    assert game.hold_piece == first
    assert game.current_piece == second
    assert game.can_hold is False
    # Second hold before locking must be rejected
    assert game.hold() is False
    game.hard_drop()
    assert game.can_hold is True


def test_gravity_and_lock_delay_tick() -> None:
    game = TetrisGame(seed=11)
    initial_y = game.piece_y
    game.tick(game.gravity_interval() + 0.01)
    assert game.piece_y == initial_y + 1

    # Drop to floor via soft drops and verify lock delay commits the piece
    while game.soft_drop():
        pass
    pieces_before = game.pieces_placed
    game.tick(LOCK_DELAY + 0.05)
    assert game.pieces_placed == pieces_before + 1


def test_game_over_when_spawn_blocked() -> None:
    game = TetrisGame(seed=1)
    # Leave column 9 empty so locking doesn't clear rows 0..2
    game.board[0:3, :9] = 3
    game.hard_drop()
    assert game.game_over is True

    # Also test lock-above-skyline game over
    game2 = TetrisGame(seed=2)
    game2.board[0:2, :9] = 1
    game2.current_piece = 2  # T piece
    game2.piece_y = -1
    game2.hard_drop()
    assert game2.game_over is True


def test_placement_enumeration_and_step_plan_equivalence() -> None:
    game = TetrisGame(seed=21)
    # Place a few random obstacles on bottom rows
    game.board[18:, :4] = 1
    game.board[19, 6:] = 2
    candidates = game.get_valid_placements(allow_hold=True)
    assert len(candidates) > 0

    for cand in candidates[:12]:
        g_direct = game.clone()
        g_direct.apply_placement(cand)

        g_anim = game.clone()
        plan = g_anim.build_step_plan(cand)
        assert plan[-1] == "hard_drop"
        for act in plan:
            if act == "hold":
                g_anim.hold()
            elif act == "rot_cw":
                g_anim.rotate(1)
            elif act == "rot_ccw":
                g_anim.rotate(-1)
            elif act == "left":
                g_anim.move(-1)
            elif act == "right":
                g_anim.move(1)
            elif act == "hard_drop":
                g_anim.hard_drop()

        assert np.array_equal(g_direct.board, g_anim.board)
        assert g_direct.lines == g_anim.lines
        assert g_direct.pieces_placed == g_anim.pieces_placed


def test_headless_pygame_render_preview(tmp_path: Path) -> None:
    out_file = tmp_path / "human_preview.png"
    play_main(["--render-preview", str(out_file), "--checkpoint", str(tmp_path / "missing.pt")])
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
