"""Core Guideline Tetris engine for both human real-time play and AI placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tetris.pieces import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    SHAPES,
    UNIQUE_ROTATIONS,
    BagRandomizer,
    spawn_position,
)

LOCK_DELAY = 0.4
MAX_LOCK_RESETS = 15
LINE_SCORES: dict[int, int] = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
WALL_KICK_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (-2, 0),
    (2, 0),
    (0, -2),
)


def fits(
    board: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    px: int,
    py: int,
) -> bool:
    """Return True if all tetromino cells lie inside valid bounds and empty squares."""
    for dr, dc in cells:
        r = py + dr
        c = px + dc
        if c < 0 or c >= BOARD_WIDTH or r >= BOARD_HEIGHT:
            return False
        if r >= 0 and board[r, c] != 0:
            return False
    return True


def _clear_full_lines(board: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (updated_board, lines_cleared) with full rows removed."""
    full_mask = (board != 0).all(axis=1)
    cleared = int(full_mask.sum())
    if cleared == 0:
        return board, 0
    remaining = board[~full_mask]
    top_pad = np.zeros((cleared, BOARD_WIDTH), dtype=board.dtype)
    return np.vstack((top_pad, remaining)), cleared


@dataclass(slots=True)
class PlacementCandidate:
    """A reachable hard-drop placement and its resulting board state."""

    use_hold: bool
    piece_id: int
    rot: int
    target_x: int
    landing_y: int
    after_board: np.ndarray
    lines_cleared: int
    topped_out: bool
    landing_height: float


class TetrisGame:
    """Complete Tetris state machine supporting human controls and AI placements."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = BagRandomizer(seed)
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
        self.current_piece: int = 0
        self.piece_x: int = 0
        self.piece_y: int = 0
        self.rotation: int = 0
        self.hold_piece: Optional[int] = None
        self.can_hold: bool = True
        self.score: int = 0
        self.lines: int = 0
        self.level: int = 1
        self.pieces_placed: int = 0
        self.tetris_count: int = 0
        self.combo: int = -1
        self.game_over: bool = False
        self.gravity_timer: float = 0.0
        self.lock_timer: float = 0.0
        self.lock_resets: int = 0
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset board, stats, and piece sequence."""
        self.rng.reset(seed)
        self.board.fill(0)
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces_placed = 0
        self.tetris_count = 0
        self.combo = -1
        self.game_over = False
        self.gravity_timer = 0.0
        self.lock_timer = 0.0
        self.lock_resets = 0
        self._spawn_piece(self.rng.next_piece())

    def clone(self) -> TetrisGame:
        """Create a deep copy of the current game state."""
        copy_game = TetrisGame.__new__(TetrisGame)
        copy_game.rng = self.rng.clone()
        copy_game.board = self.board.copy()
        copy_game.current_piece = self.current_piece
        copy_game.piece_x = self.piece_x
        copy_game.piece_y = self.piece_y
        copy_game.rotation = self.rotation
        copy_game.hold_piece = self.hold_piece
        copy_game.can_hold = self.can_hold
        copy_game.score = self.score
        copy_game.lines = self.lines
        copy_game.level = self.level
        copy_game.pieces_placed = self.pieces_placed
        copy_game.tetris_count = self.tetris_count
        copy_game.combo = self.combo
        copy_game.game_over = self.game_over
        copy_game.gravity_timer = self.gravity_timer
        copy_game.lock_timer = self.lock_timer
        copy_game.lock_resets = self.lock_resets
        return copy_game

    def current_cells(
        self,
        piece_id: Optional[int] = None,
        rotation: Optional[int] = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return relative (dr, dc) cells for the active or specified piece/rotation."""
        pid = self.current_piece if piece_id is None else piece_id
        rot = self.rotation if rotation is None else rotation
        return SHAPES[pid][rot]

    def next_queue(self, n: int = 3) -> list[int]:
        """Return the upcoming n piece IDs."""
        return self.rng.peek(n)

    def gravity_interval(self) -> float:
        """Seconds per gravity row drop for the current level."""
        return max(0.06, 0.8 * (0.85 ** (self.level - 1)))

    def _is_grounded(self) -> bool:
        return not fits(self.board, self.current_cells(), self.piece_x, self.piece_y + 1)

    def _reset_lock_delay_if_grounded(self) -> None:
        if self._is_grounded() and self.lock_resets < MAX_LOCK_RESETS:
            self.lock_timer = 0.0
            self.lock_resets += 1

    def _spawn_piece(self, piece_id: int) -> bool:
        self.current_piece = piece_id
        self.rotation = 0
        self.piece_x, self.piece_y = spawn_position(piece_id)
        self.gravity_timer = 0.0
        self.lock_timer = 0.0
        self.lock_resets = 0
        if not fits(self.board, self.current_cells(), self.piece_x, self.piece_y):
            self.game_over = True
            return False
        return True

    def move(self, dx: int) -> bool:
        """Move active piece horizontally by dx (-1 or +1)."""
        if self.game_over:
            return False
        if fits(self.board, self.current_cells(), self.piece_x + dx, self.piece_y):
            self.piece_x += dx
            self._reset_lock_delay_if_grounded()
            return True
        return False

    def rotate(self, direction: int = 1) -> bool:
        """Rotate active piece (+1 CW, -1 CCW) with simple wall-kick offsets."""
        if self.game_over:
            return False
        new_rot = (self.rotation + direction) % 4
        cells = self.current_cells(self.current_piece, new_rot)
        for kx, ky in WALL_KICK_OFFSETS:
            nx = self.piece_x + kx
            ny = self.piece_y + ky
            if fits(self.board, cells, nx, ny):
                self.rotation = new_rot
                self.piece_x = nx
                self.piece_y = ny
                self._reset_lock_delay_if_grounded()
                return True
        return False

    def soft_drop(self) -> bool:
        """Advance piece down 1 row and award +1 point."""
        if self.game_over:
            return False
        if fits(self.board, self.current_cells(), self.piece_x, self.piece_y + 1):
            self.piece_y += 1
            self.score += 1
            self.gravity_timer = 0.0
            self.lock_timer = 0.0
            return True
        return False

    def get_ghost_y(self) -> int:
        """Return the landing y coordinate for a hard drop from the current position."""
        gy = self.piece_y
        cells = self.current_cells()
        while fits(self.board, cells, self.piece_x, gy + 1):
            gy += 1
        return gy

    def hard_drop(self) -> int:
        """Drop active piece immediately to the floor, lock it, and return lines cleared."""
        if self.game_over:
            return 0
        gy = self.get_ghost_y()
        drop_dist = max(0, gy - self.piece_y)
        self.piece_y = gy
        self.score += 2 * drop_dist
        return self._lock_piece()

    def hold(self) -> bool:
        """Swap active piece with hold slot (once per piece lock)."""
        if self.game_over or not self.can_hold:
            return False
        previous = self.current_piece
        if self.hold_piece is None:
            self.hold_piece = previous
            next_piece = self.rng.next_piece()
        else:
            next_piece = self.hold_piece
            self.hold_piece = previous
        self.can_hold = False
        self._spawn_piece(next_piece)
        return True

    def _lock_piece(self) -> int:
        cells = self.current_cells()
        topped_out = False
        for dr, dc in cells:
            r = self.piece_y + dr
            c = self.piece_x + dc
            if r < 0:
                topped_out = True
            elif 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                self.board[r, c] = self.current_piece + 1

        self.pieces_placed += 1

        if topped_out:
            self.game_over = True
            return 0

        self.board, cleared = _clear_full_lines(self.board)
        if cleared > 0:
            self.lines += cleared
            self.level = 1 + self.lines // 10
            if cleared == 4:
                self.tetris_count += 1
            self.combo += 1
            combo_bonus = 50 * self.combo * self.level if self.combo > 0 else 0
            self.score += LINE_SCORES[cleared] * self.level + combo_bonus
        else:
            self.combo = -1

        self.can_hold = True
        self._spawn_piece(self.rng.next_piece())
        return cleared

    def tick(self, dt: float) -> None:
        """Advance gravity and lock delay by dt seconds."""
        if self.game_over or dt <= 0.0:
            return
        interval = self.gravity_interval()
        self.gravity_timer += dt
        cells = self.current_cells()
        while self.gravity_timer >= interval and not self.game_over:
            self.gravity_timer -= interval
            if fits(self.board, cells, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
                self.lock_timer = 0.0
            else:
                self.gravity_timer = 0.0
                break

        if self._is_grounded():
            self.lock_timer += dt
            if self.lock_timer >= LOCK_DELAY:
                self._lock_piece()
        else:
            self.lock_timer = 0.0

    def _enumerate_for_piece(
        self,
        piece_id: int,
        use_hold: bool,
    ) -> list[PlacementCandidate]:
        candidates: list[PlacementCandidate] = []
        sx, sy = spawn_position(piece_id)
        board = self.board
        color_val = piece_id + 1

        for rot in UNIQUE_ROTATIONS[piece_id]:
            cells = SHAPES[piece_id][rot]
            if fits(board, cells, sx, sy):
                y0 = sy
            elif fits(board, cells, sx, sy - 1):
                y0 = sy - 1
            else:
                continue

            xs: list[int] = []
            tx = sx
            while fits(board, cells, tx, y0):
                xs.append(tx)
                tx -= 1
            tx = sx + 1
            while fits(board, cells, tx, y0):
                xs.append(tx)
                tx += 1

            for target_x in xs:
                ly = y0
                while fits(board, cells, target_x, ly + 1):
                    ly += 1
                topped_out = any(ly + dr < 0 for dr, _ in cells)
                landing_height = BOARD_HEIGHT - sum(ly + dr for dr, _ in cells) / 4.0

                after = board.copy()
                for dr, dc in cells:
                    r = ly + dr
                    c = target_x + dc
                    if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                        after[r, c] = color_val

                if not topped_out:
                    after, cleared = _clear_full_lines(after)
                else:
                    cleared = 0

                candidates.append(
                    PlacementCandidate(
                        use_hold=use_hold,
                        piece_id=piece_id,
                        rot=rot,
                        target_x=target_x,
                        landing_y=ly,
                        after_board=after,
                        lines_cleared=cleared,
                        topped_out=topped_out,
                        landing_height=landing_height,
                    )
                )

        return candidates

    def get_valid_placements(self, allow_hold: bool = True) -> list[PlacementCandidate]:
        """Enumerate all reachable placements for the current piece (and hold if enabled)."""
        if self.game_over:
            return []

        candidates = self._enumerate_for_piece(self.current_piece, use_hold=False)
        if allow_hold and self.can_hold:
            hold_target = (
                self.hold_piece if self.hold_piece is not None else self.rng.peek(1)[0]
            )
            if self.hold_piece is None or hold_target != self.current_piece:
                # Only allow hold placements if the held piece can actually spawn at rot 0
                hx, hy = spawn_position(hold_target)
                if fits(self.board, SHAPES[hold_target][0], hx, hy):
                    candidates.extend(self._enumerate_for_piece(hold_target, use_hold=True))
        return candidates

    def apply_placement(self, candidate: PlacementCandidate) -> int:
        """Directly execute a PlacementCandidate and return lines cleared."""
        if self.game_over:
            return 0
        if candidate.use_hold:
            self.hold()
            if self.game_over:
                return 0
        self.rotation = candidate.rot
        self.piece_x = candidate.target_x
        drop_dist = max(0, candidate.landing_y - self.piece_y)
        self.piece_y = candidate.landing_y
        self.score += 2 * drop_dist
        return self._lock_piece()

    def build_step_plan(self, candidate: PlacementCandidate) -> list[str]:
        """Return primitive UI actions ('hold', 'rot_cw', 'left', ..., 'hard_drop') for animation."""
        sim = self.clone()
        actions: list[str] = []
        if candidate.use_hold:
            if sim.hold():
                actions.append("hold")
            if sim.game_over:
                return actions

        rot_diff = (candidate.rot - sim.rotation) % 4
        if rot_diff == 1:
            if sim.rotate(1):
                actions.append("rot_cw")
        elif rot_diff == 2:
            for _ in range(2):
                if sim.rotate(1):
                    actions.append("rot_cw")
        elif rot_diff == 3:
            if sim.rotate(-1):
                actions.append("rot_ccw")

        dx = candidate.target_x - sim.piece_x
        step = -1 if dx < 0 else 1
        name = "left" if dx < 0 else "right"
        for _ in range(abs(dx)):
            if sim.move(step):
                actions.append(name)
            else:
                break

        actions.append("hard_drop")
        return actions
