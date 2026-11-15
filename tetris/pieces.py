"""Tetromino definitions, SRS shapes, color palette, and 7-bag randomizer."""

from __future__ import annotations

import random
from typing import Optional

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
NUM_PIECES = 7

PIECE_NAMES: tuple[str, ...] = ("I", "O", "T", "S", "Z", "J", "L")

# Modern vibrant RGB colors per piece_id (0..6)
PIECE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 229, 255),    # I - Cyan
    1: (255, 213, 0),    # O - Yellow
    2: (178, 75, 243),   # T - Purple
    3: (56, 217, 114),   # S - Emerald Green
    4: (255, 71, 87),    # Z - Crimson Red
    5: (52, 130, 246),   # J - Cobalt Blue
    6: (255, 146, 43),   # L - Amber Orange
}

PIECE_BOX_WIDTH: dict[int, int] = {
    0: 4,
    1: 2,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
}

# Exact 4-state SRS (dr, dc) cell offsets
SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    # I (4x4)
    0: (
        ((1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 1), (1, 1), (2, 1), (3, 1)),
    ),
    # O (2x2)
    1: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    # T (3x3)
    2: (
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ),
    # S (3x3)
    3: (
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
    ),
    # Z (3x3)
    4: (
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 2), (1, 1), (1, 2), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    # J (3x3)
    5: (
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
    # L (3x3)
    6: (
        ((0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
    ),
}

# Deduplicated orientation indices for RL candidate enumeration
UNIQUE_ROTATIONS: dict[int, tuple[int, ...]] = {
    0: (0, 1),
    1: (0,),
    2: (0, 1, 2, 3),
    3: (0, 1),
    4: (0, 1),
    5: (0, 1, 2, 3),
    6: (0, 1, 2, 3),
}


def spawn_position(piece_id: int) -> tuple[int, int]:
    """Return canonical (spawn_x, spawn_y) for a piece_id."""
    spawn_x = (BOARD_WIDTH - PIECE_BOX_WIDTH[piece_id]) // 2
    spawn_y = -1 if piece_id == 0 else 0
    return spawn_x, spawn_y


def cell_shades(color: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return (highlight, shadow) RGB colors for bevelled block rendering."""
    highlight = tuple(min(255, int(c + (255 - c) * 0.35)) for c in color)
    shadow = tuple(max(0, int(c * 0.58)) for c in color)
    return highlight, shadow  # type: ignore[return-value]


class BagRandomizer:
    """Guideline 7-bag tetromino generator with lookahead."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._queue: list[int] = []
        self._ensure(14)

    def _refill_bag(self) -> None:
        bag = list(range(NUM_PIECES))
        self._rng.shuffle(bag)
        self._queue.extend(bag)

    def _ensure(self, count: int) -> None:
        while len(self._queue) < count:
            self._refill_bag()

    def next_piece(self) -> int:
        """Pop and return the next piece ID (0..6)."""
        self._ensure(8)
        return self._queue.pop(0)

    def peek(self, n: int = 3) -> list[int]:
        """Return the next n piece IDs without consuming them."""
        self._ensure(n)
        return list(self._queue[:n])

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset generator state with an optional new seed."""
        self._rng = random.Random(seed)
        self._queue.clear()
        self._ensure(14)

    def clone(self) -> BagRandomizer:
        """Return an independent copy preserving exact future sequence."""
        copy_rng = BagRandomizer.__new__(BagRandomizer)
        copy_rng._rng = random.Random()
        copy_rng._rng.setstate(self._rng.getstate())
        copy_rng._queue = list(self._queue)
        return copy_rng
