"""Vectorized NumPy board feature extraction and potential-based reward shaping."""

from __future__ import annotations

import numpy as np

from tetris.pieces import BOARD_HEIGHT, BOARD_WIDTH

NUM_FEATURES = 12


def _compute_structural_stats(
    occupied: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized structural board metrics for a batch (N, 20, 10) bool array."""
    has_block = occupied.any(axis=1)
    first_occ_row = occupied.argmax(axis=1)
    col_heights = np.where(has_block, BOARD_HEIGHT - first_occ_row, 0)

    cum_occ = np.maximum.accumulate(occupied, axis=1)
    holes_mask = cum_occ & (~occupied)
    total_holes = holes_mask.sum(axis=(1, 2)).astype(np.float32)

    blocks_above = np.cumsum(occupied.astype(np.int32), axis=1)
    covered_weight = (blocks_above * holes_mask).sum(axis=(1, 2)).astype(np.float32)

    bumpiness = np.abs(np.diff(col_heights, axis=1)).sum(axis=1).astype(np.float32)
    sum_height = col_heights.sum(axis=1).astype(np.float32)
    return col_heights, total_holes, covered_weight, bumpiness, sum_height, occupied


def extract_candidate_features(
    boards: np.ndarray,
    lines_cleared: np.ndarray,
    landing_heights: np.ndarray,
) -> np.ndarray:
    """Compute (N, 12) normalized float32 feature matrix for N candidate afterstates."""
    n = boards.shape[0]
    if n == 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)

    occupied = boards != 0
    col_heights, total_holes, covered_weight, bumpiness, sum_height, _ = (
        _compute_structural_stats(occupied)
    )

    max_height = col_heights.max(axis=1).astype(np.float32)
    min_height = col_heights.min(axis=1).astype(np.float32)

    # Row transitions (walls outside left/right are solid True)
    padded_rows = np.pad(
        occupied, ((0, 0), (0, 0), (1, 1)), mode="constant", constant_values=True
    )
    row_transitions = (
        (padded_rows[:, :, 1:] != padded_rows[:, :, :-1])
        .sum(axis=(1, 2))
        .astype(np.float32)
    )

    # Column transitions (sky above row 0 is False, floor below row 19 is True)
    padded_cols = np.empty((n, BOARD_HEIGHT + 2, BOARD_WIDTH), dtype=bool)
    padded_cols[:, 0, :] = False
    padded_cols[:, 1:-1, :] = occupied
    padded_cols[:, -1, :] = True
    col_transitions = (
        (padded_cols[:, 1:, :] != padded_cols[:, :-1, :])
        .sum(axis=(1, 2))
        .astype(np.float32)
    )

    # Well sums (walls outside columns have height BOARD_HEIGHT)
    padded_h = np.pad(
        col_heights,
        ((0, 0), (1, 1)),
        mode="constant",
        constant_values=BOARD_HEIGHT,
    )
    neighbor_min = np.minimum(padded_h[:, :-2], padded_h[:, 2:])
    well_depth = np.maximum(0, neighbor_min - col_heights)
    well_sums = (well_depth * (well_depth + 1) // 2).sum(axis=1).astype(np.float32)

    lc = lines_cleared.astype(np.float32)
    lh = landing_heights.astype(np.float32)

    feats = np.stack(
        [
            lc / 4.0,
            (lines_cleared == 4).astype(np.float32),
            total_holes / 40.0,
            np.minimum(covered_weight, 100.0) / 100.0,
            bumpiness / 40.0,
            sum_height / 200.0,
            max_height / 20.0,
            (max_height - min_height) / 20.0,
            row_transitions / 100.0,
            col_transitions / 100.0,
            np.minimum(well_sums, 60.0) / 60.0,
            lh / 20.0,
        ],
        axis=1,
    )
    return feats.astype(np.float32, copy=False)


def compute_board_potential(board: np.ndarray) -> float:
    """Compute scalar potential phi(s) for potential-based reward shaping."""
    occupied = (board != 0)[np.newaxis, :, :]
    _, total_holes, covered_weight, bumpiness, sum_height, _ = (
        _compute_structural_stats(occupied)
    )
    holes = float(total_holes[0])
    covered = float(min(covered_weight[0], 50.0))
    bump = float(bumpiness[0])
    height = float(sum_height[0])
    return -(0.50 * holes + 0.15 * bump + 0.10 * height + 0.05 * covered) * 0.02
