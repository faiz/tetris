"""Gym-style single and vectorized Tetris RL environments over reachable placements."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tetris.engine import PlacementCandidate, TetrisGame
from tetris.pieces import BOARD_HEIGHT, BOARD_WIDTH
from tetris.rl.features import (
    NUM_FEATURES,
    compute_board_potential,
    extract_candidate_features,
)

MAX_CANDIDATES = 72
LINE_REWARD_BONUSES: dict[int, float] = {0: 0.0, 1: 0.4, 2: 1.0, 3: 2.2, 4: 4.5}


def encode_candidates(
    candidates: list[PlacementCandidate],
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack a list of PlacementCandidates into fixed-size (grids, feats, mask) arrays."""
    grids = np.zeros((max_candidates, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    feats = np.zeros((max_candidates, NUM_FEATURES), dtype=np.float32)
    mask = np.zeros((max_candidates,), dtype=bool)

    n = min(len(candidates), max_candidates)
    if n == 0:
        return grids, feats, mask

    boards = np.stack([c.after_board for c in candidates[:n]], axis=0)
    binary = (boards != 0).astype(np.uint8)
    lines = np.array([c.lines_cleared for c in candidates[:n]], dtype=np.float32)
    landing_h = np.array([c.landing_height for c in candidates[:n]], dtype=np.float32)

    grids[:n] = binary
    feats[:n] = extract_candidate_features(binary, lines, landing_h)
    mask[:n] = True
    return grids, feats, mask


class TetrisRLEnv:
    """Single-board placement-level Tetris RL environment with potential shaping."""

    def __init__(
        self,
        seed: Optional[int] = None,
        gamma: float = 0.99,
        max_pieces_per_episode: int = 2500,
    ) -> None:
        self.gamma = gamma
        self.max_pieces_per_episode = max_pieces_per_episode
        self.game = TetrisGame(seed=seed)
        self._candidates: list[PlacementCandidate] = []
        self._prev_potential: float = 0.0
        self.reset(seed=seed)

    def reset(
        self, seed: Optional[int] = None
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
        self.game.reset(seed=seed)
        self._candidates = self.game.get_valid_placements(allow_hold=True)
        self._prev_potential = compute_board_potential(self.game.board)
        obs = encode_candidates(self._candidates)
        return obs, {}

    def step(
        self, action_idx: int
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if not self._candidates:
            raise RuntimeError("Cannot step environment when no valid candidates exist; call reset().")
        if action_idx < 0 or action_idx >= len(self._candidates):
            raise IndexError(
                f"Action index {action_idx} out of range [0, {len(self._candidates) - 1}]"
            )

        cand = self._candidates[action_idx]
        lines_cleared = self.game.apply_placement(cand)

        next_candidates = (
            self.game.get_valid_placements(allow_hold=True)
            if not self.game.game_over
            else []
        )
        terminated = bool(self.game.game_over or len(next_candidates) == 0)
        if terminated:
            self.game.game_over = True

        truncated = (not terminated) and (
            self.game.pieces_placed >= self.max_pieces_per_episode
        )

        base_reward = 0.1 + LINE_REWARD_BONUSES[lines_cleared] + (-1.0 if terminated else 0.0)
        next_potential = (
            0.0 if terminated else compute_board_potential(self.game.board)
        )
        shaped_reward = float(
            base_reward + (self.gamma * next_potential - self._prev_potential)
        )
        self._prev_potential = next_potential

        info: dict[str, Any] = {}
        if terminated or truncated:
            tetris_rate = (
                (4.0 * self.game.tetris_count / self.game.lines)
                if self.game.lines > 0
                else 0.0
            )
            info["episode"] = {
                "lines": int(self.game.lines),
                "score": int(self.game.score),
                "pieces": int(self.game.pieces_placed),
                "tetris_count": int(self.game.tetris_count),
                "tetris_rate": float(tetris_rate),
                "level": int(self.game.level),
            }
            if truncated and next_candidates:
                obs = encode_candidates(next_candidates)
                info["terminal_observation"] = obs
                self._candidates = next_candidates
            else:
                self._candidates = []
                obs = encode_candidates([])
        else:
            self._candidates = next_candidates
            obs = encode_candidates(next_candidates)

        return obs, shaped_reward, terminated, truncated, info


class VecTetrisEnv:
    """Synchronous vectorized wrapper over multiple TetrisRLEnv instances with auto-reset."""

    def __init__(
        self,
        num_envs: int,
        base_seed: int = 0,
        gamma: float = 0.99,
        max_pieces_per_episode: int = 2500,
    ) -> None:
        self.num_envs = num_envs
        self.base_seed = base_seed
        self._episode_counters = [0] * num_envs
        self.envs = [
            TetrisRLEnv(
                seed=base_seed + i,
                gamma=gamma,
                max_pieces_per_episode=max_pieces_per_episode,
            )
            for i in range(num_envs)
        ]

    def _next_seed(self, env_idx: int) -> int:
        self._episode_counters[env_idx] += 1
        return self.base_seed + env_idx + self._episode_counters[env_idx] * 1009

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grids_list: list[np.ndarray] = []
        feats_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        for i, env in enumerate(self.envs):
            (g, f, m), _ = env.reset(seed=self.base_seed + i)
            grids_list.append(g)
            feats_list.append(f)
            masks_list.append(m)
        return (
            np.stack(grids_list, axis=0),
            np.stack(feats_list, axis=0),
            np.stack(masks_list, axis=0),
        )

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        grids = np.empty(
            (self.num_envs, MAX_CANDIDATES, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8
        )
        feats = np.empty(
            (self.num_envs, MAX_CANDIDATES, NUM_FEATURES), dtype=np.float32
        )
        masks = np.empty((self.num_envs, MAX_CANDIDATES), dtype=bool)
        rewards = np.empty((self.num_envs,), dtype=np.float32)
        terminateds = np.empty((self.num_envs,), dtype=bool)
        truncateds = np.empty((self.num_envs,), dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(int(actions[i]))
            rewards[i] = rew
            terminateds[i] = term
            truncateds[i] = trunc
            if term or trunc:
                obs, _ = env.reset(seed=self._next_seed(i))
            grids[i], feats[i], masks[i] = obs
            infos.append(info)

        return (grids, feats, masks), rewards, terminateds, truncateds, infos
