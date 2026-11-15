"""Pygame-CE UI supporting Human Play, AI Watch, and Human-vs-AI Split-Screen."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame

from tetris.engine import PlacementCandidate, TetrisGame
from tetris.pieces import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_BOX_WIDTH,
    PIECE_COLORS,
    SHAPES,
    cell_shades,
)

# Visual palette (GitHub Dark Slate)
BG_COLOR = (13, 17, 23)
PANEL_BG = (22, 27, 34)
PANEL_BORDER = (48, 54, 61)
BOARD_BG = (16, 20, 28)
GRID_COLOR = (30, 36, 48)
TEXT_PRIMARY = (240, 246, 252)
TEXT_MUTED = (139, 148, 158)
ACCENT_CYAN = (56, 189, 248)
ACCENT_GREEN = (56, 217, 114)
ACCENT_GOLD = (255, 213, 0)
ACCENT_RED = (255, 71, 87)

CELL_SIZE = 30
BOARD_PX_W = BOARD_WIDTH * CELL_SIZE
BOARD_PX_H = BOARD_HEIGHT * CELL_SIZE
SIDEBAR_W = 200
VIEW_GAP = 18
VIEW_W = BOARD_PX_W + VIEW_GAP + SIDEBAR_W
OUTER_PAD = 24
HEADER_H = 54
VERSUS_GAP = 36

DAS_DELAY = 0.15
ARR_INTERVAL = 0.04


@dataclass
class AutoRepeatKey:
    """Tracks DAS + ARR auto-repeat timing for a held movement key."""

    active: bool = False
    elapsed: float = 0.0
    das_triggered: bool = False

    def press(self) -> bool:
        self.active = True
        self.elapsed = 0.0
        self.das_triggered = False
        return True

    def release(self) -> None:
        self.active = False
        self.elapsed = 0.0
        self.das_triggered = False

    def update(self, dt: float) -> int:
        if not self.active:
            return 0
        self.elapsed += dt
        triggers = 0
        if not self.das_triggered:
            if self.elapsed >= DAS_DELAY:
                self.elapsed -= DAS_DELAY
                self.das_triggered = True
                triggers += 1
        if self.das_triggered:
            while self.elapsed >= ARR_INTERVAL:
                self.elapsed -= ARR_INTERVAL
                triggers += 1
        return triggers


def _draw_bevel_cell(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    ghost: bool = False,
    dimmed: bool = False,
) -> None:
    if dimmed:
        color = tuple(max(30, int(c * 0.45)) for c in color)  # type: ignore[assignment]

    if ghost:
        inner = rect.inflate(-2, -2)
        ghost_bg = tuple(max(18, int(c * 0.18)) for c in color)
        pygame.draw.rect(surface, ghost_bg, inner, border_radius=4)
        pygame.draw.rect(surface, color, inner, width=2, border_radius=4)
        return

    highlight, shadow = cell_shades(color)
    pygame.draw.rect(surface, color, rect)
    # Top and left highlights
    pygame.draw.line(surface, highlight, rect.topleft, (rect.right - 2, rect.top), 2)
    pygame.draw.line(surface, highlight, rect.topleft, (rect.left, rect.bottom - 2), 2)
    # Bottom and right shadows
    pygame.draw.line(surface, shadow, (rect.left + 1, rect.bottom - 2), (rect.right - 1, rect.bottom - 2), 2)
    pygame.draw.line(surface, shadow, (rect.right - 2, rect.top + 1), (rect.right - 2, rect.bottom - 1), 2)
    # Subtle outer outline
    pygame.draw.rect(surface, (12, 15, 20), rect, width=1)


def _draw_mini_piece(
    surface: pygame.Surface,
    piece_id: Optional[int],
    center_x: int,
    center_y: int,
    cell_px: int = 18,
    dimmed: bool = False,
) -> None:
    if piece_id is None:
        return
    cells = SHAPES[piece_id][0]
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    span_w = (max_c - min_c + 1) * cell_px
    span_h = (max_r - min_r + 1) * cell_px
    start_x = center_x - span_w // 2
    start_y = center_y - span_h // 2
    color = PIECE_COLORS[piece_id]
    for dr, dc in cells:
        rx = start_x + (dc - min_c) * cell_px
        ry = start_y + (dr - min_r) * cell_px
        _draw_bevel_cell(surface, pygame.Rect(rx, ry, cell_px, cell_px), color, dimmed=dimmed)


def _load_font(size: int, bold: bool = False) -> pygame.font.Font:
    for name in ("menlo", "dejavusansmono", "consolas", "arial"):
        if pygame.font.match_font(name):
            return pygame.font.SysFont(name, size, bold=bold)
    return pygame.font.Font(None, size)


class BoardView:
    """Renders a single Tetris board and its sidebar HUD at (origin_x, origin_y)."""

    def __init__(self) -> None:
        self.font_title = _load_font(20, bold=True)
        self.font_label = _load_font(12, bold=True)
        self.font_value = _load_font(20, bold=True)
        self.font_small = _load_font(11)
        self.font_overlay = _load_font(26, bold=True)

    def draw(
        self,
        surface: pygame.Surface,
        game: TetrisGame,
        origin_x: int,
        origin_y: int,
        title: str,
        badge_text: str,
        badge_color: tuple[int, int, int],
        high_score: int = 0,
        ai_value: Optional[float] = None,
        ai_speed: Optional[float] = None,
        paused: bool = False,
        is_ai: bool = False,
    ) -> None:
        # Top banner above board
        title_surf = self.font_title.render(title, True, TEXT_PRIMARY)
        surface.blit(title_surf, (origin_x, origin_y - 34))

        badge_surf = self.font_label.render(badge_text, True, BG_COLOR)
        badge_rect = pygame.Rect(
            origin_x + BOARD_PX_W - badge_surf.get_width() - 16,
            origin_y - 32,
            badge_surf.get_width() + 16,
            22,
        )
        pygame.draw.rect(surface, badge_color, badge_rect, border_radius=6)
        surface.blit(badge_surf, (badge_rect.x + 8, badge_rect.y + 4))

        # Board frame & background
        board_rect = pygame.Rect(origin_x, origin_y, BOARD_PX_W, BOARD_PX_H)
        pygame.draw.rect(surface, BOARD_BG, board_rect)

        for r in range(1, BOARD_HEIGHT):
            y = origin_y + r * CELL_SIZE
            pygame.draw.line(surface, GRID_COLOR, (origin_x, y), (origin_x + BOARD_PX_W, y))
        for c in range(1, BOARD_WIDTH):
            x = origin_x + c * CELL_SIZE
            pygame.draw.line(surface, GRID_COLOR, (x, origin_y), (x, origin_y + BOARD_PX_H))

        # Locked cells
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                val = int(game.board[r, c])
                if val > 0:
                    rect = pygame.Rect(
                        origin_x + c * CELL_SIZE,
                        origin_y + r * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    )
                    _draw_bevel_cell(surface, rect, PIECE_COLORS[val - 1])

        # Ghost & active piece
        if not game.game_over:
            cells = game.current_cells()
            color = PIECE_COLORS[game.current_piece]
            ghost_y = game.get_ghost_y()
            if ghost_y != game.piece_y:
                for dr, dc in cells:
                    r = ghost_y + dr
                    c = game.piece_x + dc
                    if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                        rect = pygame.Rect(
                            origin_x + c * CELL_SIZE,
                            origin_y + r * CELL_SIZE,
                            CELL_SIZE,
                            CELL_SIZE,
                        )
                        _draw_bevel_cell(surface, rect, color, ghost=True)

            for dr, dc in cells:
                r = game.piece_y + dr
                c = game.piece_x + dc
                if 0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH:
                    rect = pygame.Rect(
                        origin_x + c * CELL_SIZE,
                        origin_y + r * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    )
                    _draw_bevel_cell(surface, rect, color)

        pygame.draw.rect(surface, PANEL_BORDER, board_rect, width=2)

        # Sidebar HUD
        sx = origin_x + BOARD_PX_W + VIEW_GAP
        sy = origin_y

        # HOLD card
        hold_rect = pygame.Rect(sx, sy, SIDEBAR_W, 92)
        self._draw_card(surface, hold_rect, "HOLD")
        _draw_mini_piece(
            surface,
            game.hold_piece,
            hold_rect.centerx,
            hold_rect.y + 56,
            dimmed=not game.can_hold,
        )

        # NEXT card (3 pieces)
        next_rect = pygame.Rect(sx, sy + 104, SIDEBAR_W, 198)
        self._draw_card(surface, next_rect, "NEXT")
        for idx, pid in enumerate(game.next_queue(3)):
            _draw_mini_piece(
                surface,
                pid,
                next_rect.centerx,
                next_rect.y + 56 + idx * 54,
                cell_px=16,
            )

        # STATS card
        stats_rect = pygame.Rect(sx, sy + 314, SIDEBAR_W, 178)
        self._draw_card(surface, stats_rect, "STATS")
        rows = [
            ("SCORE", f"{game.score:,}", ACCENT_CYAN),
            ("BEST", f"{max(high_score, game.score):,}", TEXT_MUTED),
            ("LINES", f"{game.lines}", TEXT_PRIMARY),
            ("LEVEL", f"{game.level}", ACCENT_GOLD),
            ("TETRIS", f"{game.tetris_count}", ACCENT_GREEN),
        ]
        for i, (label, val_str, col) in enumerate(rows):
            ry = stats_rect.y + 34 + i * 27
            lbl_s = self.font_label.render(label, True, TEXT_MUTED)
            val_s = self.font_label.render(val_str, True, col)
            surface.blit(lbl_s, (stats_rect.x + 14, ry))
            surface.blit(val_s, (stats_rect.right - val_s.get_width() - 14, ry))

        # CONTROLS / AI card
        info_rect = pygame.Rect(sx, sy + 504, SIDEBAR_W, 96)
        if is_ai:
            self._draw_card(surface, info_rect, "RL AGENT")
            val_txt = "V(s): --" if ai_value is None else f"V(s): {ai_value:+.2f}"
            spd_txt = "Speed: --" if ai_speed is None else f"Speed: {ai_speed:.1f} p/s"
            lines_info = [
                (val_txt, ACCENT_GREEN),
                (spd_txt, TEXT_PRIMARY),
                ("[ / ] Speed   P Pause", TEXT_MUTED),
            ]
        else:
            self._draw_card(surface, info_rect, "CONTROLS")
            lines_info = [
                ("←/→ Move   ↑/Z Rotate", TEXT_MUTED),
                ("↓ Soft     Space Drop", TEXT_MUTED),
                ("C Hold     P/R Pause/Reset", TEXT_MUTED),
            ]
        for i, (txt, col) in enumerate(lines_info):
            s = self.font_small.render(txt, True, col)
            surface.blit(s, (info_rect.x + 12, info_rect.y + 34 + i * 20))

        # Overlay on pause or game over
        if paused or game.game_over:
            overlay = pygame.Surface((BOARD_PX_W, BOARD_PX_H), pygame.SRCALPHA)
            overlay.fill((10, 12, 18, 175))
            surface.blit(overlay, (origin_x, origin_y))

            heading = "GAME OVER" if game.game_over else "PAUSED"
            h_color = ACCENT_RED if game.game_over else ACCENT_CYAN
            h_surf = self.font_overlay.render(heading, True, h_color)
            surface.blit(
                h_surf,
                (
                    origin_x + (BOARD_PX_W - h_surf.get_width()) // 2,
                    origin_y + BOARD_PX_H // 2 - 36,
                ),
            )
            sub_txt = "Press R to Restart" if game.game_over else "Press P to Resume"
            sub_surf = self.font_label.render(sub_txt, True, TEXT_PRIMARY)
            surface.blit(
                sub_surf,
                (
                    origin_x + (BOARD_PX_W - sub_surf.get_width()) // 2,
                    origin_y + BOARD_PX_H // 2 + 6,
                ),
            )

    def _draw_card(self, surface: pygame.Surface, rect: pygame.Rect, heading: str) -> None:
        pygame.draw.rect(surface, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=8)
        h_surf = self.font_label.render(heading, True, TEXT_MUTED)
        surface.blit(h_surf, (rect.x + 12, rect.y + 10))


class AIController:
    """Lazy-loading RL policy driver that plans and executes moves on a TetrisGame."""

    def __init__(self, checkpoint_path: str | Path, speed: float = 8.0, instant: bool = False) -> None:
        import numpy as np
        import torch

        from tetris.rl.env import MAX_CANDIDATES, encode_candidates
        from tetris.rl.network import load_checkpoint

        self._np = np
        self._torch = torch
        self._encode_candidates = encode_candidates
        self._max_candidates = MAX_CANDIDATES

        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"RL checkpoint not found at '{ckpt_file}'. "
                "Train a model first with: uv run tetris-train"
            )
        self.model, self.metadata = load_checkpoint(ckpt_file, device="cpu")
        self.model.eval()
        self.speed = max(1.0, min(25.0, float(speed)))
        self.instant = instant
        self.action_queue: list[str] = []
        self.timer: float = 0.0
        self.last_value: Optional[float] = None

    def reset(self) -> None:
        self.action_queue.clear()
        self.timer = 0.0
        self.last_value = None

    def _choose_candidate(self, game: TetrisGame) -> Optional[PlacementCandidate]:
        candidates = game.get_valid_placements(allow_hold=True)
        if not candidates:
            return None
        grids, feats, mask = self._encode_candidates(candidates, self._max_candidates)
        with self._torch.no_grad():
            g_t = self._torch.from_numpy(grids).unsqueeze(0).float()
            f_t = self._torch.from_numpy(feats).unsqueeze(0)
            m_t = self._torch.from_numpy(mask).unsqueeze(0)
            action, value = self.model.select_action(g_t, f_t, m_t, deterministic=True)
        idx = int(action.item())
        self.last_value = float(value.item())
        return candidates[idx]

    def update(self, game: TetrisGame, dt: float) -> None:
        if game.game_over:
            return
        self.timer += dt
        if self.instant:
            interval = 1.0 / self.speed
            while self.timer >= interval and not game.game_over:
                self.timer -= interval
                cand = self._choose_candidate(game)
                if cand is None:
                    game.game_over = True
                    break
                game.apply_placement(cand)
            return

        step_interval = 1.0 / (self.speed * 4.5)
        while self.timer >= step_interval and not game.game_over:
            self.timer -= step_interval
            if not self.action_queue:
                cand = self._choose_candidate(game)
                if cand is None:
                    game.game_over = True
                    break
                self.action_queue = game.build_step_plan(cand)
                if not self.action_queue:
                    game.apply_placement(cand)
                    continue

            act = self.action_queue.pop(0)
            if act == "hold":
                game.hold()
            elif act == "rot_cw":
                game.rotate(1)
            elif act == "rot_ccw":
                game.rotate(-1)
            elif act == "left":
                game.move(-1)
            elif act == "right":
                game.move(1)
            elif act == "hard_drop":
                game.hard_drop()


def _create_sample_human_game(seed: int = 42) -> TetrisGame:
    """Advance a deterministic game a few moves for clean preview screenshots."""
    game = TetrisGame(seed=seed)
    scripted = [
        (0, -3),
        (1, -1),
        (0, 2),
        (2, 0),
        (0, 3),
        (1, -4),
    ]
    for rots, dx in scripted:
        if game.game_over:
            break
        for _ in range(rots):
            game.rotate(1)
        step = -1 if dx < 0 else 1
        for _ in range(abs(dx)):
            game.move(step)
        game.hard_drop()
    game.hold()
    game.move(-1)
    return game


def render_preview_image(
    output_path: str | Path,
    checkpoint_path: Optional[str | Path] = None,
    seed: int = 7,
) -> Path:
    """Render a headless mid-game screenshot (Versus if checkpoint exists, else Human)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    try:
        view = BoardView()
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        use_versus = ckpt is not None and ckpt.exists()

        if use_versus and ckpt is not None:
            width = OUTER_PAD * 2 + VIEW_W * 2 + VERSUS_GAP
            height = HEADER_H + BOARD_PX_H + OUTER_PAD
            surface = pygame.Surface((width, height))
            surface.fill(BG_COLOR)

            human_game = _create_sample_human_game(seed=seed)
            ai_game = TetrisGame(seed=seed)
            ai = AIController(ckpt, speed=10.0, instant=True)
            for _ in range(36):
                if ai_game.game_over:
                    break
                cand = ai._choose_candidate(ai_game)
                if cand is None:
                    break
                ai_game.apply_placement(cand)

            view.draw(
                surface,
                human_game,
                origin_x=OUTER_PAD,
                origin_y=HEADER_H,
                title="HUMAN PLAYER",
                badge_text="PLAYER 1",
                badge_color=ACCENT_CYAN,
                high_score=human_game.score,
            )
            divider_x = OUTER_PAD + VIEW_W + VERSUS_GAP // 2
            pygame.draw.line(
                surface,
                PANEL_BORDER,
                (divider_x, HEADER_H - 12),
                (divider_x, HEADER_H + BOARD_PX_H),
                width=2,
            )
            view.draw(
                surface,
                ai_game,
                origin_x=OUTER_PAD + VIEW_W + VERSUS_GAP,
                origin_y=HEADER_H,
                title="RL AGENT (PPO)",
                badge_text="APPLE MPS",
                badge_color=ACCENT_GREEN,
                high_score=ai_game.score,
                ai_value=ai.last_value,
                ai_speed=ai.speed,
                is_ai=True,
            )
        else:
            width = OUTER_PAD * 2 + VIEW_W
            height = HEADER_H + BOARD_PX_H + OUTER_PAD
            surface = pygame.Surface((width, height))
            surface.fill(BG_COLOR)
            human_game = _create_sample_human_game(seed=seed)
            view.draw(
                surface,
                human_game,
                origin_x=OUTER_PAD,
                origin_y=HEADER_H,
                title="TETRIS",
                badge_text="HUMAN",
                badge_color=ACCENT_CYAN,
                high_score=human_game.score,
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(out))
        return out
    finally:
        pygame.quit()


def run_game_loop(
    mode: str = "human",
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    ai_speed: float = 8.0,
    instant: bool = False,
    seed: Optional[int] = None,
) -> None:
    """Launch interactive 60 FPS Pygame window for 'human', 'ai', or 'versus' mode."""
    pygame.init()
    try:
        is_versus = mode == "versus"
        is_ai_only = mode == "ai"

        width = (
            OUTER_PAD * 2 + VIEW_W * 2 + VERSUS_GAP
            if is_versus
            else OUTER_PAD * 2 + VIEW_W
        )
        height = HEADER_H + BOARD_PX_H + OUTER_PAD
        screen = pygame.display.set_mode((width, height))

        captions = {
            "human": "Tetris — Human Mode",
            "ai": "Tetris — RL Agent Watch",
            "versus": "Tetris — Human vs RL Agent",
        }
        pygame.display.set_caption(captions.get(mode, "Tetris"))

        view = BoardView()
        clock = pygame.time.Clock()

        human_game = TetrisGame(seed=seed)
        ai_game = TetrisGame(seed=seed) if (is_ai_only or is_versus) else None
        ai_ctrl = (
            AIController(checkpoint_path=checkpoint_path, speed=ai_speed, instant=instant)
            if (is_ai_only or is_versus)
            else None
        )

        human_best = 0
        ai_best = 0
        paused = False

        left_repeat = AutoRepeatKey()
        right_repeat = AutoRepeatKey()
        down_repeat = AutoRepeatKey()

        running = True
        while running:
            dt = min(0.05, clock.tick(60) / 1000.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_ESCAPE:
                        running = False
                    elif key == pygame.K_p:
                        paused = not paused
                    elif key == pygame.K_r:
                        paused = False
                        human_best = max(human_best, human_game.score)
                        human_game.reset(seed=None)
                        if ai_game is not None and ai_ctrl is not None:
                            ai_best = max(ai_best, ai_game.score)
                            ai_game.reset(seed=None)
                            ai_ctrl.reset()
                    elif ai_ctrl is not None and key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                        ai_ctrl.speed = max(1.0, ai_ctrl.speed - 1.0)
                    elif ai_ctrl is not None and key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                        ai_ctrl.speed = min(25.0, ai_ctrl.speed + 1.0)
                    elif not paused and not is_ai_only:
                        if key in (pygame.K_LEFT, pygame.K_a):
                            left_repeat.press()
                            right_repeat.release()
                            human_game.move(-1)
                        elif key in (pygame.K_RIGHT, pygame.K_d):
                            right_repeat.press()
                            left_repeat.release()
                            human_game.move(1)
                        elif key in (pygame.K_DOWN, pygame.K_s):
                            down_repeat.press()
                            human_game.soft_drop()
                        elif key in (pygame.K_UP, pygame.K_w, pygame.K_x):
                            human_game.rotate(1)
                        elif key == pygame.K_z:
                            human_game.rotate(-1)
                        elif key == pygame.K_SPACE:
                            human_game.hard_drop()
                        elif key in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                            human_game.hold()

                elif event.type == pygame.KEYUP:
                    key = event.key
                    if key in (pygame.K_LEFT, pygame.K_a):
                        left_repeat.release()
                    elif key in (pygame.K_RIGHT, pygame.K_d):
                        right_repeat.release()
                    elif key in (pygame.K_DOWN, pygame.K_s):
                        down_repeat.release()

            if not paused:
                if not is_ai_only:
                    for _ in range(left_repeat.update(dt)):
                        human_game.move(-1)
                    for _ in range(right_repeat.update(dt)):
                        human_game.move(1)
                    for _ in range(down_repeat.update(dt)):
                        human_game.soft_drop()
                    human_game.tick(dt)
                    human_best = max(human_best, human_game.score)

                if ai_game is not None and ai_ctrl is not None:
                    ai_ctrl.update(ai_game, dt)
                    ai_best = max(ai_best, ai_game.score)

            screen.fill(BG_COLOR)
            if is_versus and ai_game is not None and ai_ctrl is not None:
                view.draw(
                    screen,
                    human_game,
                    origin_x=OUTER_PAD,
                    origin_y=HEADER_H,
                    title="HUMAN PLAYER",
                    badge_text="PLAYER 1",
                    badge_color=ACCENT_CYAN,
                    high_score=human_best,
                    paused=paused,
                )
                divider_x = OUTER_PAD + VIEW_W + VERSUS_GAP // 2
                pygame.draw.line(
                    screen,
                    PANEL_BORDER,
                    (divider_x, HEADER_H - 12),
                    (divider_x, HEADER_H + BOARD_PX_H),
                    width=2,
                )
                view.draw(
                    screen,
                    ai_game,
                    origin_x=OUTER_PAD + VIEW_W + VERSUS_GAP,
                    origin_y=HEADER_H,
                    title="RL AGENT (PPO)",
                    badge_text="AI",
                    badge_color=ACCENT_GREEN,
                    high_score=ai_best,
                    ai_value=ai_ctrl.last_value,
                    ai_speed=ai_ctrl.speed,
                    paused=paused,
                    is_ai=True,
                )
            elif is_ai_only and ai_game is not None and ai_ctrl is not None:
                view.draw(
                    screen,
                    ai_game,
                    origin_x=OUTER_PAD,
                    origin_y=HEADER_H,
                    title="RL AGENT (PPO)",
                    badge_text="AI WATCH",
                    badge_color=ACCENT_GREEN,
                    high_score=ai_best,
                    ai_value=ai_ctrl.last_value,
                    ai_speed=ai_ctrl.speed,
                    paused=paused,
                    is_ai=True,
                )
            else:
                view.draw(
                    screen,
                    human_game,
                    origin_x=OUTER_PAD,
                    origin_y=HEADER_H,
                    title="TETRIS",
                    badge_text="HUMAN",
                    badge_color=ACCENT_CYAN,
                    high_score=human_best,
                    paused=paused,
                )

            pygame.display.flip()
    finally:
        pygame.quit()
