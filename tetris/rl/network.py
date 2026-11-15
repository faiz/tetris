"""Dual-Stream Spatial 2D CNN + Structural MLP Actor-Critic for Tetris."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tetris.rl.features import NUM_FEATURES


def _init_linear(layer: nn.Module, std: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class TetrisActorCritic(nn.Module):
    """Evaluates all candidate placements jointly via CNN + MLP streams."""

    def __init__(self, num_features: int = NUM_FEATURES) -> None:
        super().__init__()
        self.num_features = num_features

        # Spatial 2D CNN stream over (N, 1, 20, 10) candidate occupancy grids
        self.spatial_conv = nn.Sequential(
            _init_linear(nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)),
            nn.SiLU(),
            _init_linear(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            nn.SiLU(),
            nn.Flatten(),
            _init_linear(nn.Linear(32 * 5 * 3, 64)),
            nn.LayerNorm(64),
            nn.SiLU(),
        )

        # Structural MLP stream over (N, 12) board heuristic features
        self.struct_mlp = nn.Sequential(
            _init_linear(nn.Linear(num_features, 64)),
            nn.LayerNorm(64),
            nn.SiLU(),
        )

        # Candidate fusion -> per-candidate embedding of size 128
        self.fusion = nn.Sequential(
            _init_linear(nn.Linear(128, 128)),
            nn.SiLU(),
        )

        # Actor head: scores each candidate independently
        self.actor_head = nn.Sequential(
            _init_linear(nn.Linear(128, 64)),
            nn.SiLU(),
            _init_linear(nn.Linear(64, 1), std=0.01),
        )

        # Critic head: permutation-invariant max + mean pooling over valid candidates
        self.critic_head = nn.Sequential(
            _init_linear(nn.Linear(256, 128)),
            nn.SiLU(),
            _init_linear(nn.Linear(128, 1), std=1.0),
        )

    def forward(
        self,
        cand_grids: torch.Tensor,
        cand_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_cands = cand_grids.shape[:2]

        flat_grids = cand_grids.reshape(batch_size * max_cands, 1, 20, 10).float()
        flat_feats = cand_feats.reshape(batch_size * max_cands, self.num_features)

        spatial_emb = self.spatial_conv(flat_grids)
        struct_emb = self.struct_mlp(flat_feats)
        fused = self.fusion(torch.cat((spatial_emb, struct_emb), dim=-1))

        # Per-candidate actor logits masked over invalid candidate slots
        logits = self.actor_head(fused).reshape(batch_size, max_cands)
        logits = logits.masked_fill(~action_mask, -1e9)

        # Permutation-invariant pooled representation for V(s)
        emb = fused.reshape(batch_size, max_cands, 128)
        mask_3d = action_mask.unsqueeze(-1)
        max_pool = emb.masked_fill(~mask_3d, -1e9).max(dim=1).values
        mask_f = mask_3d.float()
        mean_pool = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        state_repr = torch.cat((max_pool, mean_pool), dim=-1)
        value = self.critic_head(state_repr).squeeze(-1)
        return logits, value

    def get_value(
        self,
        cand_grids: torch.Tensor,
        cand_feats: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, value = self(cand_grids, cand_feats, action_mask)
        return value

    def get_action_and_value(
        self,
        cand_grids: torch.Tensor,
        cand_feats: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(cand_grids, cand_feats, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def select_action(
        self,
        cand_grids: torch.Tensor,
        cand_feats: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self(cand_grids, cand_feats, action_mask)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        return action, value


def save_checkpoint(
    path: str | Path,
    model: TetrisActorCritic,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Save CPU state_dict and JSON-safe metadata to path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "model_state_dict": cpu_state,
        "config": {"num_features": int(model.num_features)},
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TetrisActorCritic, dict[str, Any]]:
    """Load TetrisActorCritic weights and metadata onto device."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    cfg = payload.get("config", {})
    model = TetrisActorCritic(num_features=int(cfg.get("num_features", NUM_FEATURES)))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, dict(payload.get("metadata", {}))
