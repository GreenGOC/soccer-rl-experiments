"""Global Entity Encoder (GEE) for variable-size teammate/opponent sets."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


class GEEEncoder(nn.Module):
    def __init__(self, ego_dim: int, entity_dim: int, hidden_dim: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_dim),
            nn.ReLU(),
        )

    def masked_max_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, H], mask: [B, N]
        minus_inf = torch.full_like(x, -1e9)
        x_masked = torch.where(mask.unsqueeze(-1) > 0.5, x, minus_inf)
        pooled = x_masked.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled

    def forward(
        self,
        ego: torch.Tensor,
        teammates: torch.Tensor,
        tm_mask: torch.Tensor,
        opponents: torch.Tensor,
        opp_mask: torch.Tensor,
    ) -> torch.Tensor:
        ego_h = self.ego_mlp(ego)
        tm_h = self.entity_mlp(teammates)
        opp_h = self.entity_mlp(opponents)

        tm_pool = self.masked_max_pool(tm_h, tm_mask)
        opp_pool = self.masked_max_pool(opp_h, opp_mask)
        return self.fuse(torch.cat([ego_h, tm_pool, opp_pool], dim=-1))


def collate_obs_batch(
    obs_list: List[Dict[str, np.ndarray]],
    max_teammates: int,
    max_opponents: int,
    entity_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ego_batch = []
    tm_batch = []
    tm_mask = []
    opp_batch = []
    opp_mask = []

    for obs in obs_list:
        ego = obs["ego"].astype(np.float32)
        tm = obs["teammates"].astype(np.float32)
        opp = obs["opponents"].astype(np.float32)

        tm_pad = np.zeros((max_teammates, entity_dim), dtype=np.float32)
        op_pad = np.zeros((max_opponents, entity_dim), dtype=np.float32)
        tm_m = np.zeros((max_teammates,), dtype=np.float32)
        op_m = np.zeros((max_opponents,), dtype=np.float32)

        if tm.shape[0] > 0:
            n = min(max_teammates, tm.shape[0])
            tm_pad[:n] = tm[:n]
            tm_m[:n] = 1.0
        if opp.shape[0] > 0:
            n = min(max_opponents, opp.shape[0])
            op_pad[:n] = opp[:n]
            op_m[:n] = 1.0

        ego_batch.append(ego)
        tm_batch.append(tm_pad)
        opp_batch.append(op_pad)
        tm_mask.append(tm_m)
        opp_mask.append(op_m)

    ego_t = torch.as_tensor(np.stack(ego_batch), dtype=torch.float32, device=device)
    tm_t = torch.as_tensor(np.stack(tm_batch), dtype=torch.float32, device=device)
    opp_t = torch.as_tensor(np.stack(opp_batch), dtype=torch.float32, device=device)
    tm_m_t = torch.as_tensor(np.stack(tm_mask), dtype=torch.float32, device=device)
    opp_m_t = torch.as_tensor(np.stack(opp_mask), dtype=torch.float32, device=device)
    return ego_t, tm_t, tm_m_t, opp_t, opp_m_t
