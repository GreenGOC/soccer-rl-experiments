"""Entity encoders for Section 7-1 models."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Fallback encoder kept for compatibility."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEEEncoder(nn.Module):
    """Compact GEE-like encoder over local teammate/opponent slots.

    Local observation layout (dim=14):
    - own: [x, y, vx, vy]
    - ball_rel: [x, y]
    - goal_rel: [x, y]
    - teammate_rel: [x, y]
    - opponent_rel: [x, y]
    - flags: [has_ball, team_has_ball]
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        if obs_dim < 14:
            raise ValueError(f"GEEEncoder expects obs_dim >= 14, got {obs_dim}.")

        self.ego_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.entity_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        ego = torch.cat([obs[:, 0:8], obs[:, 12:14]], dim=-1)  # [B, 10]
        teammate = obs[:, 8:10].unsqueeze(1)  # [B, 1, 2]
        opponent = obs[:, 10:12].unsqueeze(1)  # [B, 1, 2]

        ego_h = self.ego_mlp(ego)
        tm_h = self.entity_mlp(teammate).max(dim=1).values
        opp_h = self.entity_mlp(opponent).max(dim=1).values
        return self.fuse(torch.cat([ego_h, tm_h, opp_h], dim=-1))

