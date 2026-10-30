"""DDPG with GEE encoder for Section 6-1-2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gee_encoder import GEEEncoder, collate_obs_batch


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_obs: Dict[str, np.ndarray]
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int = 150_000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append(Transition(obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, ego_dim: int, entity_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = GEEEncoder(ego_dim, entity_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, ego, tm, tm_mask, opp, opp_mask):
        z = self.encoder(ego, tm, tm_mask, opp, opp_mask)
        return self.head(z)


class Critic(nn.Module):
    def __init__(self, ego_dim: int, entity_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = GEEEncoder(ego_dim, entity_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ego, tm, tm_mask, opp, opp_mask, action):
        z = self.encoder(ego, tm, tm_mask, opp, opp_mask)
        return self.head(torch.cat([z, action], dim=-1))


class DDPGAgent:
    def __init__(
        self,
        ego_dim: int,
        entity_dim: int,
        max_teammates: int,
        max_opponents: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents
        self.entity_dim = entity_dim

        self.actor = Actor(ego_dim, entity_dim, action_dim).to(self.device)
        self.actor_t = Actor(ego_dim, entity_dim, action_dim).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = Critic(ego_dim, entity_dim, action_dim).to(self.device)
        self.critic_t = Critic(ego_dim, entity_dim, action_dim).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs: Dict[str, np.ndarray], noise_std: float = 0.2, evaluate: bool = False) -> np.ndarray:
        ego, tm, tm_m, opp, opp_m = collate_obs_batch(
            [obs],
            max_teammates=self.max_teammates,
            max_opponents=self.max_opponents,
            entity_dim=self.entity_dim,
            device=self.device,
        )
        with torch.no_grad():
            a = self.actor(ego, tm, tm_m, opp, opp_m).cpu().numpy()[0]
        if not evaluate:
            a = a + np.random.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def update(self, replay: ReplayBuffer, batch_size: int = 128) -> Dict[str, float]:
        if len(replay) < batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        batch = replay.sample(batch_size)
        obs = [b.obs for b in batch]
        next_obs = [b.next_obs for b in batch]
        actions = torch.as_tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.array([b.reward for b in batch], dtype=np.float32), device=self.device).unsqueeze(1)
        dones = torch.as_tensor(np.array([b.done for b in batch], dtype=np.float32), device=self.device).unsqueeze(1)

        e, t, tm, o, om = collate_obs_batch(obs, self.max_teammates, self.max_opponents, self.entity_dim, self.device)
        ne, nt, ntm, no, nom = collate_obs_batch(next_obs, self.max_teammates, self.max_opponents, self.entity_dim, self.device)

        with torch.no_grad():
            next_a = self.actor_t(ne, nt, ntm, no, nom)
            target_q = self.critic_t(ne, nt, ntm, no, nom, next_a)
            y = rewards + self.gamma * (1.0 - dones) * target_q

        q = self.critic(e, t, tm, o, om, actions)
        critic_loss = F.mse_loss(q, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_critic.step()

        pred_a = self.actor(e, t, tm, o, om)
        actor_loss = -self.critic(e, t, tm, o, om, pred_a).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.opt_actor.step()

        self._soft_update(self.actor, self.actor_t)
        self._soft_update(self.critic, self.critic_t)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
