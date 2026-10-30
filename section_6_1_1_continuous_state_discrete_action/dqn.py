"""DQN agent for Section 6-1-1."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(obs, action, reward, next_obs, float(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([b.obs for b in batch], axis=0)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_obs = np.stack([b.next_obs for b in batch], axis=0)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(x)
            return int(torch.argmax(q, dim=1).item())

    def update(self, replay: ReplayBuffer, batch_size: int = 128) -> float:
        if len(replay) < batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = replay.sample(batch_size)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(obs_t).gather(1, act_t)
        with torch.no_grad():
            next_max_q = self.target_q_net(next_obs_t).max(dim=1, keepdim=True).values
            target = rew_t + self.gamma * (1.0 - done_t) * next_max_q

        loss = F.mse_loss(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.opt.step()

        self._soft_update()
        return float(loss.item())

    def _soft_update(self) -> None:
        for p, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
