"""Independent PPO (IPPO) for Section 7-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gee_encoder import GEEEncoder


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
LOG2PI = float(np.log(2.0 * np.pi))


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * torch.log((1 + x) / (1 - x))


class LocalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.enc = GEEEncoder(obs_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.enc(obs)
        mean = self.mean(h)
        value = self.value(h)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, value


@dataclass
class IPPORollout:
    obs_steps: List[np.ndarray]  # [T, A, obs]
    act_steps: List[np.ndarray]  # [T, A, act]
    logp_steps: List[np.ndarray]  # [T, A]
    rew_steps: List[np.ndarray]  # [T, A]
    done_steps: List[float]  # [T]
    val_steps: List[np.ndarray]  # [T, A]


class IPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        team_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 5,
        minibatch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.team_size = team_size

        self.model = LocalActorCritic(obs_dim, action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _sample(self, obs_t: torch.Tensor):
        mean, log_std, value = self.model(obs_t)
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        u = mean + std * eps
        a = torch.tanh(u)

        logp = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + LOG2PI)
        logp = logp.sum(dim=-1)
        logp = logp - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)

        entropy = (0.5 + 0.5 * LOG2PI + log_std).sum()
        return a, logp, value.squeeze(-1), entropy

    def act(self, obs_dict: Dict[str, np.ndarray], agent_ids: List[str]):
        obs_arr = np.stack([obs_dict[aid] for aid in agent_ids], axis=0)
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a, logp, value, _ = self._sample(obs_t)
        actions = {aid: a[i].cpu().numpy().astype(np.float32) for i, aid in enumerate(agent_ids)}
        logps = {aid: float(logp[i].item()) for i, aid in enumerate(agent_ids)}
        values = {aid: float(value[i].item()) for i, aid in enumerate(agent_ids)}
        return actions, logps, values

    def update(self, rollout: IPPORollout, last_values: np.ndarray) -> float:
        T = len(rollout.done_steps)
        if T == 0:
            return 0.0

        obs = np.asarray(rollout.obs_steps, dtype=np.float32)  # [T,A,O]
        act = np.asarray(rollout.act_steps, dtype=np.float32)  # [T,A,U]
        logp = np.asarray(rollout.logp_steps, dtype=np.float32)  # [T,A]
        rew = np.asarray(rollout.rew_steps, dtype=np.float32)  # [T,A]
        done = np.asarray(rollout.done_steps, dtype=np.float32)  # [T]
        val = np.asarray(rollout.val_steps, dtype=np.float32)  # [T,A]

        adv = np.zeros_like(rew, dtype=np.float32)
        ret = np.zeros_like(rew, dtype=np.float32)
        for a in range(self.team_size):
            values = np.concatenate([val[:, a], np.array([last_values[a]], dtype=np.float32)], axis=0)
            gae = 0.0
            for t in reversed(range(T)):
                mask = 1.0 - done[t]
                delta = rew[t, a] + self.gamma * values[t + 1] * mask - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                adv[t, a] = gae
            ret[:, a] = adv[:, a] + val[:, a]

        adv_flat = adv.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t = torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act.reshape(-1, act.shape[-1]), dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(logp.reshape(-1), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_flat, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=self.device)

        idx = np.arange(obs_t.shape[0])
        total_loss = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]

                mean, log_std, value = self.model(obs_t[mb])
                std = torch.exp(log_std)
                u = atanh(act_t[mb])
                new_logp = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + LOG2PI)
                new_logp = new_logp.sum(dim=-1)
                new_logp = new_logp - torch.log(1.0 - act_t[mb].pow(2) + 1e-6).sum(dim=-1)

                ratio = torch.exp(new_logp - old_logp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()

                value_loss = self.value_coef * F.mse_loss(value.squeeze(-1), ret_t[mb])
                entropy = (0.5 + 0.5 * LOG2PI + log_std).sum()

                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()

                total_loss += float(loss.item())

        return total_loss / max(1, self.update_epochs)

