"""Multi-agent continuous soccer environment for Section 7-1.

Controlled side: team 0 (N agents).
Opponent side: team 1 (N agents, scripted or policy-driven).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class StepResult:
    local_obs: Dict[str, np.ndarray]
    global_state: np.ndarray
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, float]


class MultiAgentContinuousSoccerEnv:
    def __init__(
        self,
        team_size: int = 2,
        max_steps: int = 250,
        seed: int = 0,
        dt: float = 0.08,
        move_gain: float = 2.8,
        friction: float = 0.9,
        kick_gain: float = 4.0,
    ) -> None:
        self.team_size = team_size
        self.max_steps = max_steps
        self.dt = dt
        self.move_gain = move_gain
        self.friction = friction
        self.kick_gain = kick_gain

        self._rng = np.random.default_rng(seed)
        self.steps = 0

        scale = max(1.0, np.sqrt(self.team_size / 2.0))
        self.field_min = np.array([-1.0 * scale, -0.6 * scale], dtype=np.float32)
        self.field_max = np.array([1.0 * scale, 0.6 * scale], dtype=np.float32)

        self.controlled_ids = [f"t0_{i}" for i in range(self.team_size)]
        self.team1_ids = [f"t1_{i}" for i in range(self.team_size)]

        self.action_dim = 3  # vx_cmd, vy_cmd, kick_cmd
        self.local_obs_dim = 14
        self.global_state_dim = self.team_size * 8 + 4

        self.team0_pos = np.zeros((self.team_size, 2), dtype=np.float32)
        self.team0_vel = np.zeros((self.team_size, 2), dtype=np.float32)
        self.team1_pos = np.zeros((self.team_size, 2), dtype=np.float32)
        self.team1_vel = np.zeros((self.team_size, 2), dtype=np.float32)

        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_owner: Tuple[int, int] | None = None  # (team, idx)
        self.prev_owner: Tuple[int, int] | None = None

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.team0_vel.fill(0.0)
        self.team1_vel.fill(0.0)

        y_min = 0.85 * self.field_min[1]
        y_max = 0.85 * self.field_max[1]
        y_slots = np.linspace(y_min, y_max, self.team_size, dtype=np.float32)
        for i in range(self.team_size):
            self.team0_pos[i] = np.array([0.65 * self.field_min[0], y_slots[i]], dtype=np.float32)
            self.team1_pos[i] = np.array([0.65 * self.field_max[0], y_slots[i]], dtype=np.float32)

        self.ball_pos = np.array([0.0, self._rng.uniform(-0.1, 0.1)], dtype=np.float32)
        self.ball_vel[:] = 0.0

        self.ball_owner = self._nearest_owner_unbiased()
        self.prev_owner = self.ball_owner
        self._sync_ball()
        return self._local_obs(), self._global_state()

    def get_team_obs(self, team_id: int) -> Dict[str, np.ndarray]:
        if team_id not in (0, 1):
            raise ValueError(f"team_id must be 0 or 1, got {team_id}")
        return self._local_obs_team(team_id)

    def step(
        self,
        team0_actions: Dict[str, np.ndarray],
        team1_actions: Dict[str, np.ndarray] | None = None,
    ) -> StepResult:
        self.steps += 1
        reward_team0 = -0.01
        done = False

        prev_ball_x = float(self.ball_pos[0])
        pass_count = 0.0
        collision = 0.0

        for i, aid in enumerate(self.controlled_ids):
            action = np.asarray(team0_actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            self.team0_vel[i] += self.dt * action[:2] * self.move_gain

        if team1_actions is None:
            for i in range(self.team_size):
                if self.ball_owner == (1, i):
                    acc = np.zeros(2, dtype=np.float32)
                else:
                    acc = self._chase_acc(self.team1_pos[i], self.ball_pos)
                self.team1_vel[i] += self.dt * acc
        else:
            for i, aid in enumerate(self.team1_ids):
                action = np.asarray(team1_actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
                self.team1_vel[i] += self.dt * action[:2] * self.move_gain

        self.team0_vel *= self.friction
        self.team1_vel *= self.friction

        self.team0_pos += self.dt * self.team0_vel
        self.team1_pos += self.dt * self.team1_vel

        for i in range(self.team_size):
            self.team0_pos[i], _ = self._clip(self.team0_pos[i])
            self.team1_pos[i], _ = self._clip(self.team1_pos[i])

        if self.ball_owner is not None:
            t, idx = self.ball_owner
            if t == 0:
                aid = self.controlled_ids[idx]
                a = np.asarray(team0_actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32)
                if a[2] > 0.3:
                    direction = np.array([a[0] + 1.2, a[1]], dtype=np.float32)
                    n = float(np.linalg.norm(direction))
                    self.ball_owner = None
                    self.ball_vel = direction / (n + 1e-6) * self.kick_gain
            else:
                if team1_actions is None:
                    direction = np.array([-1.0, 0.0], dtype=np.float32)
                    self.ball_owner = None
                    self.ball_vel = direction * self.kick_gain
                else:
                    aid = self.team1_ids[idx]
                    a = np.asarray(team1_actions.get(aid, np.zeros(self.action_dim, dtype=np.float32)), dtype=np.float32)
                    if a[2] > 0.3:
                        direction = np.array([a[0] - 1.2, a[1]], dtype=np.float32)
                        n = float(np.linalg.norm(direction))
                        self.ball_owner = None
                        self.ball_vel = direction / (n + 1e-6) * self.kick_gain

        if self.ball_owner is None:
            self.ball_vel *= 0.97
            self.ball_pos += self.dt * self.ball_vel
            self.ball_pos, _ = self._clip(self.ball_pos)
            self._capture_ball()
        else:
            self._sync_ball()

        if self.prev_owner is not None and self.ball_owner is not None:
            if self.prev_owner[0] == 0 and self.ball_owner[0] == 0 and self.prev_owner[1] != self.ball_owner[1]:
                pass_count = 1.0
                reward_team0 += 0.3

        if self.ball_owner is not None and self.ball_owner[0] == 0:
            reward_team0 += 0.05

        reward_team0 += 0.35 * (float(self.ball_pos[0]) - prev_ball_x)

        for i in range(self.team_size):
            for j in range(self.team_size):
                if np.linalg.norm(self.team0_pos[i] - self.team1_pos[j]) < 0.05:
                    collision += 1.0
        reward_team0 -= 0.02 * collision

        success = 0.0
        conceded = 0.0
        if self.ball_pos[0] >= self.field_max[0]:
            reward_team0 += 100.0
            success = 1.0
            done = True
        elif self.ball_pos[0] <= self.field_min[0]:
            reward_team0 -= 100.0
            conceded = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        reward_team1 = -reward_team0
        rewards = {aid: float(reward_team0) for aid in self.controlled_ids}

        info = {
            "success": success,
            "conceded": conceded,
            "pass_count": pass_count,
            "collision": collision,
            "team0_reward": float(reward_team0),
            "team1_reward": float(reward_team1),
            "ball_x": float(self.ball_pos[0]),
        }

        self.prev_owner = self.ball_owner
        return StepResult(self._local_obs(), self._global_state(), rewards, done, info)

    def _capture_ball(self) -> None:
        owner, best = self._nearest_owner_unbiased(return_distance=True)
        if best < 0.08 and owner is not None:
            self.ball_owner = owner
            self._sync_ball()

    def _nearest_owner_unbiased(self, return_distance: bool = False):
        # Avoid deterministic team-0 bias when distances are tied by symmetry.
        candidates = []
        for i in range(self.team_size):
            d = float(np.linalg.norm(self.team0_pos[i] - self.ball_pos))
            candidates.append((d, (0, i)))
        for i in range(self.team_size):
            d = float(np.linalg.norm(self.team1_pos[i] - self.ball_pos))
            candidates.append((d, (1, i)))

        best = min(d for d, _ in candidates)
        tol = 1e-6
        nearest = [owner for d, owner in candidates if abs(d - best) <= tol]
        chosen = nearest[int(self._rng.integers(0, len(nearest)))]
        if return_distance:
            return chosen, best
        return chosen

    def _sync_ball(self) -> None:
        if self.ball_owner is None:
            return
        t, i = self.ball_owner
        if t == 0:
            self.ball_pos = self.team0_pos[i].copy()
        else:
            self.ball_pos = self.team1_pos[i].copy()
        self.ball_vel[:] = 0.0

    def _chase_acc(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        d = target - pos
        n = np.linalg.norm(d)
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return self.move_gain * d / n

    def _clip(self, p: np.ndarray):
        c = np.clip(p, self.field_min, self.field_max)
        return c.astype(np.float32), bool(np.any(np.abs(c - p) > 1e-8))

    def _local_obs(self) -> Dict[str, np.ndarray]:
        return self._local_obs_team(0)

    def _local_obs_team(self, team_id: int) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        if team_id == 0:
            ids = self.controlled_ids
            own_pos = self.team0_pos
            own_vel = self.team0_vel
            opp_pos = self.team1_pos
            goal = np.array([self.field_max[0], 0.0], dtype=np.float32)
            team_tag = 0
        else:
            ids = self.team1_ids
            own_pos = self.team1_pos
            own_vel = self.team1_vel
            opp_pos = self.team0_pos
            goal = np.array([self.field_min[0], 0.0], dtype=np.float32)
            team_tag = 1

        for i, aid in enumerate(ids):
            own = own_pos[i]
            own_v = own_vel[i]

            mate_idx = self._nearest_same_team_idx(own_pos, i)
            mate_rel = own_pos[mate_idx] - own

            nearest_opp_idx = int(np.argmin(np.linalg.norm(opp_pos - own, axis=1)))
            opp_rel = opp_pos[nearest_opp_idx] - own

            ball_rel = self.ball_pos - own
            goal_rel = goal - own

            has_ball = 1.0 if self.ball_owner == (team_tag, i) else 0.0
            team_has_ball = 1.0 if self.ball_owner is not None and self.ball_owner[0] == team_tag else 0.0

            obs[aid] = np.array(
                [
                    own[0],
                    own[1],
                    own_v[0],
                    own_v[1],
                    ball_rel[0],
                    ball_rel[1],
                    goal_rel[0],
                    goal_rel[1],
                    mate_rel[0],
                    mate_rel[1],
                    opp_rel[0],
                    opp_rel[1],
                    has_ball,
                    team_has_ball,
                ],
                dtype=np.float32,
            )
        return obs

    def _global_state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.team0_pos.reshape(-1),
                self.team0_vel.reshape(-1),
                self.team1_pos.reshape(-1),
                self.team1_vel.reshape(-1),
                self.ball_pos,
                self.ball_vel,
            ]
        ).astype(np.float32)

    def _nearest_same_team_idx(self, team_pos: np.ndarray, idx: int) -> int:
        if self.team_size <= 1:
            return idx
        own = team_pos[idx]
        best_idx = idx
        best_dist = 1e9
        for j in range(self.team_size):
            if j == idx:
                continue
            d = float(np.linalg.norm(team_pos[j] - own))
            if d < best_dist:
                best_dist = d
                best_idx = j
        return best_idx
