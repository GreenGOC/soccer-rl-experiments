"""Continuous-action soccer environment for Section 6-1-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, float]


class ContinuousActionSoccerEnv:
    """One learning agent + scripted teammates/opponents with variable counts.

    Action = [vx_cmd, vy_cmd, vtheta_cmd, kx_cmd, ky_cmd] in [-1, 1].
    """

    def __init__(
        self,
        max_steps: int = 250,
        seed: int = 0,
        max_teammates: int = 2,
        max_opponents: int = 3,
        randomize_counts: bool = True,
        dt: float = 0.08,
        move_gain: float = 2.8,
        friction: float = 0.9,
        kick_gain: float = 4.0,
    ) -> None:
        self.max_steps = max_steps
        self.max_teammates = max_teammates
        self.max_opponents = max_opponents
        self.randomize_counts = randomize_counts
        self.dt = dt
        self.move_gain = move_gain
        self.friction = friction
        self.kick_gain = kick_gain

        self._rng = np.random.default_rng(seed)
        self.steps = 0

        # Scale field with team size to avoid agent overcrowding in larger modes.
        max_team_size = max(1 + self.max_teammates, self.max_opponents)
        scale = max(1.0, np.sqrt(max_team_size / 2.0))
        self.field_min = np.array([-1.0 * scale, -0.6 * scale], dtype=np.float32)
        self.field_max = np.array([1.0 * scale, 0.6 * scale], dtype=np.float32)

        self.ego_dim = 15
        self.entity_dim = 6
        self.action_dim = 5
        self.flat_dim = self.ego_dim + self.max_teammates * self.entity_dim + self.max_opponents * self.entity_dim

        self.active_teammates = self.max_teammates
        self.active_opponents = self.max_opponents

        self.team0_pos = np.zeros((1 + self.max_teammates, 2), dtype=np.float32)
        self.team0_vel = np.zeros_like(self.team0_pos)
        self.team1_pos = np.zeros((self.max_opponents, 2), dtype=np.float32)
        self.team1_vel = np.zeros_like(self.team1_pos)

        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_owner = (0, 0)  # (team, idx) or (-1, -1) free
        self.ego_theta = 0.0

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        if self.randomize_counts:
            self.active_teammates = int(self._rng.integers(1, self.max_teammates + 1))
            self.active_opponents = int(self._rng.integers(1, self.max_opponents + 1))
        else:
            self.active_teammates = self.max_teammates
            self.active_opponents = self.max_opponents

        self.team0_pos.fill(0.0)
        self.team0_vel.fill(0.0)
        self.team1_pos.fill(0.0)
        self.team1_vel.fill(0.0)

        # Learning agent at index 0; spread teammates/opponents over vertical lanes.
        x_left = 0.7 * self.field_min[0]
        x_mid_left = 0.45 * self.field_min[0]
        x_mid_right = 0.45 * self.field_max[0]
        y_min = 0.85 * self.field_min[1]
        y_max = 0.85 * self.field_max[1]

        self.team0_pos[0] = np.array([x_left, self._rng.uniform(-0.15 * abs(self.field_max[1]), 0.15 * abs(self.field_max[1]))], dtype=np.float32)
        if self.active_teammates > 0:
            y_slots_tm = np.linspace(y_min, y_max, self.active_teammates, dtype=np.float32)
            for i in range(1, self.active_teammates + 1):
                self.team0_pos[i] = np.array([x_mid_left, y_slots_tm[i - 1]], dtype=np.float32)

        if self.active_opponents > 0:
            y_slots_op = np.linspace(y_min, y_max, self.active_opponents, dtype=np.float32)
            for i in range(self.active_opponents):
                self.team1_pos[i] = np.array([x_mid_right, y_slots_op[i]], dtype=np.float32)

        self.ball_pos = np.array([0.0, self._rng.uniform(-0.1, 0.1)], dtype=np.float32)
        self.ball_vel[:] = 0.0
        self.ego_theta = float(self._rng.uniform(-0.1, 0.1))

        # nearest owner
        nearest_team = -1
        nearest_idx = -1
        best = 1e9

        for i in range(self.active_teammates + 1):
            d = float(np.linalg.norm(self.team0_pos[i] - self.ball_pos))
            if d < best:
                best, nearest_team, nearest_idx = d, 0, i
        for i in range(self.active_opponents):
            d = float(np.linalg.norm(self.team1_pos[i] - self.ball_pos))
            if d < best:
                best, nearest_team, nearest_idx = d, 1, i

        self.ball_owner = (nearest_team, nearest_idx)
        self._sync_ball_with_owner()
        return self._get_obs()

    def step(self, action: np.ndarray) -> StepResult:
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        reward = -0.01
        done = False
        info: Dict[str, float] = {
            "scored": 0.0,
            "conceded": 0.0,
            "ball_x": float(self.ball_pos[0]),
            "pass_success": 0.0,
            "collision": 0.0,
        }

        prev_ball_x = float(self.ball_pos[0])

        # learning agent control: body-frame velocity + angular rate.
        omega = float(action[2]) * np.pi
        self.ego_theta += self.dt * omega
        self.ego_theta = float((self.ego_theta + np.pi) % (2.0 * np.pi) - np.pi)

        c, s = float(np.cos(self.ego_theta)), float(np.sin(self.ego_theta))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        cmd_v_body = action[:2] * self.move_gain
        cmd_v_world = rot @ cmd_v_body
        self.team0_vel[0] += self.dt * cmd_v_world

        # scripted teammates/opponents move to ball
        for i in range(1, self.active_teammates + 1):
            self.team0_vel[i] += self.dt * self._chase_acc(self.team0_pos[i], self.ball_pos)
        for i in range(self.active_opponents):
            self.team1_vel[i] += self.dt * self._chase_acc(self.team1_pos[i], self.ball_pos)

        self.team0_vel *= self.friction
        self.team1_vel *= self.friction

        self.team0_pos[: self.active_teammates + 1] += self.dt * self.team0_vel[: self.active_teammates + 1]
        self.team1_pos[: self.active_opponents] += self.dt * self.team1_vel[: self.active_opponents]

        for i in range(self.active_teammates + 1):
            self.team0_pos[i], _ = self._clip(self.team0_pos[i])
        for i in range(self.active_opponents):
            self.team1_pos[i], _ = self._clip(self.team1_pos[i])

        # kick by ego
        if self.ball_owner == (0, 0):
            reward += 0.1
            kvec_body = action[3:5]
            kvec = rot @ kvec_body
            kn = np.linalg.norm(kvec)
            if kn > 0.3:
                self.ball_owner = (-1, -1)
                self.ball_vel = (kvec / (kn + 1e-6)) * self.kick_gain

        # scripted kicks
        if self.ball_owner[0] == 0 and self.ball_owner[1] > 0:
            self.ball_owner = (-1, -1)
            self.ball_vel = np.array([self.kick_gain, 0.0], dtype=np.float32)
            info["pass_success"] = 1.0
        elif self.ball_owner[0] == 1:
            self.ball_owner = (-1, -1)
            self.ball_vel = np.array([-self.kick_gain, 0.0], dtype=np.float32)

        if self.ball_owner == (-1, -1):
            self.ball_vel *= 0.97
            self.ball_pos += self.dt * self.ball_vel
            self.ball_pos, _ = self._clip(self.ball_pos)
            self._capture_ball_if_close()
        else:
            self._sync_ball_with_owner()

        # collisions near ego
        for i in range(1, self.active_teammates + 1):
            if np.linalg.norm(self.team0_pos[i] - self.team0_pos[0]) < 0.05:
                reward -= 0.2
                info["collision"] += 1.0
        for i in range(self.active_opponents):
            if np.linalg.norm(self.team1_pos[i] - self.team0_pos[0]) < 0.05:
                reward -= 0.25
                info["collision"] += 1.0

        # ball progress toward right goal
        reward += 0.25 * (float(self.ball_pos[0]) - prev_ball_x)

        # goals
        if self.ball_pos[0] >= self.field_max[0]:
            reward += 120.0
            done = True
            info["scored"] = 1.0
        elif self.ball_pos[0] <= self.field_min[0]:
            reward -= 120.0
            done = True
            info["conceded"] = 1.0

        if self.steps >= self.max_steps:
            done = True

        return StepResult(self._get_obs(), float(reward), done, info)

    def _chase_acc(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        d = target - pos
        n = np.linalg.norm(d)
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return self.move_gain * d / n

    def _clip(self, pos: np.ndarray) -> Tuple[np.ndarray, bool]:
        c = np.clip(pos, self.field_min, self.field_max)
        return c.astype(np.float32), bool(np.any(np.abs(c - pos) > 1e-8))

    def _sync_ball_with_owner(self) -> None:
        t, i = self.ball_owner
        if t == 0:
            self.ball_pos = self.team0_pos[i].copy()
            self.ball_vel[:] = 0.0
        elif t == 1:
            self.ball_pos = self.team1_pos[i].copy()
            self.ball_vel[:] = 0.0

    def _capture_ball_if_close(self) -> None:
        best = 1e9
        owner = (-1, -1)
        for i in range(self.active_teammates + 1):
            d = float(np.linalg.norm(self.team0_pos[i] - self.ball_pos))
            if d < best:
                best = d
                owner = (0, i)
        for i in range(self.active_opponents):
            d = float(np.linalg.norm(self.team1_pos[i] - self.ball_pos))
            if d < best:
                best = d
                owner = (1, i)
        if best < 0.08:
            self.ball_owner = owner
            self._sync_ball_with_owner()

    def _entity_feat(self, ent_pos: np.ndarray, ent_vel: np.ndarray, team_flag: float, has_ball: float) -> np.ndarray:
        rel = ent_pos - self.team0_pos[0]
        return np.array([rel[0], rel[1], ent_vel[0], ent_vel[1], team_flag, has_ball], dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ego_pos = self.team0_pos[0]
        ego_vel = self.team0_vel[0]
        ball_rel = self.ball_pos - ego_pos
        goal_rel = np.array([self.field_max[0], 0.0], dtype=np.float32) - ego_pos
        theta_sin = float(np.sin(self.ego_theta))
        theta_cos = float(np.cos(self.ego_theta))

        has_ball = 1.0 if self.ball_owner == (0, 0) else 0.0
        ego = np.array(
            [
                ego_pos[0],
                ego_pos[1],
                ego_vel[0],
                ego_vel[1],
                self.ball_pos[0],
                self.ball_pos[1],
                self.ball_vel[0],
                self.ball_vel[1],
                ball_rel[0],
                ball_rel[1],
                goal_rel[0],
                goal_rel[1],
                theta_sin,
                theta_cos,
                has_ball,
            ],
            dtype=np.float32,
        )

        teammates = []
        for i in range(1, self.active_teammates + 1):
            teammates.append(
                self._entity_feat(
                    self.team0_pos[i],
                    self.team0_vel[i],
                    team_flag=1.0,
                    has_ball=1.0 if self.ball_owner == (0, i) else 0.0,
                )
            )

        opponents = []
        for i in range(self.active_opponents):
            opponents.append(
                self._entity_feat(
                    self.team1_pos[i],
                    self.team1_vel[i],
                    team_flag=-1.0,
                    has_ball=1.0 if self.ball_owner == (1, i) else 0.0,
                )
            )

        teammates_arr = np.array(teammates, dtype=np.float32) if teammates else np.zeros((0, self.entity_dim), dtype=np.float32)
        opponents_arr = np.array(opponents, dtype=np.float32) if opponents else np.zeros((0, self.entity_dim), dtype=np.float32)

        tm_pad = np.zeros((self.max_teammates, self.entity_dim), dtype=np.float32)
        op_pad = np.zeros((self.max_opponents, self.entity_dim), dtype=np.float32)
        if teammates_arr.shape[0] > 0:
            tm_pad[: teammates_arr.shape[0]] = teammates_arr
        if opponents_arr.shape[0] > 0:
            op_pad[: opponents_arr.shape[0]] = opponents_arr

        flat = np.concatenate([ego, tm_pad.reshape(-1), op_pad.reshape(-1)]).astype(np.float32)
        return {
            "ego": ego,
            "teammates": teammates_arr,
            "opponents": opponents_arr,
            "flat": flat,
        }
