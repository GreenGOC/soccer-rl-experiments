"""Continuous-state, discrete-action soccer environment for Section 6-1-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class ContinuousStateDiscreteActionSoccerEnv:
    """Single-agent environment with scripted opponent.

    State is continuous. Actions are discrete:
    0 up, 1 down, 2 left, 3 right, 4 kick, 5 stay.
    """

    def __init__(
        self,
        team_size: int = 2,
        max_steps: int = 250,
        seed: int = 0,
        dt: float = 0.1,
        accel: float = 2.5,
        friction: float = 0.92,
        kick_speed: float = 3.5,
        reward_goal: float = 100.0,
        reward_concede: float = -100.0,
        reward_step: float = -0.02,
        reward_progress: float = 0.4,
        reward_possession: float = 0.05,
        reward_out: float = -1.0,
    ) -> None:
        self.team_size = max(2, int(team_size))
        self.max_steps = max_steps
        self.dt = dt
        self.accel = accel
        self.friction = friction
        self.kick_speed = kick_speed

        self.reward_goal = reward_goal
        self.reward_concede = reward_concede
        self.reward_step = reward_step
        self.reward_progress = reward_progress
        self.reward_possession = reward_possession
        self.reward_out = reward_out

        self._rng = np.random.default_rng(seed)
        self.steps = 0

        # Scale field with team size to support 2v2/3v3/... modes.
        scale = max(1.0, np.sqrt(self.team_size / 2.0))
        self.field_min = np.array([-1.0 * scale, -0.6 * scale], dtype=np.float32)
        self.field_max = np.array([1.0 * scale, 0.6 * scale], dtype=np.float32)

        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.opp_pos = np.zeros(2, dtype=np.float32)
        self.opp_vel = np.zeros(2, dtype=np.float32)
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_owner = 0  # 1 agent, -1 opponent, 0 free

        self.action_n = 6
        self.obs_dim = 18

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        x_spawn = 0.65 * float(self.field_max[0])
        y_agent = 0.42 * float(self.field_max[1])
        y_ball = 0.17 * float(self.field_max[1])
        self.agent_pos = np.array([-x_spawn, self._rng.uniform(-y_agent, y_agent)], dtype=np.float32)
        self.opp_pos = np.array([x_spawn, self._rng.uniform(-y_agent, y_agent)], dtype=np.float32)
        self.ball_pos = np.array([0.0, self._rng.uniform(-y_ball, y_ball)], dtype=np.float32)

        self.agent_vel[:] = 0.0
        self.opp_vel[:] = 0.0
        self.ball_vel[:] = 0.0

        # initial possession by closest
        da = np.linalg.norm(self.agent_pos - self.ball_pos)
        do = np.linalg.norm(self.opp_pos - self.ball_pos)
        if da < do:
            self.ball_owner = 1
            self.ball_pos = self.agent_pos.copy()
        elif do < da:
            self.ball_owner = -1
            self.ball_pos = self.opp_pos.copy()
        else:
            self.ball_owner = 1 if self._rng.random() < 0.5 else -1
            self.ball_pos = (self.agent_pos if self.ball_owner == 1 else self.opp_pos).copy()

        return self._obs()

    def step(self, action: int) -> StepResult:
        self.steps += 1
        reward = self.reward_step
        done = False
        info: Dict[str, float] = {"scored": 0.0, "conceded": 0.0, "out_of_bounds": 0.0}

        prev_ball_x = float(self.ball_pos[0])

        # controlled action
        a_acc = self._acc_from_action(action)
        self.agent_vel += self.dt * a_acc

        # scripted opponent
        opp_action = self._scripted_opponent_action()
        o_acc = self._acc_from_action(opp_action)
        self.opp_vel += self.dt * o_acc

        self.agent_vel *= self.friction
        self.opp_vel *= self.friction

        self.agent_pos += self.dt * self.agent_vel
        self.opp_pos += self.dt * self.opp_vel

        self.agent_pos, agent_oob = self._clip_position(self.agent_pos)
        self.opp_pos, _ = self._clip_position(self.opp_pos)
        if agent_oob:
            reward += self.reward_out
            info["out_of_bounds"] = 1.0

        # kick and possession logic
        if self.ball_owner == 1 and action == 4:
            self.ball_owner = 0
            kick_dir = np.array([1.0, 0.0], dtype=np.float32)
            self.ball_vel = self.kick_speed * kick_dir
        elif self.ball_owner == -1 and opp_action == 4:
            self.ball_owner = 0
            kick_dir = np.array([-1.0, 0.0], dtype=np.float32)
            self.ball_vel = self.kick_speed * kick_dir

        if self.ball_owner == 1:
            self.ball_pos = self.agent_pos.copy()
            self.ball_vel[:] = 0.0
        elif self.ball_owner == -1:
            self.ball_pos = self.opp_pos.copy()
            self.ball_vel[:] = 0.0
        else:
            self.ball_vel *= 0.97
            self.ball_pos += self.dt * self.ball_vel
            self.ball_pos, _ = self._clip_position(self.ball_pos)

            # capture free ball
            da = np.linalg.norm(self.agent_pos - self.ball_pos)
            do = np.linalg.norm(self.opp_pos - self.ball_pos)
            if da < 0.08 and da < do:
                self.ball_owner = 1
            elif do < 0.08 and do < da:
                self.ball_owner = -1

        if self.ball_owner == 1:
            reward += self.reward_possession

        # dense reward: ball progress to right goal (agent scores at x >= 1)
        reward += self.reward_progress * (float(self.ball_pos[0]) - prev_ball_x)

        # goal checks
        if self.ball_pos[0] >= self.field_max[0]:
            reward += self.reward_goal
            done = True
            info["scored"] = 1.0
        elif self.ball_pos[0] <= self.field_min[0]:
            reward += self.reward_concede
            done = True
            info["conceded"] = 1.0

        if self.steps >= self.max_steps:
            done = True

        return StepResult(self._obs(), float(reward), done, info)

    def _scripted_opponent_action(self) -> int:
        if self.ball_owner == -1:
            return 4
        target = self.ball_pos
        delta = target - self.opp_pos
        if abs(delta[0]) > abs(delta[1]):
            return 3 if delta[0] > 0 else 2
        return 1 if delta[1] > 0 else 0

    def _acc_from_action(self, action: int) -> np.ndarray:
        if action == 0:
            return np.array([0.0, self.accel], dtype=np.float32)
        if action == 1:
            return np.array([0.0, -self.accel], dtype=np.float32)
        if action == 2:
            return np.array([-self.accel, 0.0], dtype=np.float32)
        if action == 3:
            return np.array([self.accel, 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _clip_position(self, p: np.ndarray) -> Tuple[np.ndarray, bool]:
        clipped = np.clip(p, self.field_min, self.field_max)
        oob = bool(np.any(np.abs(clipped - p) > 1e-8))
        return clipped.astype(np.float32), oob

    def _obs(self) -> np.ndarray:
        goal = np.array([self.field_max[0], 0.0], dtype=np.float32)
        rel_ball = self.ball_pos - self.agent_pos
        rel_goal = goal - self.agent_pos
        obs = np.concatenate(
            [
                self.agent_pos,
                self.agent_vel,
                self.opp_pos,
                self.opp_vel,
                self.ball_pos,
                self.ball_vel,
                rel_ball,
                rel_goal,
                np.array([1.0 if self.ball_owner == 1 else 0.0], dtype=np.float32),
                np.array([1.0 if self.ball_owner == -1 else 0.0], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)
