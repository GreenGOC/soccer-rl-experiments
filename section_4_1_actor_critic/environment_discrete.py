"""Discrete single-agent soccer environment for Section 4-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class StepResult:
    next_state: Tuple[int, int, int, int, int]
    reward: float
    done: bool
    info: Dict[str, float]


class DiscreteSoccerEnv:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_KICK = 4
    ACTION_STAY = 5

    def __init__(
        self,
        grid_size: int = 12,
        max_steps: int = 250,
        seed: int = 0,
        reward_goal: float = 100.0,
        reward_concede: float = -100.0,
        reward_step: float = -0.05,
        reward_ball_progress: float = 0.2,
        reward_collision: float = -1.0,
        reward_out: float = -1.0,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_concede = reward_concede
        self.reward_step = reward_step
        self.reward_ball_progress = reward_ball_progress
        self.reward_collision = reward_collision
        self.reward_out = reward_out

        self._rng = np.random.default_rng(seed)
        self.steps = 0

        self.agent_goal_x = self.grid_size - 1
        self.opponent_goal_x = 0

        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.opponent_pos = np.zeros(2, dtype=np.int32)
        self.ball_pos = np.zeros(2, dtype=np.int32)
        self.has_ball = False

    @property
    def n_actions(self) -> int:
        return 6

    def reset(self, seed: int | None = None) -> Tuple[int, int, int, int, int]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.agent_pos = np.array([1, self._rng.integers(0, self.grid_size)], dtype=np.int32)
        self.opponent_pos = np.array([self.grid_size - 2, self._rng.integers(0, self.grid_size)], dtype=np.int32)
        self.ball_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int32)

        dist_agent = np.abs(self.agent_pos - self.ball_pos).sum()
        dist_opp = np.abs(self.opponent_pos - self.ball_pos).sum()
        if dist_agent < dist_opp:
            self.has_ball = True
        elif dist_opp < dist_agent:
            self.has_ball = False
        else:
            self.has_ball = bool(self._rng.integers(0, 2))

        if self.has_ball:
            self.ball_pos = self.agent_pos.copy()
        else:
            self.ball_pos = self.opponent_pos.copy()

        return self._state()

    def step(self, action: int) -> StepResult:
        self.steps += 1
        reward = self.reward_step
        done = False
        info: Dict[str, float] = {
            "scored": 0.0,
            "conceded": 0.0,
            "out_of_bounds": 0.0,
            "collision": 0.0,
        }

        prev_ball_goal_dist = self._ball_to_goal_dist()

        self.agent_pos, agent_blocked, agent_oob = self._apply_move(self.agent_pos, action)
        if agent_oob:
            reward += self.reward_out
            info["out_of_bounds"] = 1.0
        if agent_blocked:
            reward += self.reward_collision
            info["collision"] = 1.0

        opp_action = self._opponent_policy()
        self.opponent_pos, _, _ = self._apply_move(self.opponent_pos, opp_action)

        if np.array_equal(self.agent_pos, self.opponent_pos):
            if self._rng.random() < 0.5:
                self.agent_pos = np.clip(self.agent_pos + np.array([0, 1]), 0, self.grid_size - 1)
            else:
                self.opponent_pos = np.clip(self.opponent_pos + np.array([0, -1]), 0, self.grid_size - 1)

        if action == self.ACTION_KICK and self.has_ball:
            self.ball_pos[0] = min(self.grid_size - 1, self.ball_pos[0] + 1)
            self.ball_pos[1] = self.agent_pos[1]
            self.has_ball = False

        if opp_action == self.ACTION_KICK and np.array_equal(self.opponent_pos, self.ball_pos):
            self.ball_pos[0] = max(0, self.ball_pos[0] - 1)
            self.ball_pos[1] = self.opponent_pos[1]

        if self.has_ball:
            self.ball_pos = self.agent_pos.copy()
        elif np.array_equal(self.opponent_pos, self.ball_pos):
            self.ball_pos = self.opponent_pos.copy()

        if np.array_equal(self.agent_pos, self.ball_pos):
            self.has_ball = True
        elif np.array_equal(self.opponent_pos, self.ball_pos):
            self.has_ball = False

        if self.ball_pos[0] >= self.agent_goal_x:
            reward += self.reward_goal
            done = True
            info["scored"] = 1.0
        elif self.ball_pos[0] <= self.opponent_goal_x:
            reward += self.reward_concede
            done = True
            info["conceded"] = 1.0

        new_ball_goal_dist = self._ball_to_goal_dist()
        reward += self.reward_ball_progress * float(prev_ball_goal_dist - new_ball_goal_dist)

        if self.steps >= self.max_steps:
            done = True

        return StepResult(self._state(), float(reward), done, info)

    def _state(self) -> Tuple[int, int, int, int, int]:
        return (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(self.ball_pos[0]),
            int(self.ball_pos[1]),
            int(self.has_ball),
        )

    def _ball_to_goal_dist(self) -> float:
        return float((self.grid_size - 1) - self.ball_pos[0])

    def _opponent_policy(self) -> int:
        if np.array_equal(self.opponent_pos, self.ball_pos):
            return self.ACTION_KICK

        dx = self.ball_pos[0] - self.opponent_pos[0]
        dy = self.ball_pos[1] - self.opponent_pos[1]
        if abs(dx) > abs(dy):
            return self.ACTION_RIGHT if dx > 0 else self.ACTION_LEFT
        if dy != 0:
            return self.ACTION_DOWN if dy > 0 else self.ACTION_UP
        return self.ACTION_STAY

    def _apply_move(self, pos: np.ndarray, action: int):
        move = np.array([0, 0], dtype=np.int32)
        if action == self.ACTION_UP:
            move = np.array([0, -1], dtype=np.int32)
        elif action == self.ACTION_DOWN:
            move = np.array([0, 1], dtype=np.int32)
        elif action == self.ACTION_LEFT:
            move = np.array([-1, 0], dtype=np.int32)
        elif action == self.ACTION_RIGHT:
            move = np.array([1, 0], dtype=np.int32)

        proposed = pos + move
        oob = bool(np.any(proposed < 0) or np.any(proposed >= self.grid_size))
        clipped = np.clip(proposed, 0, self.grid_size - 1)
        blocked = bool(np.array_equal(clipped, pos) and np.any(move != 0))
        return clipped.astype(np.int32), blocked, oob
