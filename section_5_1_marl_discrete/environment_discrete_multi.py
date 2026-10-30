"""Discrete multi-agent soccer environment for Section 5-1.

Actions:
0 up, 1 down, 2 left, 3 right, 4 kick, 5 stay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class StepOutput:
    observations: Dict[str, Tuple[int, int, int, int, int, int]]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, float]


class MultiAgentDiscreteSoccerEnv:
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
        agents_per_team: int = 2,
        seed: int = 0,
        reward_goal: float = 100.0,
        reward_concede: float = -100.0,
        reward_step: float = -0.03,
        reward_ball_progress: float = 0.15,
        reward_collision: float = -0.5,
        reward_out: float = -0.5,
        cooperative: bool = True,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agents_per_team = agents_per_team
        self.reward_goal = reward_goal
        self.reward_concede = reward_concede
        self.reward_step = reward_step
        self.reward_ball_progress = reward_ball_progress
        self.reward_collision = reward_collision
        self.reward_out = reward_out
        self.cooperative = cooperative

        self._rng = np.random.default_rng(seed)
        self.steps = 0

        self.team_ids = (0, 1)
        self.agent_ids = [f"t{t}_a{i}" for t in self.team_ids for i in range(self.agents_per_team)]
        self.action_space_n = 6

        self.positions: Dict[str, np.ndarray] = {}
        self.ball_pos = np.zeros(2, dtype=np.int32)
        self.ball_owner: str | None = None

    def reset(self, seed: int | None = None) -> Dict[str, Tuple[int, int, int, int, int, int]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.positions = {}

        for i in range(self.agents_per_team):
            self.positions[f"t0_a{i}"] = np.array([1, self._rng.integers(0, self.grid_size)], dtype=np.int32)
            self.positions[f"t1_a{i}"] = np.array([self.grid_size - 2, self._rng.integers(0, self.grid_size)], dtype=np.int32)

        self.ball_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int32)

        # Assign owner to closest agent.
        dists = {aid: int(np.abs(pos - self.ball_pos).sum()) for aid, pos in self.positions.items()}
        min_dist = min(dists.values())
        candidates = [aid for aid, d in dists.items() if d == min_dist]
        self.ball_owner = candidates[int(self._rng.integers(0, len(candidates)))]
        self.ball_pos = self.positions[self.ball_owner].copy()

        return self._observations()

    def step(self, actions: Dict[str, int]) -> StepOutput:
        self.steps += 1

        rewards = {aid: self.reward_step for aid in self.agent_ids}
        info = {
            "team0_scored": 0.0,
            "team1_scored": 0.0,
            "ball_x": float(self.ball_pos[0]),
            "collisions": 0.0,
            "out_of_bounds": 0.0,
        }

        prev_ball_x = float(self.ball_pos[0])

        order = self.agent_ids.copy()
        self._rng.shuffle(order)
        for aid in order:
            action = int(actions.get(aid, self.ACTION_STAY))
            new_pos, blocked, oob = self._apply_move(self.positions[aid], action)
            self.positions[aid] = new_pos
            if blocked:
                rewards[aid] += self.reward_collision
                info["collisions"] += 1.0
            if oob:
                rewards[aid] += self.reward_out
                info["out_of_bounds"] += 1.0

        # Handle collisions between agents at same tile.
        coord_map: Dict[Tuple[int, int], List[str]] = {}
        for aid, pos in self.positions.items():
            coord_map.setdefault((int(pos[0]), int(pos[1])), []).append(aid)
        for _, agents in coord_map.items():
            if len(agents) > 1:
                info["collisions"] += 1.0
                for aid in agents:
                    rewards[aid] += self.reward_collision

        # Kick logic.
        for aid, action in actions.items():
            if action != self.ACTION_KICK:
                continue
            if self.ball_owner != aid:
                continue
            team = self.team_of(aid)
            direction = 1 if team == 0 else -1
            self.ball_pos[0] = np.clip(self.ball_pos[0] + direction, 0, self.grid_size - 1)
            self.ball_pos[1] = self.positions[aid][1]
            self.ball_owner = None

        # If no owner, nearest agent to ball captures it.
        if self.ball_owner is None:
            dists = {aid: int(np.abs(pos - self.ball_pos).sum()) for aid, pos in self.positions.items()}
            min_dist = min(dists.values())
            candidates = [aid for aid, d in dists.items() if d == min_dist]
            self.ball_owner = candidates[int(self._rng.integers(0, len(candidates)))]

        # Owner carries ball.
        if self.ball_owner is not None:
            self.ball_pos = self.positions[self.ball_owner].copy()

        # Dense ball-progress reward.
        delta_x = float(self.ball_pos[0] - prev_ball_x)
        for aid in self.agent_ids:
            team = self.team_of(aid)
            progress = delta_x if team == 0 else -delta_x
            rewards[aid] += self.reward_ball_progress * progress

        # Goal checks: team0 scores on right, team1 scores on left.
        done = False
        if self.ball_pos[0] >= self.grid_size - 1:
            done = True
            info["team0_scored"] = 1.0
        elif self.ball_pos[0] <= 0:
            done = True
            info["team1_scored"] = 1.0

        if done:
            team0_bonus = self.reward_goal if info["team0_scored"] > 0.5 else self.reward_concede
            team1_bonus = self.reward_goal if info["team1_scored"] > 0.5 else self.reward_concede
            for aid in self.agent_ids:
                if self.team_of(aid) == 0:
                    rewards[aid] += team0_bonus
                else:
                    rewards[aid] += team1_bonus

        if self.steps >= self.max_steps:
            done = True

        if self.cooperative:
            team_rewards = {
                0: float(np.mean([rewards[aid] for aid in self.team_agents(0)])),
                1: float(np.mean([rewards[aid] for aid in self.team_agents(1)])),
            }
            for aid in self.team_agents(0):
                rewards[aid] = team_rewards[0]
            for aid in self.team_agents(1):
                rewards[aid] = team_rewards[1]
        else:
            # Zero-sum shaping.
            team0_sum = float(np.mean([rewards[aid] for aid in self.team_agents(0)]))
            team1_sum = -team0_sum
            for aid in self.team_agents(0):
                rewards[aid] = team0_sum
            for aid in self.team_agents(1):
                rewards[aid] = team1_sum

        return StepOutput(self._observations(), rewards, done, info)

    def _observations(self) -> Dict[str, Tuple[int, int, int, int, int, int]]:
        obs = {}
        for aid in self.agent_ids:
            team = self.team_of(aid)
            own = self.positions[aid]
            mate = self._nearest_teammate(aid)
            obs[aid] = (
                int(own[0]),
                int(own[1]),
                int(self.ball_pos[0]),
                int(self.ball_pos[1]),
                int(self.ball_owner == aid),
                int(mate[0] - own[0] + self.grid_size),  # shifted for non-negative bins
            )
            # keep tuple small and hashable for tabular methods.
            _ = team
        return obs

    def _nearest_teammate(self, aid: str) -> np.ndarray:
        team = self.team_of(aid)
        mates = [x for x in self.team_agents(team) if x != aid]
        if not mates:
            return self.positions[aid]
        own = self.positions[aid]
        best = min(mates, key=lambda m: int(np.abs(self.positions[m] - own).sum()))
        return self.positions[best]

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

    def team_of(self, aid: str) -> int:
        return int(aid.split("_")[0][1:])

    def team_agents(self, team_id: int) -> List[str]:
        return [aid for aid in self.agent_ids if self.team_of(aid) == team_id]
