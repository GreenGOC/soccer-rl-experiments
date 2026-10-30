"""Joint Action Learner for Section 5-1.

For small teams (<=3), all team-0 agents are controlled jointly.
For larger teams, it controls the first 3 agents jointly and uses heuristics for the rest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import DefaultDict, Dict, List, Tuple
import numpy as np

from environment_discrete_multi import MultiAgentDiscreteSoccerEnv
from independent_q_learning import EpisodeLog, linear_epsilon


TeamState = Tuple[int, ...]


class TeamHeuristic:
    def action(self, env: MultiAgentDiscreteSoccerEnv, aid: str) -> int:
        pos = env.positions[aid]
        ball = env.ball_pos
        if env.ball_owner == aid:
            return env.ACTION_KICK

        dx = int(ball[0] - pos[0])
        dy = int(ball[1] - pos[1])
        if abs(dx) > abs(dy):
            return env.ACTION_RIGHT if dx > 0 else env.ACTION_LEFT
        if dy != 0:
            return env.ACTION_DOWN if dy > 0 else env.ACTION_UP
        return env.ACTION_STAY


class JointActionLearner:
    def __init__(self, env: MultiAgentDiscreteSoccerEnv, alpha: float, gamma: float) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.team0_agents = env.team_agents(0)
        self.team1_agents = env.team_agents(1)

        if len(self.team0_agents) <= 3:
            self.controlled_team0_agents = self.team0_agents
        else:
            # Keep joint action space tractable for larger modes.
            self.controlled_team0_agents = self.team0_agents[:3]

        self.joint_actions = list(product(range(env.action_space_n), repeat=len(self.controlled_team0_agents)))
        self.q_table: DefaultDict[TeamState, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.joint_actions), dtype=np.float64)
        )
        self.heur = TeamHeuristic()

    def team_state(self, obs: Dict[str, Tuple[int, ...]]) -> TeamState:
        pieces = []
        for aid in self.controlled_team0_agents:
            pieces.extend(obs[aid])
        return tuple(pieces)

    def choose_joint_action(self, state: TeamState, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, len(self.joint_actions)))
        return int(np.argmax(self.q_table[state]))

    def build_actions(self, joint_idx: int) -> Dict[str, int]:
        acts = {}
        joint = self.joint_actions[joint_idx]
        for i, aid in enumerate(self.controlled_team0_agents):
            acts[aid] = int(joint[i])
        for aid in self.team0_agents[len(self.controlled_team0_agents) :]:
            acts[aid] = self.heur.action(self.env, aid)
        for aid in self.team1_agents:
            acts[aid] = self.heur.action(self.env, aid)
        return acts


def train_jal(
    env: MultiAgentDiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    seed: int,
    setting: str,
    mode: str,
) -> List[EpisodeLog]:
    rng = np.random.default_rng(seed)
    jal = JointActionLearner(env=env, alpha=alpha, gamma=gamma)
    logs: List[EpisodeLog] = []

    for ep in range(episodes):
        eps = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        steps = 0
        td_vals = []
        team_reward = {0: 0.0, 1: 0.0}
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            state = jal.team_state(obs)
            joint_idx = jal.choose_joint_action(state, eps, rng)
            acts = jal.build_actions(joint_idx)

            out = env.step(acts)
            next_state = jal.team_state(out.observations)

            r_team0 = float(np.mean([out.rewards[aid] for aid in env.team_agents(0)]))
            q_sa = jal.q_table[state][joint_idx]
            target = r_team0 + gamma * (0.0 if out.done else np.max(jal.q_table[next_state]))
            td = target - q_sa
            jal.q_table[state][joint_idx] += alpha * td

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_vals.append(abs(td))
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(
            EpisodeLog(
                algorithm="jal",
                setting=setting,
                mode=mode,
                seed=seed,
                episode=ep,
                team0_reward=float(team_reward[0]),
                team1_reward=float(team_reward[1]),
                team0_scored=int(info_last.get("team0_scored", 0.0) > 0.5),
                team1_scored=int(info_last.get("team1_scored", 0.0) > 0.5),
                episode_length=steps,
                mean_abs_td_error=float(np.mean(td_vals) if td_vals else 0.0),
            )
        )

    return logs

