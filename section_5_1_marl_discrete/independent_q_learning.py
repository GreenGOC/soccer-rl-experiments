"""Independent Q-Learning for Section 5-1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple
import numpy as np

from environment_discrete_multi import MultiAgentDiscreteSoccerEnv


State = Tuple[int, ...]


@dataclass
class EpisodeLog:
    algorithm: str
    setting: str
    mode: str
    seed: int
    episode: int
    team0_reward: float
    team1_reward: float
    team0_scored: int
    team1_scored: int
    episode_length: int
    mean_abs_td_error: float


def linear_epsilon(ep: int, start: float, end: float, decay_eps: int) -> float:
    if ep >= decay_eps:
        return end
    frac = ep / max(1, decay_eps)
    return start + frac * (end - start)


class IndependentQLearner:
    def __init__(self, env: MultiAgentDiscreteSoccerEnv, alpha: float, gamma: float) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.q_tables: Dict[str, DefaultDict[State, np.ndarray]] = {
            aid: defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float64))
            for aid in env.agent_ids
        }

    def choose_actions(self, obs: Dict[str, State], epsilon: float, rng: np.random.Generator) -> Dict[str, int]:
        acts: Dict[str, int] = {}
        for aid, state in obs.items():
            q = self.q_tables[aid][state]
            if rng.random() < epsilon:
                acts[aid] = int(rng.integers(0, len(q)))
            else:
                acts[aid] = int(np.argmax(q))
        return acts

    def update(
        self,
        obs: Dict[str, State],
        acts: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, State],
        done: bool,
    ) -> float:
        td_errors = []
        for aid in self.env.agent_ids:
            s = obs[aid]
            a = acts[aid]
            r = rewards[aid]
            ns = next_obs[aid]

            q_sa = self.q_tables[aid][s][a]
            target = r + self.gamma * (0.0 if done else np.max(self.q_tables[aid][ns]))
            td = target - q_sa
            self.q_tables[aid][s][a] += self.alpha * td
            td_errors.append(abs(td))
        return float(np.mean(td_errors))


def train_iql(
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
    learner = IndependentQLearner(env=env, alpha=alpha, gamma=gamma)
    logs: List[EpisodeLog] = []

    for ep in range(episodes):
        epsilon = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        team_reward = {0: 0.0, 1: 0.0}
        steps = 0
        td_list = []
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            acts = learner.choose_actions(obs, epsilon, rng)
            out = env.step(acts)
            td_mag = learner.update(obs, acts, out.rewards, out.observations, out.done)

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_list.append(td_mag)
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(
            EpisodeLog(
                algorithm="iql",
                setting=setting,
                mode=mode,
                seed=seed,
                episode=ep,
                team0_reward=float(team_reward[0]),
                team1_reward=float(team_reward[1]),
                team0_scored=int(info_last.get("team0_scored", 0.0) > 0.5),
                team1_scored=int(info_last.get("team1_scored", 0.0) > 0.5),
                episode_length=steps,
                mean_abs_td_error=float(np.mean(td_list) if td_list else 0.0),
            )
        )

    return logs
