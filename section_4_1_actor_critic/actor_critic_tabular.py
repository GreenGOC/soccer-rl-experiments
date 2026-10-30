"""Tabular Actor-Critic for Section 4-1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple
import numpy as np

from environment_discrete import DiscreteSoccerEnv


State = Tuple[int, int, int, int, int]


@dataclass
class EpisodeStat:
    episode: int
    episode_reward: float
    success: int
    episode_length: int


class TabularActorCritic:
    def __init__(self, n_actions: int, alpha_v: float, alpha_pi: float, gamma: float, entropy_beta: float = 0.0) -> None:
        self.n_actions = n_actions
        self.alpha_v = alpha_v
        self.alpha_pi = alpha_pi
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        self.values: DefaultDict[State, float] = defaultdict(float)
        self.preferences: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def policy(self, state: State) -> np.ndarray:
        pref = self.preferences[state]
        shifted = pref - np.max(pref)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)
        return probs

    def act(self, state: State, rng: np.random.Generator) -> int:
        probs = self.policy(state)
        return int(rng.choice(self.n_actions, p=probs))

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool) -> float:
        v_s = self.values[state]
        v_next = 0.0 if done else self.values[next_state]
        td_error = reward + self.gamma * v_next - v_s

        # Critic update.
        self.values[state] += self.alpha_v * td_error

        # Actor update via policy gradient with baseline (advantage ~= TD error).
        probs = self.policy(state)
        grad_log_pi = -probs
        grad_log_pi[action] += 1.0
        self.preferences[state] += self.alpha_pi * td_error * grad_log_pi

        if self.entropy_beta > 0.0:
            entropy_grad = -np.log(np.clip(probs, 1e-8, 1.0)) - 1.0
            self.preferences[state] += self.alpha_pi * self.entropy_beta * entropy_grad

        return float(td_error)


def train_actor_critic(
    env: DiscreteSoccerEnv,
    episodes: int,
    alpha_v: float,
    alpha_pi: float,
    gamma: float,
    entropy_beta: float,
    seed: int,
) -> Tuple[TabularActorCritic, List[EpisodeStat]]:
    rng = np.random.default_rng(seed)
    agent = TabularActorCritic(
        n_actions=env.n_actions,
        alpha_v=alpha_v,
        alpha_pi=alpha_pi,
        gamma=gamma,
        entropy_beta=entropy_beta,
    )

    stats: List[EpisodeStat] = []

    for ep in range(episodes):
        state = env.reset(seed=seed * 100_000 + ep)
        done = False
        total_reward = 0.0
        steps = 0
        success = 0

        while not done and steps < env.max_steps:
            action = agent.act(state, rng)
            result = env.step(action)
            agent.update(state, action, result.reward, result.next_state, result.done)

            state = result.next_state
            total_reward += result.reward
            steps += 1
            done = result.done
            if result.info.get("scored", 0.0) > 0.5:
                success = 1

        stats.append(EpisodeStat(ep, total_reward, success, steps))

    return agent, stats
