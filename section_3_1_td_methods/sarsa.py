"""Tabular SARSA training utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Tuple
import numpy as np
from pathlib import Path

from environment_discrete import DiscreteSoccerEnv
from q_learning import EpisodeStat, linear_epsilon


QTable = DefaultDict[Tuple[int, int, int, int, int], np.ndarray]


def make_q_table(n_actions: int) -> QTable:
    return defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))


def choose_action(q_table: QTable, state: Tuple[int, int, int, int, int], epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_table[state])))
    return int(np.argmax(q_table[state]))


def train_sarsa(
    env: DiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    seed: int,
) -> Tuple[QTable, List[EpisodeStat]]:
    rng = np.random.default_rng(seed)
    q_table = make_q_table(env.n_actions)
    stats: List[EpisodeStat] = []

    log_path = Path("sarsa_log.txt")
    with log_path.open("w", encoding="utf-8") as log_file:
        for ep in range(episodes):
            epsilon = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
            state = env.reset(seed=seed * 100_000 + ep)
            action = choose_action(q_table, state, epsilon, rng)

            done = False
            total_reward = 0.0
            steps = 0
            success = 0

            while not done and steps < env.max_steps:
                result = env.step(action)
                next_state = result.next_state

                next_epsilon = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
                next_action = choose_action(q_table, next_state, next_epsilon, rng)

                # Log positions and action
                log_file.write(
                    f"Episode {ep}, Step {steps}, Action {action}, Agent Pos {env.agent_pos}, Opponent Pos {env.opponent_pos}, Ball Pos {env.ball_pos}\n"
                )

                td_target = result.reward + gamma * q_table[next_state][next_action] * (0.0 if result.done else 1.0)
                td_error = td_target - q_table[state][action]
                q_table[state][action] += alpha * td_error

                state = next_state
                action = next_action
                total_reward += result.reward
                done = result.done
                steps += 1

                if result.info.get("scored", 0.0) > 0.5:
                    success = 1

            stats.append(EpisodeStat(ep, total_reward, success, steps))

    return q_table, stats
