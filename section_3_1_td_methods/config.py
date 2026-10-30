"""Configuration for Section 3-1 TD methods experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 12
    max_steps: int = 250
    seed: int = 0

    # Rewards
    reward_goal: float = 100.0
    reward_concede: float = -100.0
    reward_step: float = -0.05
    reward_ball_progress: float = 0.2
    reward_collision: float = -1.0
    reward_out: float = -1.0


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 8000
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1500
    max_steps_per_episode: int = 250


@dataclass(frozen=True)
class ExperimentConfig:
    gammas: Tuple[float, ...] = (0.1, 0.5, 0.89, 0.3, 0.7, 0.99)
    seeds: Tuple[int, ...] = (0, 1)
    output_dir: str = "section_3_1_td_methods/results"


RESULT_COLUMNS = [
    "algorithm",
    "gamma",
    "seed",
    "episode",
    "episode_reward",
    "success",
    "episode_length",
]


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
