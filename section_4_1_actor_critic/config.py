"""Configuration for Section 4-1 tabular Actor-Critic experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent


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
    alpha_v: float = 0.1
    alpha_pi: float = 0.03
    entropy_beta: float = 0.001


@dataclass(frozen=True)
class ExperimentConfig:
    gammas: Tuple[float, ...] = (0.1, 0.5, 0.89, 0.3, 0.7, 0.99)
    seeds: Tuple[int, ...] = (0, 1)
    output_dir_name: str = "results"


RESULT_COLUMNS = [
    "algorithm",
    "gamma",
    "seed",
    "episode",
    "episode_reward",
    "success",
    "episode_length",
]


def output_dir() -> Path:
    out = BASE_DIR / ExperimentConfig().output_dir_name
    out.mkdir(parents=True, exist_ok=True)
    return out
