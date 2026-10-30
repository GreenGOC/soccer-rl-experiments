"""Configuration for Section 5-1 MARL in discrete state/action spaces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_MODES = ("2v2", "3v3")


def normalize_mode(mode: str) -> str:
    m = mode.strip().lower().replace(" ", "")
    if "x" in m:
        m = m.replace("x", "v")
    if m.isdigit():
        m = f"{m}v{m}"
    if m not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode='{mode}'. Supported: {SUPPORTED_MODES}")
    return m


def parse_mode(mode: str) -> int:
    m = normalize_mode(mode)
    return int(m.split("v")[0])


def grid_size_for_mode(mode: str) -> int:
    return 12


@dataclass(frozen=True)
class EnvConfig:
    mode: str = "2v2"
    grid_size: int = 12
    max_steps: int = 250
    agents_per_team: int = 2
    seed: int = 0

    reward_goal: float = 100.0
    reward_concede: float = -100.0
    reward_step: float = -0.03
    reward_ball_progress: float = 0.15
    reward_collision: float = -0.5
    reward_out: float = -0.5


@dataclass(frozen=True)
class DistributedQLearningConfig:
    episodes: int = 8000
    alpha: float = 0.1
    gamma: float = 0.89
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1000
    consensus_tau: float = 0.2


@dataclass(frozen=True)
class FriendOrFoeConfig:
    episodes: int = 8000
    alpha: float = 0.08
    gamma: float = 0.89
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 900


@dataclass(frozen=True)
class MinimaxQConfig:
    episodes: int = 8000
    alpha: float = 0.08
    gamma: float = 0.89
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 900


@dataclass(frozen=True)
class FMQConfig:
    episodes: int = 8000
    alpha: float = 0.08
    gamma: float = 0.89
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 900
    fmq_c: float = 0.5


@dataclass(frozen=True)
class JALConfig:
    episodes: int = 8000
    alpha: float = 0.08
    gamma: float = 0.89
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 900


@dataclass(frozen=True)
class ExperimentConfig:
    modes: Tuple[str, ...] = SUPPORTED_MODES
    seeds: Tuple[int, ...] = (0, 1)
    output_dir_name: str = "results"


RESULT_COLUMNS = [
    "algorithm",
    "setting",
    "mode",
    "seed",
    "episode",
    "team0_reward",
    "team1_reward",
    "team0_scored",
    "team1_scored",
    "episode_length",
    "mean_abs_td_error",
]


def output_dir() -> Path:
    out = BASE_DIR / ExperimentConfig().output_dir_name
    out.mkdir(parents=True, exist_ok=True)
    return out
