"""Config for Section 7-1 deep MARL experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_MODES = ("2v2", "3v3", "5v5", "7v7", "8v8", "11v11", "13v13")


def parse_mode(mode: str) -> int:
    m = mode.strip().lower().replace(" ", "")
    if "x" in m:
        m = m.replace("x", "v")
    if m.isdigit():
        m = f"{m}v{m}"
    if m not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode='{mode}'. Supported: {SUPPORTED_MODES}")
    return int(m.split("v")[0])


@dataclass(frozen=True)
class EnvConfig:
    mode: str = "2v2"
    team_size: int = 2
    max_steps: int = 250
    seed: int = 0
    dt: float = 0.08


@dataclass(frozen=True)
class CTDEConfig:
    episodes: int = 8000
    rollout_horizon: int = 2000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    clip_ratio: float = 0.2
    update_epochs: int = 5
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5


@dataclass(frozen=True)
class IPPOConfig:
    episodes: int = 8000
    rollout_horizon: int = 2000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_ratio: float = 0.2
    update_epochs: int = 5
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: Tuple[int, ...] = (0, 1)
    output_dir: str = "results"


def result_dir() -> Path:
    out = BASE_DIR / ExperimentConfig().output_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


RESULT_COLUMNS = [
    "algorithm",
    "mode",
    "seed",
    "episode",
    "timestep",
    "team0_reward",
    "team1_reward",
    "success",
    "episode_length",
    "pass_count",
    "mean_loss",
]
