"""Heuristic baselines for Section 8-1.

Generates:
- baseline_discrete_results.csv (section 6-1-1)
- baseline_marl_results.csv (section 7-1 for all supported modes)
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from config import RESULT_DIR, SUPPORTED_MODES

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_modes import parse_mode
from section_6_1_1_continuous_state_discrete_action.environment_continuous_state import (
    ContinuousStateDiscreteActionSoccerEnv,
)
from section_7_1_deep_marl.environment_continuous_action import MultiAgentContinuousSoccerEnv


def heuristic_action_discrete(obs: np.ndarray) -> int:
    ball_rel = obs[12:14]
    has_ball = obs[16] > 0.5
    if has_ball:
        return 4
    dx, dy = float(ball_rel[0]), float(ball_rel[1])
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2
    return 0 if dy > 0 else 1


def run_discrete_baseline(episodes: int = 8000, seeds=(0, 1, 2, 3, 4)) -> Path:
    out_csv = RESULT_DIR / "baseline_discrete_results.csv"
    cols = ["algorithm", "mode", "seed", "episode", "timestep", "episode_reward", "success", "episode_length"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for seed in seeds:
            env = ContinuousStateDiscreteActionSoccerEnv(seed=seed)
            t = 0
            for ep in range(episodes):
                obs = env.reset(seed=seed * 100_000 + ep)
                done = False
                total_r = 0.0
                length = 0
                success = 0

                while not done and length < env.max_steps:
                    act = heuristic_action_discrete(obs)
                    out = env.step(act)
                    obs = out.obs
                    total_r += out.reward
                    length += 1
                    t += 1
                    done = out.done
                    if out.info.get("scored", 0.0) > 0.5:
                        success = 1

                w.writerow(
                    {
                        "algorithm": "heuristic_discrete",
                        "mode": "single-agent",
                        "seed": seed,
                        "episode": ep,
                        "timestep": t,
                        "episode_reward": total_r,
                        "success": success,
                        "episode_length": length,
                    }
                )

    return out_csv


def heuristic_action_marl(env: MultiAgentContinuousSoccerEnv, agent_idx: int) -> np.ndarray:
    pos = env.team0_pos[agent_idx]
    ball = env.ball_pos
    if env.ball_owner == (0, agent_idx):
        return np.array([1.0, 0.0, 1.0], dtype=np.float32)

    d = ball - pos
    n = np.linalg.norm(d)
    if n > 1e-6:
        d = d / n
    return np.array([d[0], d[1], 0.0], dtype=np.float32)


def run_marl_baseline(episodes: int = 8000, seeds=(0, 1, 2, 3, 4)) -> Path:
    out_csv = RESULT_DIR / "baseline_marl_results.csv"
    cols = [
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
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for mode in SUPPORTED_MODES:
            _, team_size = parse_mode(mode)
            for seed in seeds:
                env = MultiAgentContinuousSoccerEnv(team_size=team_size, seed=seed)
                t = 0
                for ep in range(episodes):
                    _, _ = env.reset(seed=seed * 100_000 + ep)
                    done = False
                    ep_r0 = 0.0
                    ep_r1 = 0.0
                    ep_len = 0
                    ep_pass = 0.0
                    success = 0

                    while not done and ep_len < env.max_steps:
                        actions = {aid: heuristic_action_marl(env, i) for i, aid in enumerate(env.controlled_ids)}
                        out = env.step(actions)

                        ep_r0 += float(out.info["team0_reward"])
                        ep_r1 += float(out.info["team1_reward"])
                        ep_pass += float(out.info.get("pass_count", 0.0))
                        if out.info.get("success", 0.0) > 0.5:
                            success = 1

                        done = out.done
                        ep_len += 1
                        t += 1

                    w.writerow(
                        {
                            "algorithm": "heuristic_marl",
                            "mode": mode,
                            "seed": seed,
                            "episode": ep,
                            "timestep": t,
                            "team0_reward": ep_r0,
                            "team1_reward": ep_r1,
                            "success": success,
                            "episode_length": ep_len,
                            "pass_count": ep_pass,
                        }
                    )

    return out_csv


def main() -> None:
    d = run_discrete_baseline()
    m = run_marl_baseline()
    print(f"Saved {d}")
    print(f"Saved {m}")


if __name__ == "__main__":
    main()
