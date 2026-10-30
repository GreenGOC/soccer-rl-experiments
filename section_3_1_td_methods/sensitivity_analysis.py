"""Run Section 3-1 sensitivity analysis for Q-learning and SARSA.

Usage:
    python sensitivity_analysis.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv

from config import EnvConfig, ExperimentConfig, TrainConfig, RESULT_COLUMNS, ensure_output_dir
from environment_discrete import DiscreteSoccerEnv
from q_learning import train_q_learning
from sarsa import train_sarsa


def build_env(env_cfg: EnvConfig, seed: int) -> DiscreteSoccerEnv:
    return DiscreteSoccerEnv(
        grid_size=env_cfg.grid_size,
        max_steps=env_cfg.max_steps,
        seed=seed,
        reward_goal=env_cfg.reward_goal,
        reward_concede=env_cfg.reward_concede,
        reward_step=env_cfg.reward_step,
        reward_ball_progress=env_cfg.reward_ball_progress,
        reward_collision=env_cfg.reward_collision,
        reward_out=env_cfg.reward_out,
    )


def run_all() -> Path:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()
    exp_cfg = ExperimentConfig()

    out_dir = ensure_output_dir(exp_cfg.output_dir)
    csv_path = out_dir / "section_3_1_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for gamma in exp_cfg.gammas:
            for seed in exp_cfg.seeds:
                print(f"[Q-learning] gamma={gamma} seed={seed}")
                env_q = build_env(env_cfg, seed)

                _, q_stats = train_q_learning(
                    env=env_q,
                    episodes=train_cfg.episodes,
                    alpha=train_cfg.alpha,
                    gamma=gamma,
                    epsilon_start=train_cfg.epsilon_start,
                    epsilon_end=train_cfg.epsilon_end,
                    epsilon_decay_episodes=train_cfg.epsilon_decay_episodes,
                    seed=seed,
                )
                for s in q_stats:
                    writer.writerow(
                        {
                            "algorithm": "q_learning",
                            "gamma": gamma,
                            "seed": seed,
                            "episode": s.episode,
                            "episode_reward": s.episode_reward,
                            "success": s.success,
                            "episode_length": s.episode_length,
                        }
                    )

                print(f"[SARSA] gamma={gamma} seed={seed}")
                env_s = build_env(env_cfg, seed)

                _, s_stats = train_sarsa(
                    env=env_s,
                    episodes=train_cfg.episodes,
                    alpha=train_cfg.alpha,
                    gamma=gamma,
                    epsilon_start=train_cfg.epsilon_start,
                    epsilon_end=train_cfg.epsilon_end,
                    epsilon_decay_episodes=train_cfg.epsilon_decay_episodes,
                    seed=seed,
                )
                for s in s_stats:
                    writer.writerow(
                        {
                            "algorithm": "sarsa",
                            "gamma": gamma,
                            "seed": seed,
                            "episode": s.episode,
                            "episode_reward": s.episode_reward,
                            "success": s.success,
                            "episode_length": s.episode_length,
                        }
                    )

    # Save config snapshot for report reproducibility.
    config_snapshot = out_dir / "run_config.txt"
    config_snapshot.write_text(
        "\n".join(
            [
                "EnvConfig=" + str(asdict(env_cfg)),
                "TrainConfig=" + str(asdict(train_cfg)),
                "ExperimentConfig=" + str(asdict(exp_cfg)),
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved: {csv_path}")
    print("Next: python plot_results.py")
    return csv_path


if __name__ == "__main__":
    run_all()
