"""Run Section 4-1 tabular Actor-Critic experiments.

Usage:
    python run_experiments.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv

from config import EnvConfig, TrainConfig, ExperimentConfig, RESULT_COLUMNS, output_dir
from environment_discrete import DiscreteSoccerEnv
from actor_critic_tabular import train_actor_critic


def build_env(cfg: EnvConfig, seed: int) -> DiscreteSoccerEnv:
    return DiscreteSoccerEnv(
        grid_size=cfg.grid_size,
        max_steps=cfg.max_steps,
        seed=seed,
        reward_goal=cfg.reward_goal,
        reward_concede=cfg.reward_concede,
        reward_step=cfg.reward_step,
        reward_ball_progress=cfg.reward_ball_progress,
        reward_collision=cfg.reward_collision,
        reward_out=cfg.reward_out,
    )


def run() -> Path:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()
    exp_cfg = ExperimentConfig()

    out_dir = output_dir()
    csv_path = out_dir / "section_4_1_actor_critic_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for gamma in exp_cfg.gammas:
            for seed in exp_cfg.seeds:
                print(f"[Actor-Critic] gamma={gamma} seed={seed}")
                env = build_env(env_cfg, seed)
                _, stats = train_actor_critic(
                    env=env,
                    episodes=train_cfg.episodes,
                    alpha_v=train_cfg.alpha_v,
                    alpha_pi=train_cfg.alpha_pi,
                    gamma=gamma,
                    entropy_beta=train_cfg.entropy_beta,
                    seed=seed,
                )

                for s in stats:
                    writer.writerow(
                        {
                            "algorithm": "actor_critic_tabular",
                            "gamma": gamma,
                            "seed": seed,
                            "episode": s.episode,
                            "episode_reward": s.episode_reward,
                            "success": s.success,
                            "episode_length": s.episode_length,
                        }
                    )

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
    print("Next: python plot_variance.py")
    return csv_path


if __name__ == "__main__":
    run()
