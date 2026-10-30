"""Generate hyperparameter tables (CSV + Markdown) for Section 8-1 report."""

from __future__ import annotations

import csv
from pathlib import Path

from config import RESULT_DIR


def hyperparameter_rows():
    return [
        # Section 3-1
        ("q_learning", "alpha", "0.1"),
        ("q_learning", "gamma", "0.1/0.5/0.89"),
        ("q_learning", "epsilon", "1.0 -> 0.05 linear"),
        ("sarsa", "alpha", "0.1"),
        ("sarsa", "gamma", "0.1/0.5/0.89"),
        ("sarsa", "epsilon", "1.0 -> 0.05 linear"),
        # Section 4-1
        ("actor_critic_tabular", "alpha_v", "0.1"),
        ("actor_critic_tabular", "alpha_pi", "0.05"),
        ("actor_critic_tabular", "gamma", "0.1/0.5/0.89"),
        # Section 5-1
        ("iql", "alpha", "0.1"),
        ("iql", "gamma", "0.89"),
        ("jal", "alpha", "0.08"),
        ("jal", "gamma", "0.89"),
        # Section 6-1
        ("dqn", "learning_rate", "3e-4"),
        ("dqn", "batch_size", "128"),
        ("dqn", "gamma", "0.99"),
        ("ppo_discrete", "learning_rate", "3e-4"),
        ("ppo_discrete", "clip_ratio", "0.2"),
        ("ppo_discrete", "gae_lambda", "0.95"),
        ("ddpg", "lr_actor/lr_critic", "3e-4 / 3e-4"),
        ("ddpg", "gamma", "0.99"),
        ("ddpg", "tau", "0.01"),
        ("sac", "learning_rate", "3e-4"),
        ("sac", "gamma", "0.99"),
        ("sac", "alpha", "0.2"),
        # Section 7-1
        ("ctde_mappo", "lr_actor/lr_critic", "3e-4 / 3e-4"),
        ("ctde_mappo", "clip_ratio", "0.2"),
        ("ctde_mappo", "gae_lambda", "0.95"),
        ("ippo", "learning_rate", "3e-4"),
        ("ippo", "clip_ratio", "0.2"),
        ("ippo", "gae_lambda", "0.95"),
    ]


def main() -> None:
    csv_path = RESULT_DIR / "hyperparameters.csv"
    md_path = RESULT_DIR / "hyperparameters.md"

    rows = hyperparameter_rows()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "parameter", "value"])
        w.writerows(rows)

    lines = ["# Hyperparameter Tables", "", "| Algorithm | Parameter | Value |", "|---|---|---|"]
    for a, p, v in rows:
        lines.append(f"| {a} | {p} | {v} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
