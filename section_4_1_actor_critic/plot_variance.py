"""Plot Section 4-1 learning and variance comparison vs Section 3-1.

Usage:
    python plot_variance.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import BASE_DIR, output_dir


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise RuntimeError(f"No rows in {path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def moving_average(x: np.ndarray, window: int = 500) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def aggregate_metric(rows, metric: str):
    episodes = np.unique(rows["episode"])
    means = []
    stds = []
    for ep in episodes:
        vals = rows[rows["episode"] == ep][metric].astype(np.float64)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return episodes, np.array(means), np.array(stds)


def filter_by_gamma(rows, gamma: float):
    gamma_vals = rows["gamma"].astype(np.float64)
    return rows[np.isclose(gamma_vals, float(gamma))]


def find_section_3_csv() -> Path:
    candidates = [
        BASE_DIR.parent / "section_3_1_td_methods" / "results" / "section_3_1_results.csv",
        BASE_DIR.parent / "section_3_1_td_methods" / "section_3_1_td_methods" / "results" / "section_3_1_results.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find Section 3-1 results CSV.")


def plot_actor_critic_learning(ac_rows, out_dir: Path) -> None:
    plt.figure(figsize=(20, 10))
    for gamma in sorted(set(ac_rows["gamma"].tolist())):
        subset = ac_rows[ac_rows["gamma"] == gamma]
        x, mean, std = aggregate_metric(subset, "episode_reward")
        mean_s = moving_average(mean)
        std_s = moving_average(std)
        plt.plot(x, mean_s, label=f"Actor-Critic gamma={gamma}")
        plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)

    plt.title("Section 4-1 Actor-Critic Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "actor_critic_learning_curve.png", dpi=200)
    plt.close()


def plot_actor_critic_variance_by_gamma(ac_rows, out_dir: Path) -> None:
    gammas = sorted(set(ac_rows["gamma"].tolist()))
    labels = []
    values = []
    summary_rows = []

    for gamma in gammas:
        subset = filter_by_gamma(ac_rows, float(gamma))
        if subset.size == 0:
            continue
        _, _, std = aggregate_metric(subset, "episode_reward")
        if std.size == 0:
            continue

        mean_episode_std = float(np.mean(std))
        labels.append(f"gamma={gamma}")
        values.append(mean_episode_std)
        summary_rows.append(
            {
                "algorithm": "actor_critic_tabular",
                "gamma": float(gamma),
                "metric": "episode_reward",
                "mean_episode_std": mean_episode_std,
                "max_episode_std": float(np.max(std)),
                "min_episode_std": float(np.min(std)),
            }
        )

    if not summary_rows:
        raise RuntimeError("No variance rows were computed for Section 4-1 Actor-Critic.")

    with (out_dir / "actor_critic_variance_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "gamma",
                "metric",
                "mean_episode_std",
                "max_episode_std",
                "min_episode_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.figure(figsize=(20, 10))
    x = np.arange(len(labels))
    bars = plt.bar(x, values)
    y_max = max(values) if values else 0.0
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01 * max(1.0, y_max),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean Std(Episode Reward) over Episodes")
    plt.title("Section 4-1 Actor-Critic Variance by Gamma")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "actor_critic_variance_by_gamma.png", dpi=200)
    plt.close()


def plot_variance_comparison(section3_rows, ac_rows, out_dir: Path) -> None:
    plt.figure(figsize=(20, 10))

    labels = []
    values = []
    summary_rows = []
    target_gamma = 0.89

    # Use common gamma=0.89 for direct variance comparison.
    for algo_key, label in [
        ("q_learning", "Q-learning"),
        ("sarsa", "SARSA"),
    ]:
        subset = section3_rows[(section3_rows["algorithm"] == algo_key) & np.isclose(section3_rows["gamma"].astype(np.float64), target_gamma)]
        if subset.size > 0:
            _, _, std = aggregate_metric(subset, "episode_reward")
            mean_episode_std = float(np.mean(std))
            labels.append(label)
            values.append(mean_episode_std)
            summary_rows.append(
                {
                    "algorithm": algo_key,
                    "gamma": target_gamma,
                    "metric": "episode_reward",
                    "mean_episode_std": mean_episode_std,
                }
            )

    ac_subset = filter_by_gamma(ac_rows, target_gamma)
    if ac_subset.size > 0:
        _, _, ac_std = aggregate_metric(ac_subset, "episode_reward")
        mean_episode_std = float(np.mean(ac_std))
        labels.append("Actor-Critic")
        values.append(mean_episode_std)
        summary_rows.append(
            {
                "algorithm": "actor_critic_tabular",
                "gamma": target_gamma,
                "metric": "episode_reward",
                "mean_episode_std": mean_episode_std,
            }
        )

    with (out_dir / "variance_comparison_gamma_0_89.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "gamma", "metric", "mean_episode_std"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    x = np.arange(len(labels))
    bars = plt.bar(x, values)
    y_max = max(values) if values else 0.0
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01 * max(1.0, y_max),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(x, labels)
    plt.ylabel("Mean Std(Episode Reward) over Episodes")
    plt.title("Variance Comparison at gamma=0.89")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "variance_comparison_q_sarsa_actor_critic.png", dpi=200)
    plt.close()


def plot_combined_curves(section3_rows, ac_rows, out_dir: Path) -> None:
    plt.figure(figsize=(20, 10))
    target_gamma = 0.89

    combos = [
        (section3_rows, "q_learning", "Q-learning"),
        (section3_rows, "sarsa", "SARSA"),
        (ac_rows, "actor_critic_tabular", "Actor-Critic"),
    ]

    for rows, key, label in combos:
        subset = rows[(rows["algorithm"] == key) & np.isclose(rows["gamma"].astype(np.float64), target_gamma)]
        if subset.size == 0:
            continue
        x, mean, std = aggregate_metric(subset, "episode_reward")
        mean_s = moving_average(mean)
        std_s = moving_average(std)
        plt.plot(x, mean_s, label=label)
        plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)

    plt.title("Reward Curves Comparison at gamma=0.89")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve_comparison_gamma_0_89.png", dpi=200)
    plt.close()


def _outcome_counts_last_n(rows, last_n: int) -> tuple[int, int, int]:
    if rows.size == 0:
        return 0, 0, 0

    wins, losses, draws = 0, 0, 0
    seed_values = sorted(set(rows["seed"].tolist()))
    for seed in seed_values:
        seed_rows = rows[rows["seed"] == seed]
        if seed_rows.size == 0:
            continue
        ordered = np.sort(seed_rows, order="episode")
        tail = ordered[-last_n:] if ordered.size > last_n else ordered
        reward = tail["episode_reward"].astype(np.float64)
        success = tail["success"].astype(np.int32)

        wins += int(np.sum(success == 1))
        no_win = success == 0
        losses += int(np.sum(no_win & (reward <= -50.0)))
        draws += int(np.sum(no_win & (reward > -50.0)))

    return wins, losses, draws


def plot_outcomes_last_n(ac_rows, out_dir: Path, last_n: int = 2000) -> None:
    gammas = sorted(set(ac_rows["gamma"].tolist()))

    labels = []
    values = []
    summary_rows = []

    for gamma in gammas:
        subset = filter_by_gamma(ac_rows, float(gamma))
        if subset.size == 0:
            continue
        wins, losses, draws = _outcome_counts_last_n(subset, last_n=last_n)
        labels.append(f"gamma={gamma}")
        values.append((wins, losses, draws))
        summary_rows.append(
            {
                "algorithm": "actor_critic_tabular",
                "gamma": float(gamma),
                "last_n": int(last_n),
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
            }
        )

    with (out_dir / "actor_critic_outcomes_last_2000.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "gamma", "last_n", "wins", "losses", "draws"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    n = len(labels)
    cols = min(3, max(1, n))
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5 * rows_n))
    axes = np.array(axes).reshape(-1)
    colors = ["green", "red", "gray"]
    legend_labels = ["Wins", "Losses", "Draws"]

    for i, (label, (wins, losses, draws)) in enumerate(zip(labels, values)):
        ax = axes[i]
        sizes = [wins, losses, draws]
        total = sum(sizes)
        if total <= 0:
            sizes = [1, 0, 0]
            total = 1
        ax.pie(
            sizes,
            labels=legend_labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
        )
        ax.set_title(f"{label}\n(last {last_n})")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Section 4-1 Outcome Percentages in Last {last_n} Games", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "actor_critic_outcomes_last_2000.png", dpi=200)
    plt.close()


def main() -> None:
    out_dir = output_dir()

    ac_csv = out_dir / "section_4_1_actor_critic_results.csv"
    s3_csv = find_section_3_csv()

    ac_rows = load_csv(ac_csv)
    section3_rows = load_csv(s3_csv)

    plot_actor_critic_learning(ac_rows, out_dir)
    plot_actor_critic_variance_by_gamma(ac_rows, out_dir)
    plot_variance_comparison(section3_rows, ac_rows, out_dir)
    plot_combined_curves(section3_rows, ac_rows, out_dir)
    plot_outcomes_last_n(ac_rows, out_dir, last_n=2000)

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
