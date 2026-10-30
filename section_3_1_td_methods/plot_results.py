"""Plot Section 3-1 results with confidence bands.

Usage:
    python plot_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import ExperimentConfig


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def load_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise RuntimeError(f"No rows in {path}")
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def aggregate_by_episode(rows, metric: str):
    episodes = np.unique(rows["episode"])
    means = []
    stds = []
    for ep in episodes:
        vals = rows[rows["episode"] == ep][metric].astype(np.float64)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return episodes, np.array(means), np.array(stds)


def plot_metric(rows, metric: str, ylabel: str, out_path: Path, smooth_window: int = 500):
    plt.figure(figsize=(20, 10))

    algorithms = sorted(set(rows["algorithm"].tolist()))
    gammas = sorted(set(rows["gamma"].tolist()))

    for algo in algorithms:
        for gamma in gammas:
            subset = rows[(rows["algorithm"] == algo) & (rows["gamma"] == gamma)]
            if subset.size == 0:
                continue

            x, mean, std = aggregate_by_episode(subset, metric)
            mean_s = moving_average(mean, smooth_window)
            std_s = moving_average(std, smooth_window)

            label = f"{algo}, gamma={gamma}"
            plt.plot(x, mean_s, label=label)
            plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"Section 3-1: {ylabel} (mean +/- std over seeds)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_variance_summary(rows, metric: str, out_plot_path: Path, out_csv_path: Path) -> None:
    algorithms = sorted(set(rows["algorithm"].tolist()))
    gammas = sorted(set(rows["gamma"].tolist()))

    labels = []
    values = []
    summary_rows = []

    for algo in algorithms:
        for gamma in gammas:
            subset = rows[(rows["algorithm"] == algo) & (rows["gamma"] == gamma)]
            if subset.size == 0:
                continue

            _, _, std = aggregate_by_episode(subset, metric)
            if std.size == 0:
                continue

            mean_episode_std = float(np.mean(std))
            labels.append(f"{algo}\\ngamma={gamma}")
            values.append(mean_episode_std)
            summary_rows.append(
                {
                    "algorithm": algo,
                    "gamma": float(gamma),
                    "metric": metric,
                    "mean_episode_std": mean_episode_std,
                    "max_episode_std": float(np.max(std)),
                    "min_episode_std": float(np.min(std)),
                }
            )

    if not summary_rows:
        raise RuntimeError("No variance rows were computed for Section 3-1.")

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
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
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Mean Std(Episode Reward) over Episodes")
    plt.title("Section 3-1 Variance Summary")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(0.0, y_max * 1.12 if y_max > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
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


def plot_outcomes_last_n(rows, out_plot_path: Path, out_csv_path: Path, last_n: int = 2000) -> None:
    algorithms = sorted(set(rows["algorithm"].tolist()))
    gammas = sorted(set(rows["gamma"].tolist()))

    labels = []
    values = []
    summary_rows = []

    for algo in algorithms:
        for gamma in gammas:
            subset = rows[(rows["algorithm"] == algo) & (rows["gamma"] == gamma)]
            if subset.size == 0:
                continue
            wins, losses, draws = _outcome_counts_last_n(subset, last_n=last_n)
            labels.append(f"{algo}, gamma={gamma}")
            values.append((wins, losses, draws))
            summary_rows.append(
                {
                    "algorithm": algo,
                    "gamma": float(gamma),
                    "last_n": int(last_n),
                    "wins": int(wins),
                    "losses": int(losses),
                    "draws": int(draws),
                }
            )

    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
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

    fig.suptitle(f"Section 3-1 Outcome Percentages in Last {last_n} Games", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200)
    plt.close()


def main() -> None:
    exp_cfg = ExperimentConfig()
    out_dir = Path(exp_cfg.output_dir)
    csv_path = out_dir / "section_3_1_results.csv"

    rows = load_csv(csv_path)

    plot_metric(
        rows,
        metric="episode_reward",
        ylabel="Episode Reward",
        out_path=out_dir / "learning_curve_reward.png",
        smooth_window=500,
    )

    plot_metric(
        rows,
        metric="success",
        ylabel="Success Rate (per episode)",
        out_path=out_dir / "learning_curve_success.png",
        smooth_window=500,
    )

    plot_metric(
        rows,
        metric="episode_length",
        ylabel="Episode Length",
        out_path=out_dir / "learning_curve_length.png",
        smooth_window=500,
    )

    plot_variance_summary(
        rows,
        metric="episode_reward",
        out_plot_path=out_dir / "variance_summary_reward.png",
        out_csv_path=out_dir / "variance_summary_reward.csv",
    )
    plot_outcomes_last_n(
        rows,
        out_plot_path=out_dir / "outcomes_last_2000.png",
        out_csv_path=out_dir / "outcomes_last_2000.csv",
        last_n=2000,
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
