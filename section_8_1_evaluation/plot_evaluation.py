"""Plot unified Section 8-1 evaluation charts with confidence bands."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import RESULT_DIR


def load_unified(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def moving_average(x: np.ndarray, w: int = 500) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, kernel, mode="same")


def aggregate_curve(rows, x_key: str, y_key: str):
    x_vals = np.unique(rows[x_key])
    means, stds = [], []
    for xv in x_vals:
        vals = rows[rows[x_key] == xv][y_key].astype(np.float64)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(x_vals, dtype=np.float64), np.array(means), np.array(stds)


def plot_section(rows, section: str, metric: str, x_key: str, out_name: str, ylabel: str):
    plt.figure(figsize=(20, 10))
    sec = rows[rows["section"] == section]
    if sec.size == 0:
        plt.close()
        return

    # Group by algorithm+mode to keep mode curves separate.
    group_keys = sorted({(str(r["algorithm"]), str(r["mode"])) for r in sec})

    for algo, mode in group_keys:
        part = sec[(sec["algorithm"] == algo) & (sec["mode"] == mode)]
        x, m, s = aggregate_curve(part, x_key=x_key, y_key=metric)
        ms, ss = moving_average(m), moving_average(s)
        plt.plot(x, ms, label=f"{algo} [{mode}]")
        plt.fill_between(x, ms - ss, ms + ss, alpha=0.18)

    plt.title(f"Section {section} - {ylabel}")
    plt.xlabel("Timesteps" if x_key == "timestep" else "Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 0:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / out_name, dpi=200)
    plt.close()


def main() -> None:
    rows = load_unified(RESULT_DIR / "unified_metrics.csv")

    plot_section(rows, "3-1", "reward", "episode", "sec3_reward.png", "Reward")
    plot_section(rows, "4-1", "reward", "episode", "sec4_reward.png", "Reward")
    plot_section(rows, "5-1", "reward", "episode", "sec5_reward.png", "Team-0 Reward")
    plot_section(rows, "6-1", "reward", "timestep", "sec6_sample_efficiency.png", "Sample Efficiency (Reward)")
    plot_section(rows, "7-1", "reward", "timestep", "sec7_sample_efficiency.png", "Sample Efficiency (Team-0 Reward)")

    plot_section(rows, "6-1", "success", "episode", "sec6_success.png", "Success Rate")
    plot_section(rows, "7-1", "success", "episode", "sec7_success.png", "Success Rate")

    print(f"Saved evaluation plots in {RESULT_DIR}")


if __name__ == "__main__":
    main()
