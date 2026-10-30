"""Plot Section 5-1 cooperative vs competitive MARL results.

Usage:
    python plot_results.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import output_dir


def load_csv(path: Path):
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return None
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def moving_average(x: np.ndarray, window: int = 500) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def aggregate(rows, metric: str):
    episodes = np.unique(rows["episode"])
    means = []
    stds = []
    for ep in episodes:
        vals = rows[rows["episode"] == ep][metric].astype(np.float64)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return episodes, np.array(means), np.array(stds)


def plot_setting(rows, setting: str, mode: str, out_dir: Path):
    if rows is None:
        return

    plt.figure(figsize=(20, 10))
    for algo in sorted(set(rows["algorithm"].tolist())):
        subset = rows[rows["algorithm"] == algo]
        x, mean, std = aggregate(subset, "team0_reward")
        m, s = moving_average(mean), moving_average(std)
        plt.plot(x, m, label=algo)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.title(f"Team-0 Reward ({setting}, {mode})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"learning_curve_team0_reward_{setting}_{mode}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(20, 10))
    for algo in sorted(set(rows["algorithm"].tolist())):
        subset = rows[rows["algorithm"] == algo]
        x, mean, std = aggregate(subset, "team0_scored")
        m, s = moving_average(mean), moving_average(std)
        plt.plot(x, m, label=algo)
        plt.fill_between(x, np.clip(m - s, 0.0, 1.0), np.clip(m + s, 0.0, 1.0), alpha=0.2)
    plt.title(f"Team-0 Success Rate ({setting}, {mode})")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"learning_curve_success_{setting}_{mode}.png", dpi=200)
    plt.close()


def main() -> None:
    out_dir = output_dir()
    coop = load_csv(out_dir / "cooperative_results.csv")
    comp = load_csv(out_dir / "competitive_results.csv")
    if coop is None and comp is None:
        raise RuntimeError("Missing cooperative_results.csv and competitive_results.csv in results/.")

    modes = set()
    if coop is not None:
        modes.update(np.unique(coop["mode"]).tolist())
    if comp is not None:
        modes.update(np.unique(comp["mode"]).tolist())

    for mode in sorted(modes):
        coop_mode = None if coop is None else coop[coop["mode"] == mode]
        comp_mode = None if comp is None else comp[comp["mode"] == mode]
        plot_setting(coop_mode, "cooperative", str(mode), out_dir)
        plot_setting(comp_mode, "competitive", str(mode), out_dir)

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
