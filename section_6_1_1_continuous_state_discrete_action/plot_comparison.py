"""Plot reward/success/length comparison for Section 6-1-1.

Usage:
    python plot_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESULT_DIR = Path(__file__).resolve().parent / "results"


def load_rows(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        return None
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def moving_average(x: np.ndarray, w: int = 500) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, kernel, mode="same")


def aggregate(rows, metric: str):
    episodes = np.unique(rows["episode"])
    mean, std = [], []
    for ep in episodes:
        vals = rows[rows["episode"] == ep][metric].astype(np.float64)
        mean.append(np.mean(vals))
        std.append(np.std(vals))
    return episodes, np.array(mean), np.array(std)


def plot_metric(dqn, ppo, metric: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(20, 10))
    for rows, label in [(dqn, "DQN"), (ppo, "PPO")]:
        x, m, s = aggregate(rows, metric)
        ms, ss = moving_average(m), moving_average(s)
        plt.plot(x, ms, label=label)
        plt.fill_between(x, ms - ss, ms + ss, alpha=0.2)

    plt.title(f"Section 6-1-1 {ylabel}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / out_name, dpi=200)
    plt.close()


def main() -> None:
    dqn = load_rows(RESULT_DIR / "dqn_results.csv")
    ppo = load_rows(RESULT_DIR / "ppo_results.csv")

    if dqn is None or ppo is None:
        raise RuntimeError("Missing dqn_results.csv or ppo_results.csv in results/")

    dqn_modes = set(np.unique(dqn["mode"]).tolist())
    ppo_modes = set(np.unique(ppo["mode"]).tolist())
    modes = sorted(dqn_modes.intersection(ppo_modes))
    if not modes:
        raise RuntimeError("No overlapping modes found between dqn_results.csv and ppo_results.csv.")

    for mode in modes:
        dqn_m = dqn[dqn["mode"] == mode]
        ppo_m = ppo[ppo["mode"] == mode]
        plot_metric(dqn_m, ppo_m, "episode_reward", f"Episode Reward ({mode})", f"comparison_reward_{mode}.png")
        plot_metric(dqn_m, ppo_m, "success", f"Success Rate ({mode})", f"comparison_success_{mode}.png")
        plot_metric(
            dqn_m,
            ppo_m,
            "episode_length",
            f"Episode Length ({mode})",
            f"comparison_episode_length_{mode}.png",
        )

    print(f"Saved plots in {RESULT_DIR}")


if __name__ == "__main__":
    main()
