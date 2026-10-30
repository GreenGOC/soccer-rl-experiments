"""Compare sample efficiency for DQN vs PPO in Section 6-1-1.

Usage:
    python compare_sample_efficiency.py
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



def aggregate_by_episode(rows, metric: str):
    episodes = np.unique(rows["episode"])
    means = []
    stds = []
    tmeans = []
    for ep in episodes:
        part = rows[rows["episode"] == ep]
        means.append(np.mean(part[metric].astype(np.float64)))
        stds.append(np.std(part[metric].astype(np.float64)))
        tmeans.append(np.mean(part["timestep"].astype(np.float64)))
    return np.array(episodes), np.array(tmeans), np.array(means), np.array(stds)


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

        plt.figure(figsize=(20, 10))
        for rows, label in [(dqn_m, "DQN"), (ppo_m, "PPO")]:
            _, steps, mean, std = aggregate_by_episode(rows, "episode_reward")
            m = moving_average(mean)
            s = moving_average(std)
            plt.plot(steps, m, label=label)
            plt.fill_between(steps, m - s, m + s, alpha=0.2)

        plt.title(f"Section 6-1-1 Sample Efficiency ({mode})")
        plt.xlabel("Environment Timesteps")
        plt.ylabel("Episode Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = RESULT_DIR / f"sample_efficiency_dqn_vs_ppo_{mode}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"Saved plots in {RESULT_DIR}")


if __name__ == "__main__":
    main()
