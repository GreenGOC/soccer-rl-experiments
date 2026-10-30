"""Compare CTDE vs Independent training for Section 7-1.

Usage:
    python compare_ctde_vs_independent.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESULT_DIR = Path(__file__).resolve().parent / "results"


def load_rows(path: Path):
    if not path.exists():
        return None
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
    eps = np.unique(rows["episode"])
    means = []
    stds = []
    steps = []
    for ep in eps:
        part = rows[rows["episode"] == ep]
        vals = part[metric].astype(np.float64)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        steps.append(np.mean(part["timestep"].astype(np.float64)))
    return np.array(eps), np.array(steps), np.array(means), np.array(stds)


def plot_metric(ctde, indep, metric: str, ylabel: str, out_name: str, use_steps: bool = False):
    plt.figure(figsize=(20, 10))
    for rows, label in [(ctde, "CTDE (MAPPO-style)"), (indep, "Independent (IPPO)")]:
        if rows is None:
            continue
        eps, steps, m, s = aggregate(rows, metric)
        x = steps if use_steps else eps
        ms, ss = moving_average(m), moving_average(s)
        plt.plot(x, ms, label=label)
        plt.fill_between(x, ms - ss, ms + ss, alpha=0.2)

    plt.title(f"Section 7-1: {ylabel}")
    plt.xlabel("Timesteps" if use_steps else "Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / out_name, dpi=200)
    plt.close()


def main() -> None:
    ctde = load_rows(RESULT_DIR / "ctde_results.csv")
    indep = load_rows(RESULT_DIR / "independent_results.csv")

    if ctde is None or indep is None:
        raise RuntimeError("Need unified result files: ctde_results.csv and independent_results.csv.")

    ctde_modes = set(np.unique(ctde["mode"]).tolist())
    indep_modes = set(np.unique(indep["mode"]).tolist())
    modes = sorted(ctde_modes.intersection(indep_modes))
    if not modes:
        raise RuntimeError("No overlapping modes found between ctde and independent result files.")

    for mode in modes:
        ctde_mode = ctde[ctde["mode"] == mode]
        indep_mode = indep[indep["mode"] == mode]
        if len(ctde_mode) == 0 or len(indep_mode) == 0:
            continue

        plot_metric(ctde_mode, indep_mode, "team0_reward", f"Team-0 Reward ({mode})", f"compare_reward_{mode}.png")
        plot_metric(ctde_mode, indep_mode, "success", f"Success Rate ({mode})", f"compare_success_{mode}.png")
        plot_metric(ctde_mode, indep_mode, "pass_count", f"Pass Count ({mode})", f"compare_pass_count_{mode}.png")
        plot_metric(
            ctde_mode,
            indep_mode,
            "team0_reward",
            f"Sample Efficiency (Team-0 Reward, {mode})",
            f"compare_sample_efficiency_{mode}.png",
            use_steps=True,
        )

    print(f"Saved plots in {RESULT_DIR}")


if __name__ == "__main__":
    main()
