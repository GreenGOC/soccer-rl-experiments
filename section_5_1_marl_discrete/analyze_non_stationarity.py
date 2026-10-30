"""Analyze non-stationarity from Section 5-1 logs.

Usage:
    python analyze_non_stationarity.py
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


def summarize(rows, setting: str, mode: str, out_dir: Path):
    if rows is None:
        return

    plt.figure(figsize=(20, 10))
    for algo in sorted(set(rows["algorithm"].tolist())):
        subset = rows[rows["algorithm"] == algo]
        episodes = np.unique(subset["episode"])
        mean_td = []
        for ep in episodes:
            vals = subset[subset["episode"] == ep]["mean_abs_td_error"].astype(np.float64)
            mean_td.append(np.mean(vals))
        mean_td = np.array(mean_td)
        osc = moving_average(np.abs(np.diff(mean_td, prepend=mean_td[0])), 500)
        plt.plot(episodes, moving_average(mean_td), label=f"{algo} TD-error")
        plt.plot(episodes, osc, linestyle="--", alpha=0.7, label=f"{algo} oscillation")

    plt.title(f"Non-stationarity ({setting}, {mode})")
    plt.xlabel("Episode")
    plt.ylabel("Magnitude")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"non_stationarity_{setting}_{mode}.png", dpi=200)
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
        summarize(coop_mode, "cooperative", str(mode), out_dir)
        summarize(comp_mode, "competitive", str(mode), out_dir)

    print(f"Saved non-stationarity plots in: {out_dir}")


if __name__ == "__main__":
    main()
