"""Plot Section 6-1-2 results and compare with Section 6-1-1 discrete-action results.

Usage:
    python plot_results.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
DISCRETE_RESULT_DIR = BASE_DIR.parent / "section_6_1_1_continuous_state_discrete_action" / "results"
ALL_MODE_KEY = "__all__"


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
    mean, std, steps = [], [], []
    for ep in eps:
        part = rows[rows["episode"] == ep]
        vals = part[metric].astype(np.float64)
        mean.append(np.mean(vals))
        std.append(np.std(vals))
        steps.append(np.mean(part["timestep"].astype(np.float64)))
    return np.array(eps), np.array(steps), np.array(mean), np.array(std)


def split_by_mode(rows):
    if rows is None:
        return {}
    if rows.dtype.names is None or "mode" not in rows.dtype.names:
        return {ALL_MODE_KEY: rows}
    out = {}
    for mode in np.unique(rows["mode"]):
        out[str(mode)] = rows[rows["mode"] == mode]
    return out


def concat_mode_groups(groups):
    if not groups:
        return None
    values = list(groups.values())
    if len(values) == 1:
        return values[0]
    return np.concatenate(values)


def plot_continuous_by_mode(ddpg_rows, sac_rows):
    ddpg_modes = split_by_mode(ddpg_rows)
    sac_modes = split_by_mode(sac_rows)
    modes = sorted(set(ddpg_modes).union(set(sac_modes)))
    for mode in modes:
        for algo, rows in [("ddpg", ddpg_modes.get(mode)), ("sac", sac_modes.get(mode))]:
            if rows is None or len(rows) == 0:
                continue
            plt.figure(figsize=(20, 10))
            _, _, m, s = aggregate(rows, "episode_reward")
            ms, ss = moving_average(m), moving_average(s)
            x = np.arange(len(ms))
            plt.plot(x, ms, label=f"{algo.upper()} {mode}")
            plt.fill_between(x, ms - ss, ms + ss, alpha=0.2)
            plt.title(f"Section 6-1-2 Reward ({algo.upper()}, {mode})")
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = RESULT_DIR / f"continuous_action_reward_{algo}_{mode}.png"
            plt.savefig(out, dpi=200)
            plt.close()


def plot_sample_efficiency_discrete_vs_continuous(dqn, ppo, continuous_rows, out_name: str):
    plt.figure(figsize=(20, 10))

    for rows, label in [(dqn, "DQN (disc act)"), (ppo, "PPO (disc act)")] + continuous_rows:
        if rows is None:
            continue
        _, steps, m, s = aggregate(rows, "episode_reward")
        ms, ss = moving_average(m), moving_average(s)
        plt.plot(steps, ms, label=label)
        plt.fill_between(steps, ms - ss, ms + ss, alpha=0.2)

    plt.title("Sample Efficiency: Discrete vs Continuous Actions")
    plt.xlabel("Environment Timesteps")
    plt.ylabel("Episode Reward")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = RESULT_DIR / out_name
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    ddpg = load_rows(RESULT_DIR / "ddpg_results.csv")
    sac = load_rows(RESULT_DIR / "sac_results.csv")

    dqn = load_rows(DISCRETE_RESULT_DIR / "dqn_results.csv")
    ppo = load_rows(DISCRETE_RESULT_DIR / "ppo_results.csv")

    if ddpg is None and sac is None:
        raise RuntimeError("No continuous-action result files found. Need ddpg_results.csv and/or sac_results.csv.")
    if dqn is None or ppo is None:
        raise RuntimeError("Missing Section 6-1-1 unified files dqn_results.csv or ppo_results.csv.")

    plot_continuous_by_mode(ddpg, sac)

    ddpg_modes = split_by_mode(ddpg)
    sac_modes = split_by_mode(sac)
    dqn_modes = split_by_mode(dqn)
    ppo_modes = split_by_mode(ppo)
    modes = sorted(set(ddpg_modes).union(set(sac_modes)).intersection(set(dqn_modes)).intersection(set(ppo_modes)))
    if modes:
        for mode in modes:
            continuous_rows = []
            if mode in ddpg_modes:
                continuous_rows.append((ddpg_modes[mode], f"DDPG ({mode})"))
            if mode in sac_modes:
                continuous_rows.append((sac_modes[mode], f"SAC ({mode})"))
            plot_sample_efficiency_discrete_vs_continuous(
                dqn_modes[mode],
                ppo_modes[mode],
                continuous_rows,
                out_name=f"sample_efficiency_discrete_vs_continuous_{mode}.png",
            )
    else:
        # Fall back to unified comparison when one or more files do not have per-mode splits.
        continuous_rows = []
        ddpg_all = concat_mode_groups(ddpg_modes)
        sac_all = concat_mode_groups(sac_modes)
        if ddpg_all is not None:
            continuous_rows.append((ddpg_all, "DDPG"))
        if sac_all is not None:
            continuous_rows.append((sac_all, "SAC"))
        plot_sample_efficiency_discrete_vs_continuous(
            concat_mode_groups(dqn_modes),
            concat_mode_groups(ppo_modes),
            continuous_rows,
            out_name="sample_efficiency_discrete_vs_continuous.png",
        )

    print(f"Saved plots in {RESULT_DIR}")


if __name__ == "__main__":
    main()
