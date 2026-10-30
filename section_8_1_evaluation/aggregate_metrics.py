"""Aggregate results from Sections 3-7 + baselines into unified metrics CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import numpy as np

from config import (
    RESULT_DIR,
    find_section3_csv,
    find_section4_csv,
    find_section5_csvs,
    find_section6_csvs,
    find_section7_csvs,
)


OUT_COLS = [
    "section",
    "algorithm",
    "setting",
    "mode",
    "seed",
    "episode",
    "timestep",
    "reward",
    "success",
    "episode_length",
    "pass_count",
]


def load_rows(path: Path):
    if path is None or not path.exists():
        return None
    raw = path.read_bytes()
    if b"\x00" in raw:
        # Some intermediate CSVs may contain NUL padding; strip it safely.
        raw = raw.replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="ignore")
    arr = np.genfromtxt(io.StringIO(text), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def append_row(rows, section, algorithm, setting, mode, seed, episode, timestep, reward, success, episode_length, pass_count):
    rows.append(
        {
            "section": section,
            "algorithm": algorithm,
            "setting": setting,
            "mode": mode,
            "seed": int(seed),
            "episode": int(episode),
            "timestep": float(timestep),
            "reward": float(reward),
            "success": float(success),
            "episode_length": float(episode_length),
            "pass_count": float(pass_count),
        }
    )


def aggregate() -> Path:
    rows = []

    s3 = load_rows(find_section3_csv())
    if s3 is not None:
        for r in s3:
            append_row(rows, "3-1", str(r["algorithm"]), f"gamma={r['gamma']}", "single-agent", r["seed"], r["episode"], r["episode"], r["episode_reward"], r["success"], r["episode_length"], 0.0)

    s4 = load_rows(find_section4_csv())
    if s4 is not None:
        for r in s4:
            append_row(rows, "4-1", str(r["algorithm"]), f"gamma={r['gamma']}", "single-agent", r["seed"], r["episode"], r["episode"], r["episode_reward"], r["success"], r["episode_length"], 0.0)

    s5 = find_section5_csvs()
    for key, p in s5.items():
        arr = load_rows(p)
        if arr is None:
            continue
        setting, mode = key.split(":", 1)
        for r in arr:
            append_row(rows, "5-1", str(r["algorithm"]), setting, str(r["mode"]) if "mode" in arr.dtype.names else mode, r["seed"], r["episode"], r["episode"], r["team0_reward"], r["team0_scored"], r["episode_length"], 0.0)

    s6 = find_section6_csvs()
    for key, p in s6.items():
        arr = load_rows(p)
        if arr is None:
            continue
        if ":" in key:
            algo, mode = key.split(":", 1)
            setting = "continuous_action"
        else:
            algo, mode = key, "single-agent"
            setting = "discrete_action"

        for r in arr:
            append_row(
                rows,
                "6-1",
                str(r["algorithm"]),
                setting,
                str(r["mode"]) if "mode" in arr.dtype.names else mode,
                r["seed"],
                r["episode"],
                r["timestep"],
                r["episode_reward"],
                r["success"],
                r["episode_length"],
                float(r["pass_success"]) if "pass_success" in arr.dtype.names else 0.0,
            )

    s7 = find_section7_csvs()
    for key, p in s7.items():
        arr = load_rows(p)
        if arr is None:
            continue
        setting, mode = key.split(":", 1)
        for r in arr:
            append_row(rows, "7-1", str(r["algorithm"]), setting, str(r["mode"]) if "mode" in arr.dtype.names else mode, r["seed"], r["episode"], r["timestep"], r["team0_reward"], r["success"], r["episode_length"], r["pass_count"])

    b_dis = load_rows(RESULT_DIR / "baseline_discrete_results.csv")
    if b_dis is not None:
        for r in b_dis:
            append_row(rows, "6-1", str(r["algorithm"]), "baseline", str(r["mode"]) if "mode" in b_dis.dtype.names else "single-agent", r["seed"], r["episode"], r["timestep"], r["episode_reward"], r["success"], r["episode_length"], 0.0)

    b_marl = load_rows(RESULT_DIR / "baseline_marl_results.csv")
    if b_marl is not None:
        for r in b_marl:
            append_row(rows, "7-1", str(r["algorithm"]), "baseline", str(r["mode"]) if "mode" in b_marl.dtype.names else "2v2", r["seed"], r["episode"], r["timestep"], r["team0_reward"], r["success"], r["episode_length"], r["pass_count"])

    out_csv = RESULT_DIR / "unified_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=OUT_COLS,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar="\\",
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {out_csv} ({len(rows)} rows)")
    return out_csv


if __name__ == "__main__":
    aggregate()
