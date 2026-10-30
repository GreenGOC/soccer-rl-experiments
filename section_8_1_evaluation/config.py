"""Config and path helpers for Section 8-1 evaluation."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODES = ("2v2", "3v3", "5v5", "7v7", "8v8", "11v11", "13v13")


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def find_section3_csv():
    return first_existing(
        [
            ROOT / "section_3_1_td_methods" / "results" / "section_3_1_results.csv",
            ROOT / "section_3_1_td_methods" / "section_3_1_td_methods" / "results" / "section_3_1_results.csv",
        ]
    )


def find_section4_csv():
    return ROOT / "section_4_1_actor_critic" / "results" / "section_4_1_actor_critic_results.csv"


def find_section5_csvs():
    base = ROOT / "section_5_1_marl_discrete" / "results"
    out = {}
    for mode in SUPPORTED_MODES:
        out[f"cooperative:{mode}"] = base / f"cooperative_results_{mode}.csv"
        out[f"competitive:{mode}"] = base / f"competitive_results_{mode}.csv"
    return out


def find_section6_csvs():
    base_d = ROOT / "section_6_1_1_continuous_state_discrete_action" / "results"
    base_c = ROOT / "section_6_1_2_continuous_action" / "results"

    out = {
        "dqn": base_d / "dqn_results.csv",
        "ppo_discrete": base_d / "ppo_results.csv",
    }
    for mode in SUPPORTED_MODES:
        out[f"ddpg:{mode}"] = base_c / f"ddpg_results_{mode}.csv"
        out[f"sac:{mode}"] = base_c / f"sac_results_{mode}.csv"
    return out


def find_section7_csvs():
    base = ROOT / "section_7_1_deep_marl" / "results"
    out = {}
    for mode in SUPPORTED_MODES:
        out[f"ctde:{mode}"] = base / f"ctde_results_{mode}.csv"
        out[f"independent:{mode}"] = base / f"independent_results_{mode}.csv"
    return out
