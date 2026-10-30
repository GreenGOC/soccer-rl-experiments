"""Shared soccer mode helpers.

Supported symmetric team modes:
- 2v2, 3v3, 5v5, 7v7, 8v8, 11v11, 13v13
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


SUPPORTED_TEAM_SIZES = (2, 3, 5, 7, 8, 11, 13)
SUPPORTED_MODES = tuple(f"{n}v{n}" for n in SUPPORTED_TEAM_SIZES)


def normalize_mode(mode: str) -> str:
    m = mode.strip().lower().replace(" ", "")
    if "x" in m:
        m = m.replace("x", "v")
    if "-" in m:
        m = m.replace("-", "")
    if "v" not in m and m.isdigit():
        m = f"{m}v{m}"
    return m


def parse_mode(mode: str | None = None, team_size: int | None = None) -> Tuple[str, int]:
    if mode is None and team_size is None:
        return "2v2", 2

    if team_size is not None:
        if team_size not in SUPPORTED_TEAM_SIZES:
            raise ValueError(
                f"Unsupported team_size={team_size}. Supported sizes: {SUPPORTED_TEAM_SIZES}."
            )
        return f"{team_size}v{team_size}", team_size

    m = normalize_mode(mode or "2v2")
    if m not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode={mode}. Supported modes: {SUPPORTED_MODES}."
        )
    n = int(m.split("v")[0])
    return m, n


def grid_size_for_team(team_size: int) -> int:
    return 12


def field_scale_for_team(team_size: int) -> float:
    # Scale arena approximately by sqrt of population.
    return max(1.0, math.sqrt(team_size / 2.0))


def ensure_supported_modes(modes: Iterable[str]) -> None:
    for m in modes:
        parse_mode(m)
