"""Hierarchical skill adapter for Section 7-1 policies.

High-level option selection is rule-based and maps to low-level motion primitives.
Policy outputs are treated as residual commands on top of the selected skill.
"""

from __future__ import annotations

import numpy as np


SKILL_PRESS_BALL = 0
SKILL_ATTACK_GOAL = 1
SKILL_SUPPORT_PASS = 2
SKILL_HOLD_SHAPE = 3


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (v / n).astype(np.float32)


def select_skill(local_obs: np.ndarray) -> int:
    ball_rel = local_obs[4:6]
    goal_rel = local_obs[6:8]
    mate_rel = local_obs[8:10]
    has_ball = bool(local_obs[12] > 0.5)
    team_has_ball = bool(local_obs[13] > 0.5)

    if has_ball and abs(float(goal_rel[0])) < 0.45:
        return SKILL_ATTACK_GOAL
    if has_ball and abs(float(mate_rel[0])) < 0.9:
        return SKILL_SUPPORT_PASS
    if not team_has_ball and float(np.linalg.norm(ball_rel)) > 0.05:
        return SKILL_PRESS_BALL
    return SKILL_HOLD_SHAPE


def skill_base_velocity(skill: int, local_obs: np.ndarray) -> np.ndarray:
    ball_rel = local_obs[4:6]
    goal_rel = local_obs[6:8]
    mate_rel = local_obs[8:10]
    if skill == SKILL_PRESS_BALL:
        return _unit(ball_rel)
    if skill == SKILL_ATTACK_GOAL:
        return _unit(goal_rel)
    if skill == SKILL_SUPPORT_PASS:
        return _unit(mate_rel)
    return np.zeros(2, dtype=np.float32)


def apply_hierarchical_controller(local_obs: np.ndarray, residual_action: np.ndarray) -> np.ndarray:
    residual = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
    skill = select_skill(local_obs)
    base = skill_base_velocity(skill, local_obs)

    # Motion primitive + residual fine control.
    move = np.clip(0.75 * base + 0.35 * residual[:2], -1.0, 1.0)

    has_ball = bool(local_obs[12] > 0.5)
    kick = float(residual[2])
    if has_ball and skill in (SKILL_ATTACK_GOAL, SKILL_SUPPORT_PASS):
        kick = max(kick, 0.6)
    kick = float(np.clip(kick, -1.0, 1.0))

    return np.array([move[0], move[1], kick], dtype=np.float32)

