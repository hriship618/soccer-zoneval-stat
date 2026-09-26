from __future__ import annotations

from dataclasses import dataclass
from math import exp
import numpy as np


@dataclass(frozen=True)
class Action:
    player_id: str
    kind: str
    origin_zone: int
    destination_zone: int | None = None
    successful: bool = True
    distance_m: float | None = None
    angle_rad: float | None = None
    assist_type: str = "none"
    loss_risk_before: float | None = None
    loss_risk_after: float | None = None


def shot_xg(distance_m: float, angle_rad: float, assist_type: str = "none") -> float:
    """Small, deliberately transparent logistic xG model."""
    assist_bonus = {"none": 0.0, "cross": 0.18, "through_ball": 0.42, "cutback": 0.55}.get(assist_type, 0.0)
    logit = 1.15 - 0.115 * distance_m + 0.92 * angle_rad + assist_bonus
    return 1.0 / (1.0 + exp(-logit))


def value_action(action: Action, zone_values: np.ndarray) -> float:
    values = np.asarray(zone_values, dtype=float)
    if values.shape != (12,):
        raise ValueError("zone_values must contain 12 values")
    if not 0 <= action.origin_zone < 12:
        raise ValueError("origin_zone is outside the 12-zone grid")
    kind = action.kind.lower()
    if kind == "shot":
        if action.distance_m is None or action.angle_rad is None:
            raise ValueError("shots require distance_m and angle_rad")
        return shot_xg(action.distance_m, action.angle_rad, action.assist_type)
    if kind in {"dribble", "take_on"}:
        if action.loss_risk_before is None or action.loss_risk_after is None:
            raise ValueError("dribbles require before/after possession-loss risks")
        return float(values[action.origin_zone] * (action.loss_risk_before - action.loss_risk_after))
    if kind in {"tackle", "interception", "regain"}:
        return float(values[action.origin_zone]) if action.successful else 0.0
    if kind in {"pass", "carry"}:
        if action.destination_zone is None or not 0 <= action.destination_zone < 12:
            raise ValueError("passes and carries require a valid destination_zone")
        return float(values[action.destination_zone] - values[action.origin_zone]) if action.successful else -float(values[action.origin_zone])
    raise ValueError(f"unsupported action kind: {action.kind}")
