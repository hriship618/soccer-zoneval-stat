from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ZoneGrid:
    """Four longitudinal bands by three lateral channels (12 zones)."""

    pitch_length: float = 105.0
    pitch_width: float = 68.0

    def index(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_bin = np.clip((np.asarray(x) / self.pitch_length * 4).astype(int), 0, 3)
        y_bin = np.clip((np.asarray(y) / self.pitch_width * 3).astype(int), 0, 2)
        return x_bin * 3 + y_bin


def fit_zone_values(
    transitions: np.ndarray,
    shots: np.ndarray,
    goals: np.ndarray,
    *,
    turnovers: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> np.ndarray:
    """Fit xT-style zone values from observed transition and terminal counts.

    ``transitions[i, j]`` counts successful moves from zone i to j. Shots and
    goals are per-origin-zone counts. Failed moves terminate with value zero.
    """
    transitions = np.asarray(transitions, dtype=float)
    shots = np.asarray(shots, dtype=float)
    goals = np.asarray(goals, dtype=float)
    if transitions.shape != (12, 12) or shots.shape != (12,) or goals.shape != (12,):
        raise ValueError("expected a 12x12 transition matrix and two 12-element vectors")
    if np.any(transitions < 0) or np.any(shots < 0) or np.any(goals < 0) or np.any(goals > shots):
        raise ValueError("counts must be non-negative and goals cannot exceed shots")

    turnovers = np.zeros(12, dtype=float) if turnovers is None else np.asarray(turnovers, dtype=float)
    if turnovers.shape != (12,) or np.any(turnovers < 0):
        raise ValueError("turnovers must contain 12 non-negative counts")
    totals = transitions.sum(axis=1) + shots + turnovers
    safe = np.where(totals == 0, 1.0, totals)
    move_probability = transitions / safe[:, None]
    immediate_goal_probability = goals / safe
    values = immediate_goal_probability.copy()
    for _ in range(max_iterations):
        updated = immediate_goal_probability + move_probability @ values
        if np.max(np.abs(updated - values)) < tolerance:
            return updated
        values = updated
    raise RuntimeError("zone value iteration did not converge")
