from __future__ import annotations

import numpy as np

from zcpv.pitch_control import PitchControlConfig, leave_one_out_zone_values, pitch_control_counterfactuals
from zcpv.pivot import exposure_reliability
from zcpv.statsbomb import SBAction, event_model_values


class FeatureProbability:
    def __init__(self, base: float, success_weight: float = 0.0):
        self.base = base
        self.success_weight = success_weight

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.clip(self.base + self.success_weight * features[:, 10], 0.001, 0.999)
        return np.column_stack((1 - probability, probability))


def action(index: int, team: int, *, success: bool = True, period: int = 1) -> SBAction:
    return SBAction(
        match_id="match", index=index, period=period, minute=1, second=index,
        possession=1, team_id=team, team_name=str(team), player_id=index,
        player_name=str(index), action_type="Pass", start_x=40, start_y=40,
        end_x=60, end_y=40, successful=success, goal=False, shot_xg=0,
        score_for=0, score_against=0,
    )


def test_event_values_use_only_current_and_prior_states_and_reset_periods():
    actions = [
        action(0, 1, success=False),
        action(1, 1, success=True),
        action(2, 2, success=True),
        action(3, 2, success=True, period=2),
    ]
    values = event_model_values(actions, FeatureProbability(0.1, 0.2), FeatureProbability(0.05))
    assert np.isclose(values[0]["event_value"], 0.05)
    assert np.isclose(values[1]["event_value"], 0.2)
    assert np.isclose(values[2]["event_value"], 0.5)
    assert np.isclose(values[3]["event_value"], 0.25)


def test_pitch_control_returns_full_and_matching_leave_one_out_surfaces():
    positions = np.asarray([[[20.0, 34.0], [85.0, 34.0]]], dtype=np.float32)
    velocities = np.zeros_like(positions)
    teams = np.asarray([0, 1], dtype=np.int8)
    config = PitchControlConfig(grid_x=8, grid_y=6)
    full, losses = pitch_control_counterfactuals(positions, velocities, teams, np.ones(12), config)
    legacy = leave_one_out_zone_values(positions, velocities, teams, np.ones(12), config)
    assert full.shape == (1, 2, 12)
    assert losses.shape == (1, 2, 12)
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(losses >= 0)
    assert np.allclose(losses, legacy)


def test_exposure_reliability_caps_correlated_frames_at_match_exposure():
    effective, reliability = exposure_reliability(1, 90)
    assert effective == 1.0
    assert reliability == 0.2
    assert exposure_reliability(1, 45) == (0.5, 0.5 / 4.5)
    assert exposure_reliability(5, 500) == (5.0, 5.0 / 9.0)
