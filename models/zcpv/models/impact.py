from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ImpactRow:
    match_id: str
    duration_seconds: float
    npxg: float
    own_players: tuple[str, ...]
    opponent_players: tuple[str, ...]
    own_profiles: Mapping[str, Sequence[float]]
    opponent_profiles: Mapping[str, Sequence[float]]
    context: tuple[float, ...] = ()


@dataclass
class ImpactFit:
    status: str
    reason: str | None
    feature_names: tuple[str, ...]
    player_ids: tuple[str, ...]
    intercept: float = 0.0
    uncentered_intercept: float = 0.0
    context_coef: np.ndarray | None = None
    offensive_profile_coef: np.ndarray | None = None
    defensive_profile_coef: np.ndarray | None = None
    offensive_residuals: dict[str, float] | None = None
    defensive_residuals: dict[str, float] | None = None
    profile_mean: np.ndarray | None = None
    profile_scale: np.ndarray | None = None
    offensive_center: float = 0.0
    defensive_center: float = 0.0
    offensive_exposure: dict[str, float] | None = None
    defensive_exposure: dict[str, float] | None = None
    reference_profiles: dict[str, tuple[float, ...]] | None = None
    training_matches: tuple[str, ...] = ()

    def player_effect(self, player_id: str, profile: Sequence[float] | None) -> tuple[float, float, float]:
        if self.status != "available":
            raise RuntimeError(self.reason or "impact model unavailable")
        if profile is None:
            return 0.0, 0.0, 0.0
        vector = np.asarray(profile, dtype=float)
        standardized = (vector - self.profile_mean) / self.profile_scale
        offense = float(standardized @ self.offensive_profile_coef + self.offensive_residuals.get(player_id, self.offensive_center) - self.offensive_center)
        defense = float(standardized @ self.defensive_profile_coef + self.defensive_residuals.get(player_id, self.defensive_center) - self.defensive_center)
        return offense, defense, offense + defense

    def predict_row(self, row: ImpactRow) -> float:
        if self.status != "available":
            raise RuntimeError(self.reason or "impact model unavailable")
        prediction = self.intercept
        if self.context_coef is not None and len(row.context):
            prediction += float(np.asarray(row.context) @ self.context_coef[:len(row.context)])
        prediction += sum(self.player_effect(player, row.own_profiles.get(player))[0] for player in row.own_players)
        prediction -= sum(self.player_effect(player, row.opponent_profiles.get(player))[1] for player in row.opponent_players)
        return float(prediction)


def _weighted_profile_moments(rows: Sequence[ImpactRow], width: int) -> tuple[np.ndarray, np.ndarray]:
    weighted_sum = np.zeros(width)
    weighted_square_sum = np.zeros(width)
    total_weight = 0.0
    for row in rows:
        weight = row.duration_seconds / 5400.0
        for profile in list(row.own_profiles.values()) + list(row.opponent_profiles.values()):
            vector = np.asarray(profile, dtype=float)
            if vector.size != width:
                raise ValueError("lagged profiles are inconsistent with feature_names")
            weighted_sum += weight * vector
            weighted_square_sum += weight * vector * vector
            total_weight += weight
    if total_weight <= 0:
        raise ValueError("lagged profiles are missing")
    mean = weighted_sum / total_weight
    variance = np.maximum(0.0, weighted_square_sum / total_weight - mean * mean)
    scale = np.sqrt(variance)
    scale[scale < 1e-9] = 1.0
    return mean, scale


def fit_impact_model(rows: Sequence[ImpactRow], feature_names: Sequence[str], beta_l2: float = 10.0, player_l2: float = 25.0) -> ImpactFit:
    eligible = [row for row in rows if row.duration_seconds > 0 and len(row.own_players) == 11 and len(row.opponent_players) == 11]
    if not eligible:
        return ImpactFit("insufficient_data", "no eligible 11-v-11 segments", tuple(feature_names), ())
    matches = tuple(sorted({row.match_id for row in eligible}))
    if len(matches) < 3:
        return ImpactFit("insufficient_data", "fewer than three eligible matches cannot identify held-out player effects", tuple(feature_names), (), training_matches=matches)
    player_ids = tuple(sorted({player for row in eligible for player in row.own_players + row.opponent_players}))
    player_index = {player: index for index, player in enumerate(player_ids)}
    p = len(feature_names)
    c = max((len(row.context) for row in eligible), default=0)
    try:
        mean, scale = _weighted_profile_moments(eligible, p)
    except ValueError as error:
        return ImpactFit("insufficient_data", str(error), tuple(feature_names), player_ids, training_matches=matches)

    width = 1 + c + 2 * p + 2 * len(player_ids)
    X = np.zeros((len(eligible), width), dtype=float)
    y = np.zeros(len(eligible), dtype=float)
    weights = np.zeros(len(eligible), dtype=float)
    off_beta, def_beta = 1 + c, 1 + c + p
    off_players, def_players = 1 + c + 2 * p, 1 + c + 2 * p + len(player_ids)
    exposures = np.zeros((2, len(player_ids)), dtype=float)
    profile_sums = np.zeros((len(player_ids), p), dtype=float)
    profile_exposure = np.zeros(len(player_ids), dtype=float)
    for r, row in enumerate(eligible):
        duration = row.duration_seconds / 5400.0
        X[r, 0] = 1.0
        X[r, 1:1 + len(row.context)] = row.context
        own_sum, opponent_sum = np.zeros(p), np.zeros(p)
        for player in row.own_players:
            vector = np.asarray(row.own_profiles[player], dtype=float)
            own_sum += (vector - mean) / scale
            index = player_index[player]
            X[r, off_players + index] = 1.0
            exposures[0, index] += duration
            profile_sums[index] += duration * vector
            profile_exposure[index] += duration
        for player in row.opponent_players:
            vector = np.asarray(row.opponent_profiles[player], dtype=float)
            opponent_sum += (vector - mean) / scale
            index = player_index[player]
            X[r, def_players + index] = -1.0
            exposures[1, index] += duration
            profile_sums[index] += duration * vector
            profile_exposure[index] += duration
        X[r, off_beta:off_beta + p] = own_sum
        X[r, def_beta:def_beta + p] = -opponent_sum
        y[r] = row.npxg / duration
        weights[r] = duration

    root_weight = np.sqrt(weights)
    Xw, yw = X * root_weight[:, None], y * root_weight
    penalty = np.zeros(width)
    penalty[1:off_players] = beta_l2
    penalty[off_players:] = player_l2
    coefficients = np.linalg.solve(Xw.T @ Xw + np.diag(penalty), Xw.T @ yw)
    offensive_beta = coefficients[off_beta:off_beta + p]
    defensive_beta = coefficients[def_beta:def_beta + p]
    offensive_residuals = coefficients[off_players:def_players].copy()
    defensive_residuals = coefficients[def_players:].copy()

    reference_profiles: dict[str, tuple[float, ...]] = {}
    complete_offense = np.zeros(len(player_ids))
    complete_defense = np.zeros(len(player_ids))
    for player, index in player_index.items():
        vector = profile_sums[index] / max(profile_exposure[index], 1e-12)
        reference_profiles[player] = tuple(float(value) for value in vector)
        standardized = (vector - mean) / scale
        complete_offense[index] = float(standardized @ offensive_beta + offensive_residuals[index])
        complete_defense[index] = float(standardized @ defensive_beta + defensive_residuals[index])
    offense_center = float(np.average(complete_offense, weights=exposures[0]))
    defense_center = float(np.average(complete_defense, weights=exposures[1]))
    uncentered_intercept = float(coefficients[0])
    coefficients[0] += 11 * offense_center - 11 * defense_center

    return ImpactFit(
        status="available", reason=None, feature_names=tuple(feature_names), player_ids=player_ids,
        intercept=float(coefficients[0]), uncentered_intercept=uncentered_intercept,
        context_coef=coefficients[1:1 + c],
        offensive_profile_coef=offensive_beta, defensive_profile_coef=defensive_beta,
        offensive_residuals=dict(zip(player_ids, offensive_residuals)), defensive_residuals=dict(zip(player_ids, defensive_residuals)),
        profile_mean=mean, profile_scale=scale, offensive_center=offense_center, defensive_center=defense_center,
        offensive_exposure=dict(zip(player_ids, exposures[0])), defensive_exposure=dict(zip(player_ids, exposures[1])),
        reference_profiles=reference_profiles, training_matches=matches,
    )
