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
    context_coef: np.ndarray | None = None
    offensive_profile_coef: np.ndarray | None = None
    defensive_profile_coef: np.ndarray | None = None
    offensive_residuals: dict[str, float] | None = None
    defensive_residuals: dict[str, float] | None = None
    profile_mean: np.ndarray | None = None
    profile_scale: np.ndarray | None = None
    training_matches: tuple[str, ...] = ()

    def player_effect(self, player_id: str, profile: Sequence[float] | None) -> tuple[float, float, float]:
        if self.status != "available":
            raise RuntimeError(self.reason or "impact model unavailable")
        # An unseen player receives the training reference profile and a zero
        # player residual, rather than a raw all-zero profile in original units.
        vector = self.profile_mean if profile is None else np.asarray(profile, dtype=float)
        standardized = (vector - self.profile_mean) / self.profile_scale
        offense = float(standardized @ self.offensive_profile_coef + self.offensive_residuals.get(player_id, 0.0))
        defense = float(standardized @ self.defensive_profile_coef + self.defensive_residuals.get(player_id, 0.0))
        return offense, defense, offense + defense


def fit_impact_model(
    rows: Sequence[ImpactRow],
    feature_names: Sequence[str],
    beta_l2: float = 10.0,
    player_l2: float = 25.0,
) -> ImpactFit:
    if not rows:
        return ImpactFit("insufficient_data", "no eligible 11-v-11 segments", tuple(feature_names), ())
    rows = [row for row in rows if row.duration_seconds > 0 and len(row.own_players) == 11 and len(row.opponent_players) == 11]
    matches = tuple(sorted({row.match_id for row in rows}))
    if len(matches) < 3:
        return ImpactFit("insufficient_data", "fewer than three eligible matches cannot identify held-out player effects", tuple(feature_names), (), training_matches=matches)
    player_ids = tuple(sorted({player for row in rows for player in row.own_players + row.opponent_players}))
    player_index = {player: index for index, player in enumerate(player_ids)}
    p = len(feature_names)
    c = max((len(row.context) for row in rows), default=0)
    all_profiles = [np.asarray(profile, dtype=float) for row in rows for profile in list(row.own_profiles.values()) + list(row.opponent_profiles.values())]
    if not all_profiles or any(profile.size != p for profile in all_profiles):
        return ImpactFit("insufficient_data", "lagged profiles are missing or inconsistent", tuple(feature_names), player_ids, training_matches=matches)
    profile_matrix = np.vstack(all_profiles)
    mean = profile_matrix.mean(axis=0)
    scale = profile_matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    # intercept, context, offensive beta, defensive beta, offensive residuals, defensive residuals
    width = 1 + c + 2 * p + 2 * len(player_ids)
    X = np.zeros((len(rows), width), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    weights = np.zeros(len(rows), dtype=float)
    off_beta = 1 + c
    def_beta = off_beta + p
    off_players = def_beta + p
    def_players = off_players + len(player_ids)
    exposures = np.zeros((2, len(player_ids)), dtype=float)
    for r, row in enumerate(rows):
        duration = row.duration_seconds / 5400.0
        X[r, 0] = 1.0
        X[r, 1:1 + len(row.context)] = row.context
        own_sum = sum(((np.asarray(row.own_profiles[player]) - mean) / scale for player in row.own_players), start=np.zeros(p))
        opponent_sum = sum(((np.asarray(row.opponent_profiles[player]) - mean) / scale for player in row.opponent_players), start=np.zeros(p))
        X[r, off_beta:off_beta + p] = own_sum
        X[r, def_beta:def_beta + p] = -opponent_sum
        for player in row.own_players:
            index = player_index[player]
            X[r, off_players + index] = 1.0
            exposures[0, index] += duration
        for player in row.opponent_players:
            index = player_index[player]
            X[r, def_players + index] = -1.0
            exposures[1, index] += duration
        y[r] = row.npxg / duration
        weights[r] = duration
    root_weight = np.sqrt(weights)
    Xw, yw = X * root_weight[:, None], y * root_weight
    penalty = np.zeros(width)
    penalty[1:off_players] = beta_l2
    penalty[off_players:] = player_l2
    coefficients = np.linalg.solve(Xw.T @ Xw + np.diag(penalty), Xw.T @ yw)
    offensive = coefficients[off_players:def_players].copy()
    defensive = coefficients[def_players:].copy()
    # Explicit exposure-weighted centering identifies the competition-average player.
    centers = []
    for values, exposure in ((offensive, exposures[0]), (defensive, exposures[1])):
        if exposure.sum() > 0:
            center = float(np.average(values, weights=exposure))
            values -= center
            centers.append(center)
        else:
            centers.append(0.0)
    coefficients[0] += 11 * centers[0] - 11 * centers[1]
    return ImpactFit(
        status="available", reason=None, feature_names=tuple(feature_names), player_ids=player_ids,
        intercept=float(coefficients[0]), context_coef=coefficients[1:1 + c],
        offensive_profile_coef=coefficients[off_beta:off_beta + p], defensive_profile_coef=coefficients[def_beta:def_beta + p],
        offensive_residuals=dict(zip(player_ids, offensive)), defensive_residuals=dict(zip(player_ids, defensive)),
        profile_mean=mean, profile_scale=scale, training_matches=matches,
    )
