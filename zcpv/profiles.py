from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, log
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlayerMatchMeasurement:
    player_id: str
    match_id: str
    match_date: str
    role: str
    minutes: float
    features: Mapping[str, float | None]
    exposures: Mapping[str, float]


@dataclass(frozen=True)
class LaggedProfile:
    player_id: str
    cutoff: str
    role: str
    values: Mapping[str, float]
    sample_counts: Mapping[str, float]
    minutes: float
    low_information: bool


def build_lagged_profiles(
    measurements: Iterable[PlayerMatchMeasurement],
    cutoff: str,
    feature_names: Sequence[str],
    shrinkage_kappa: float = 10.0,
    recency_half_life_matches: float = 8.0,
) -> dict[str, LaggedProfile]:
    """Build profiles strictly from matches earlier than ``cutoff``."""
    cutoff_date = date.fromisoformat(cutoff[:10])
    eligible = [row for row in measurements if date.fromisoformat(row.match_date[:10]) < cutoff_date]
    eligible.sort(key=lambda row: (row.match_date, row.match_id))
    roles: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in eligible:
        role = roles.setdefault(row.role, {name: [] for name in feature_names})
        for name in feature_names:
            value = row.features.get(name)
            if value is not None:
                role[name].append((float(value), max(0.0, float(row.exposures.get(name, 0.0)))))
    role_means = {
        role: {name: (sum(v * max(w, 1e-9) for v, w in values) / sum(max(w, 1e-9) for _, w in values) if values else 0.0) for name, values in features.items()}
        for role, features in roles.items()
    }
    global_mean = {
        name: float(np.mean([float(row.features[name]) for row in eligible if row.features.get(name) is not None])) if any(row.features.get(name) is not None for row in eligible) else 0.0
        for name in feature_names
    }
    by_player: dict[str, list[PlayerMatchMeasurement]] = {}
    for row in eligible:
        by_player.setdefault(row.player_id, []).append(row)
    result = {}
    for player_id, rows in by_player.items():
        role = rows[-1].role
        values, counts = {}, {}
        total_minutes = sum(row.minutes for row in rows)
        for name in feature_names:
            numerator = denominator = 0.0
            for age, row in enumerate(reversed(rows)):
                value = row.features.get(name)
                if value is None:
                    continue
                recency = exp(-log(2) * age / max(recency_half_life_matches, 1e-9))
                exposure = max(0.0, float(row.exposures.get(name, 0.0)))
                weight = recency * exposure
                numerator += float(value) * weight
                denominator += weight
            observed = numerator / denominator if denominator else role_means.get(role, {}).get(name, global_mean[name])
            reference = role_means.get(role, {}).get(name, global_mean[name])
            values[name] = denominator / (denominator + shrinkage_kappa) * observed + shrinkage_kappa / (denominator + shrinkage_kappa) * reference
            counts[name] = denominator
        result[player_id] = LaggedProfile(
            player_id=player_id, cutoff=cutoff, role=role, values=values,
            sample_counts=counts, minutes=total_minutes,
            low_information=max(counts.values(), default=0.0) < shrinkage_kappa,
        )
    return result


def reference_profile(player_id: str, cutoff: str, role: str, feature_names: Sequence[str], role_means: Mapping[str, Mapping[str, float]]) -> LaggedProfile:
    return LaggedProfile(
        player_id=player_id, cutoff=cutoff, role=role,
        values={name: float(role_means.get(role, {}).get(name, 0.0)) for name in feature_names},
        sample_counts={name: 0.0 for name in feature_names}, minutes=0.0, low_information=True,
    )
