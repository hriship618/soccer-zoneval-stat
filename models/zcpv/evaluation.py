from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    error = predicted_array - observed_array
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }

def aggregate_match_predictions(rows: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        match = output.setdefault(str(row["match_id"]), {"observed_for": 0.0, "predicted_for": 0.0, "observed_against": 0.0, "predicted_against": 0.0})
        for key in tuple(match):
            match[key] += float(row[key])
    for match in output.values():
        match["observed_net"] = match["observed_for"] - match["observed_against"]
        match["predicted_net"] = match["predicted_for"] - match["predicted_against"]
    return output


def match_block_bootstrap(
    match_ids: Sequence[str],
    fit_and_measure: Callable[[Sequence[str]], Mapping[str, float]],
    iterations: int = 200,
    seed: int = 618,
) -> dict[str, dict[str, float]]:
    """Refit by sampled match blocks; frames are never bootstrap units."""
    if not match_ids:
        return {}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(iterations):
        sample = rng.choice(match_ids, size=len(match_ids), replace=True).tolist()
        for key, value in fit_and_measure(sample).items():
            draws.setdefault(key, []).append(float(value))
    return {
        key: {"median": float(np.median(values)), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}
        for key, values in draws.items()
    }
