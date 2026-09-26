from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


@dataclass
class StateModelFit:
    status: str
    reason: str | None
    feature_names: tuple[str, ...]
    seed: int
    source: str
    cutoff: str | None
    occurrence_model: object | None = None
    conditional_xg_model: object | None = None
    direct_model: object | None = None
    metrics: dict | None = None

    def metadata(self) -> dict:
        payload = asdict(self)
        for key in ("occurrence_model", "conditional_xg_model", "direct_model"):
            payload.pop(key, None)
        return payload

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.status != "available" or self.occurrence_model is None or self.conditional_xg_model is None:
            raise RuntimeError(self.reason or "state model is unavailable")
        occurrence = self.occurrence_model.predict_proba(X)[:, 1]
        conditional = np.maximum(0.0, self.conditional_xg_model.predict(X))
        return occurrence * conditional


def _reliability(y: np.ndarray, prediction: np.ndarray, bins: int = 8) -> list[dict]:
    boundaries = np.quantile(prediction, np.linspace(0, 1, bins + 1))
    output = []
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (prediction >= boundaries[index]) & ((prediction <= boundaries[index + 1]) if right_closed else (prediction < boundaries[index + 1]))
        if mask.any():
            output.append({"n": int(mask.sum()), "predicted": float(prediction[mask].mean()), "observed": float(y[mask].mean())})
    return output


def fit_state_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    source: str,
    cutoff: str | None = None,
    min_positive_windows: int = 25,
) -> StateModelFit:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_validation = np.asarray(X_validation, dtype=float)
    y_validation = np.asarray(y_validation, dtype=float)
    positives = y_train > 0
    if X_train.ndim != 2 or X_train.shape[0] != y_train.size:
        raise ValueError("X_train and y_train shapes do not align")
    if positives.sum() < min_positive_windows:
        return StateModelFit(
            status="insufficient_data", reason=f"only {int(positives.sum())} positive windows; need {min_positive_windows}",
            feature_names=tuple(feature_names), seed=seed, source=source, cutoff=cutoff,
        )
    classifier = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=180, l2_regularization=1.0, random_state=seed)
    conditional = HistGradientBoostingRegressor(loss="squared_error", max_depth=4, learning_rate=0.06, max_iter=140, l2_regularization=1.0, random_state=seed)
    direct = HistGradientBoostingRegressor(loss="squared_error", max_depth=5, learning_rate=0.06, max_iter=180, l2_regularization=1.0, random_state=seed)
    classifier.fit(X_train, positives.astype(int))
    conditional.fit(X_train[positives], y_train[positives])
    direct.fit(X_train, y_train)
    probability = classifier.predict_proba(X_validation)[:, 1]
    prediction = probability * np.maximum(0.0, conditional.predict(X_validation))
    direct_prediction = direct.predict(X_validation)
    validation_occurrence = (y_validation > 0).astype(int)
    metrics = {
        "occurrence": {
            "brier": float(brier_score_loss(validation_occurrence, probability)),
            "log_loss": float(log_loss(validation_occurrence, np.clip(probability, 1e-8, 1 - 1e-8), labels=[0, 1])),
            "reliability": _reliability(validation_occurrence, probability),
        },
        "expected_xg": {
            "mae": float(mean_absolute_error(y_validation, prediction)),
            "rmse": float(mean_squared_error(y_validation, prediction) ** 0.5),
            "prediction_bins": _reliability(y_validation, prediction),
        },
        "direct_regression": {
            "mae": float(mean_absolute_error(y_validation, direct_prediction)),
            "rmse": float(mean_squared_error(y_validation, direct_prediction) ** 0.5),
        },
        "train_windows": int(y_train.size), "validation_windows": int(y_validation.size),
        "train_positive_windows": int(positives.sum()),
    }
    return StateModelFit(
        status="available", reason=None, feature_names=tuple(feature_names), seed=seed, source=source,
        cutoff=cutoff, occurrence_model=classifier, conditional_xg_model=conditional, direct_model=direct, metrics=metrics,
    )
