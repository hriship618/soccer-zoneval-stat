from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from zcpv.statsbomb import fit_xt, load_competition, split_match_ids_three_way, state_features, xt_action_value


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            error += mask.mean() * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return error


def metrics(labels: np.ndarray, probabilities: np.ndarray, baseline_probability: float) -> dict:
    prevalence = float(labels.mean())
    baseline = np.full(len(labels), baseline_probability)
    model_brier = float(brier_score_loss(labels, probabilities))
    baseline_brier = float(brier_score_loss(labels, baseline))
    return {
        "events": int(len(labels)),
        "positive_rate": round(prevalence, 5),
        "brier": round(model_brier, 5),
        "baseline_brier": round(baseline_brier, 5),
        "brier_improvement_pct": round(100 * (baseline_brier - model_brier) / baseline_brier, 2),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 5),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "calibration_error": round(expected_calibration_error(labels, probabilities), 5),
    }


def train_model(features: np.ndarray, labels: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=180, max_leaf_nodes=24,
        min_samples_leaf=30, l2_regularization=0.2, random_state=42,
    )
    return model.fit(features, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train xT and VAEP-style baselines on StatsBomb Open Data.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/statsbomb"))
    parser.add_argument("--output", type=Path, default=Path("app/model-report.generated.ts"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/models"))
    args = parser.parse_args()

    actions, matches = load_competition(args.data)
    train_ids, validation_ids, test_ids, validation_start, test_start = split_match_ids_three_way(matches)
    features, score_labels, concede_labels, ordered = state_features(actions)
    train_mask = np.asarray([action.match_id in train_ids for action in ordered])
    validation_mask = np.asarray([action.match_id in validation_ids for action in ordered])
    test_mask = np.asarray([action.match_id in test_ids for action in ordered])

    score_model = train_model(features[train_mask], score_labels[train_mask])
    concede_model = train_model(features[train_mask], concede_labels[train_mask])
    score_probability = score_model.predict_proba(features[test_mask])[:, 1]
    concede_probability = concede_model.predict_proba(features[test_mask])[:, 1]
    validation_score_probability = score_model.predict_proba(features[validation_mask])[:, 1]
    validation_concede_probability = concede_model.predict_proba(features[validation_mask])[:, 1]

    train_actions = [action for action in actions if action.match_id in train_ids]
    test_actions = [action for action in actions if action.match_id in test_ids]
    xt_grid = fit_xt(train_actions)
    xt_values = np.asarray([xt_action_value(action, xt_grid) for action in test_actions])
    report = {
        "dataset": {
            "name": "FIFA World Cup 2022",
            "provider": "StatsBomb Open Data",
            "repository": "https://github.com/hudl/open-data",
            "matches": len(matches),
            "train_matches": len(train_ids),
            "validation_matches": len(validation_ids),
            "test_matches": len(test_ids),
            "train_actions": int(train_mask.sum()),
            "validation_actions": int(validation_mask.sum()),
            "test_actions": int(test_mask.sum()),
            "validation_start": validation_start,
            "test_start": test_start,
            "split": "chronological by match date",
        },
        "vaep": {
            "state": "post-action forecast from current action plus two prior same-period actions",
            "horizon": "goal scored or conceded in the next 10 same-period actions; current action excluded; end-of-period windows censored",
            "estimator": "histogram gradient boosting",
            "validation": {
                "score": metrics(score_labels[validation_mask], validation_score_probability, float(score_labels[train_mask].mean())),
                "concede": metrics(concede_labels[validation_mask], validation_concede_probability, float(concede_labels[train_mask].mean())),
            },
            "score": metrics(score_labels[test_mask], score_probability, float(score_labels[train_mask].mean())),
            "concede": metrics(concede_labels[test_mask], concede_probability, float(concede_labels[train_mask].mean())),
        },
        "xt": {
            "grid": [int(xt_grid.shape[0]), int(xt_grid.shape[1])],
            "nonzero_zones": int(np.count_nonzero(xt_grid)),
            "max_zone_value": round(float(xt_grid.max()), 5),
            "test_actions_valued": int(np.count_nonzero(xt_values)),
            "mean_absolute_action_value": round(float(np.abs(xt_values).mean()), 5),
        },
        "notes": [
            "StatsBomb events train the event baselines; DFL tracking remains a separate evaluation source.",
            "Validation and test matches are disjoint and never fit either probability model or the xT grid.",
            "Penalty shootouts are excluded from ordinary-play modeling.",
            "These are baseline validation metrics, not evidence that PIVOT is already superior.",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(score_model, args.model_dir / "vaep_score.joblib")
    joblib.dump(concede_model, args.model_dir / "vaep_concede.joblib")
    np.save(args.model_dir / "xt_grid.npy", xt_grid)
    (args.model_dir / "model_report.json").write_text(serialized, encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"// Generated by scripts/train_baselines.py.\nconst modelReport = {serialized};\nexport default modelReport;\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
