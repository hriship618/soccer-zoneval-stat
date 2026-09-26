from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .train_baselines import metrics, train_model
from zcpv.pivot import aggregate_rankings, leave_one_match_out, process_dfl_match
from zcpv.statsbomb import (
    collapse_xt_grid,
    fit_xt,
    load_competition,
    shot_features,
    split_match_ids_three_way,
    state_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "models" / "data"
FRONTEND_ROOT = REPO_ROOT / "frontend"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compute real-data PIVOT rankings end to end.")
    parser.add_argument("--statsbomb", type=Path, default=DATA_ROOT / "raw" / "statsbomb")
    parser.add_argument("--dfl", type=Path, default=DATA_ROOT / "raw" / "dfl")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "processed" / "pivot")
    parser.add_argument("--dashboard", type=Path, default=FRONTEND_ROOT / "app" / "pivot-rankings.generated.ts")
    parser.add_argument("--tracking-hz", type=int, default=5)
    args = parser.parse_args()

    print("[1/4] Training World Cup event-value and shot-xG models", flush=True)
    actions, matches = load_competition(args.statsbomb)
    train_ids, validation_ids, test_ids, validation_start, test_start = split_match_ids_three_way(matches)
    features, score_labels, concede_labels, ordered = state_features(actions)
    train_mask = np.asarray([action.match_id in train_ids for action in ordered])
    validation_mask = np.asarray([action.match_id in validation_ids for action in ordered])
    test_mask = np.asarray([action.match_id in test_ids for action in ordered])
    score_model = train_model(features[train_mask], score_labels[train_mask])
    concede_model = train_model(features[train_mask], concede_labels[train_mask])
    train_shots = [action for action in actions if action.match_id in train_ids and action.action_type == "Shot"]
    validation_shots = [action for action in actions if action.match_id in validation_ids and action.action_type == "Shot"]
    test_shots = [action for action in actions if action.match_id in test_ids and action.action_type == "Shot"]
    shot_xg_model = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=160, max_leaf_nodes=16,
        min_samples_leaf=20, l2_regularization=0.2, random_state=42,
    ).fit(shot_features(train_shots), np.asarray([action.shot_xg for action in train_shots]))
    xt_grid = fit_xt([action for action in actions if action.match_id in train_ids])
    zone_values = collapse_xt_grid(xt_grid)

    event_report = {
        "dataset": {
            "name": "FIFA World Cup 2022",
            "provider": "StatsBomb Open Data",
            "matches": len(matches),
            "train_matches": len(train_ids),
            "validation_matches": len(validation_ids),
            "test_matches": len(test_ids),
            "validation_start": validation_start,
            "test_start": test_start,
            "split": "chronological, match-disjoint 38/13/13",
        },
        "target": "goal scored and goal conceded in the next 10 actions, excluding the current action and censoring period ends",
        "features": "current and two prior same-period actions: type, start/end coordinates, displacement, success, team relation, shot geometry, period, clock and score difference",
        "score": {
            "validation": metrics(
                score_labels[validation_mask], score_model.predict_proba(features[validation_mask])[:, 1],
                float(score_labels[train_mask].mean()),
            ),
            "test": metrics(
                score_labels[test_mask], score_model.predict_proba(features[test_mask])[:, 1],
                float(score_labels[train_mask].mean()),
            ),
        },
        "concede": {
            "validation": metrics(
                concede_labels[validation_mask], concede_model.predict_proba(features[validation_mask])[:, 1],
                float(concede_labels[train_mask].mean()),
            ),
            "test": metrics(
                concede_labels[test_mask], concede_model.predict_proba(features[test_mask])[:, 1],
                float(concede_labels[train_mask].mean()),
            ),
        },
        "shot_xg_transfer": {
            "features": "normalized shot endpoint, distance and angle",
            "train_shots": len(train_shots),
            "validation_shots": len(validation_shots),
            "test_shots": len(test_shots),
            "validation_mae_to_statsbomb_xg": float(mean_absolute_error(
                [action.shot_xg for action in validation_shots], shot_xg_model.predict(shot_features(validation_shots)),
            )),
            "test_mae_to_statsbomb_xg": float(mean_absolute_error(
                [action.shot_xg for action in test_shots], shot_xg_model.predict(shot_features(test_shots)),
            )),
        },
    }

    model_dir = args.output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(score_model, model_dir / "event_score.joblib")
    joblib.dump(concede_model, model_dir / "event_concede.joblib")
    joblib.dump(shot_xg_model, model_dir / "shot_xg.joblib")
    np.save(model_dir / "xt_grid.npy", xt_grid)

    print("[2/4] Processing all seven DFL tracking matches", flush=True)
    match_dirs = [
        path for path in sorted(args.dfl.iterdir())
        if path.is_dir() and all((path / name).exists() for name in ("match.xml", "events.xml", "positions.xml"))
    ]
    if len(match_dirs) != 7:
        raise SystemExit(f"Expected seven complete DFL matches, found {len(match_dirs)}")
    dfl_results = []
    for index, match_dir in enumerate(match_dirs, 1):
        print(f"  [{index}/7] {match_dir.name}", flush=True)
        dfl_results.append(process_dfl_match(
            match_dir, score_model, concede_model, shot_xg_model,
            xt_grid, zone_values, tracking_hz=args.tracking_hz,
        ))

    print("[3/4] Leave-one-match-out DFL fusion and evaluation", flush=True)
    contributions, dfl_evaluation, crossfit = leave_one_match_out(dfl_results)
    rankings, baseline_evaluation = aggregate_rankings(dfl_results, contributions)
    outfield = rankings["outfield"]
    goalkeepers = rankings["goalkeepers"]

    methodology = {
        "name": "PIVOT — Player Impact via Outcomes and Tracking",
        "version": "pivot-real-v1.1",
        "score_definition": {
            "event_value": "EV_e = [P_WC(score|post_e)-P_WC(concede|post_e)] - [P_WC(score|pre_e)-P_WC(concede|pre_e)], oriented to the actor's team",
            "fusion": "Two DFL logistic calibrators estimate score and concede from logit(World Cup probability) and actor-oriented threat-weighted pitch-control advantage.",
            "spatial_counterfactual": "SCF_i,e is the player's own-team-oriented change in fused net outcome probability when that player is removed from the synchronized tracking frame.",
            "player_event": "C_i,e = 1[i is actor] * EV_e + SCF_i,e",
            "aggregation": "C_i = sum_e C_i,e; raw_rate_i = 100*C_i/N_i. Define effective_matches_i = min(matches_i, minutes_i/90) and reliability_i = effective_matches_i/(effective_matches_i+4). Shrink raw_rate_i toward the exposure-weighted goalkeeper/outfield mean using that reliability.",
            "rating": "PIVOT_i = clip(50 + 10*(shrunk_rate_i - group_prior)/raw_group_standard_deviation, 0, 100) among players with at least 45 tracked minutes. The scale is fixed from unshrunk rates so standardization cannot undo shrinkage. Goalkeepers and outfield players are ranked separately.",
        },
        "tracking_features": [
            "reaction- and velocity-adjusted time-to-intercept pitch control on a 32x21 grid",
            "actor-oriented World Cup xT-weighted zonal control",
            "team pitch-control advantage",
            "every active player's exact algebraic leave-one-player-out control loss",
        ],
        "data_boundary": "World Cup event files are never assigned tracking features. Tracking is read only from the seven DFL matches.",
        "dfl_validation": "Seven leave-one-match-out folds. Every displayed match contribution is generated by fusion models fitted on the other six matches.",
        "normalization": "The raw per-100-event rate is retained, but uncertainty depends on match/minute exposure: effective_matches=min(matches, minutes/90), reliability=effective_matches/(effective_matches+4). A full single match has reliability 0.20 regardless of frame count; five full matches have reliability 0.556. Ratings use the unshrunk group standard deviation, preventing post-shrink restandardization from restoring discarded variance.",
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "real_data_only": True,
        "methodology": methodology,
        "world_cup_event_model": event_report,
        "dfl": {
            "matches": len(dfl_results),
            "quality": {match.match_id: match.quality for match in dfl_results},
            "evaluation": dfl_evaluation,
            "baseline_comparison": baseline_evaluation,
            "crossfit": crossfit,
        },
        "ranking": {
            "players": len(outfield) + len(goalkeepers),
            "outfield_players": len(outfield),
            "goalkeepers": len(goalkeepers),
            "qualification": "at least 45 tracked minutes and one observed match; raw sample count remains reported but does not determine reliability",
            "top_10_outfield": outfield[:10],
            "top_goalkeepers": goalkeepers[:5],
        },
        "limitations": [
            "The DFL sample is seven matches, so rankings are match-sample estimates rather than season-long talent estimates.",
            "World Cup-to-Bundesliga event transfer can be affected by competition and provider-domain shift.",
            "Pitch-control physics and the 32x21 grid are model assumptions, not direct causal identification.",
            "The 0-100-style rating is standardized within this seven-match player pool and is not comparable to another season without refitting the reference distribution.",
            "Leave-one-match-out prevents same-match fitting, but matches from the same release are not independent like multiple seasons would be.",
        ],
    }

    print("[4/4] Writing rankings and dashboard artifact", flush=True)
    write_json(args.output / "event-model-report.json", event_report)
    write_json(args.output / "dfl-evaluation.json", {"event": dfl_evaluation, "baselines": baseline_evaluation})
    write_json(args.output / "player-event-contributions.json", contributions)
    write_json(args.output / "player-rankings.json", rankings)
    write_json(args.output / "pivot-report.json", report)
    dashboard = {
        "generated_at": report["generated_at"],
        "model_version": methodology["version"],
        "real_data_only": True,
        "matches": len(dfl_results),
        "outfield_players": outfield,
        "goalkeepers": goalkeepers,
    }
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(
        "// Generated by python -m models.scripts.run_pivot. Do not edit.\n"
        f"const pivotRankings = {json.dumps(dashboard, ensure_ascii=False, indent=2)} as const;\n"
        "export default pivotRankings;\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ranked_players": len(outfield) + len(goalkeepers),
        "top_10_outfield": [{"rank": row["rank"], "name": row["name"], "team": row["team"], "rating": row["pivot_rating"]} for row in outfield[:10]],
        "top_goalkeepers": [{"rank": row["rank"], "name": row["name"], "team": row["team"], "rating": row["pivot_rating"]} for row in goalkeepers[:5]],
        "artifacts": str(args.output),
        "dashboard": str(args.dashboard),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
