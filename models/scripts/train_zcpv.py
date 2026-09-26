from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np

from zcpv.adapters.dfl import DFLAdapter
from zcpv.config import ZCPVConfig
from zcpv.evaluation import regression_metrics
from zcpv.export import research_export
from zcpv.models.impact import ImpactRow, fit_impact_model
from zcpv.models.state_value import fit_state_model
from zcpv.pipeline import EVENT_FEATURES, extract_dfl_match
from zcpv.profiles import PlayerMatchMeasurement, build_lagged_profiles
from zcpv.quality import build_quality_report, file_checksum, write_quality_report
from zcpv.schema import write_jsonl, write_schema


STAGES = ("audit", "features", "state", "profiles", "impact", "evaluate", "export", "smoke", "all")
PROFILE_FEATURES = (
    "progression", "receiving_availability", "option_quality", "pressure", "pressure_episodes",
    "lane_coverage", "danger_lane_coverage", "transition_protection",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete_matches(root: Path, match_id: str | None = None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DFL input root does not exist: {root}")
    matches = [path for path in sorted(root.iterdir()) if path.is_dir() and all((path / name).exists() for name in ("match.xml", "events.xml", "positions.xml"))]
    if match_id is not None:
        matches = [path for path in matches if path.name == match_id]
    if not matches:
        detail = f" for {match_id}" if match_id else ""
        raise FileNotFoundError(f"No complete DFL matches{detail} in {root}; expected match.xml, events.xml and positions.xml")
    return matches


def audit_stage(data_root: Path, output_root: Path, include_tracking: bool = False, match_id: str | None = None) -> dict:
    summaries = []
    write_schema(output_root / "schema.json")
    for match_dir in complete_matches(data_root, match_id):
        adapter = DFLAdapter(match_dir)
        match = adapter.match(); events, audit = adapter.events(); lineups = adapter.lineups()
        canonical = output_root / "canonical" / match.match_id
        write_jsonl(canonical / "matches.jsonl", [match]); write_jsonl(canonical / "events.jsonl", events); write_jsonl(canonical / "lineups.jsonl", lineups)
        tracking_rows = list(adapter.tracking(target_hz=1.0)) if include_tracking else None
        if tracking_rows is not None:
            write_jsonl(canonical / "tracking.jsonl", tracking_rows)
            audit.tracking = {"sample_hz": 1.0, "rows": len(tracking_rows)}
        quality = build_quality_report(match, events, lineups, tracking_rows)
        quality["ingestion_audit"] = audit.to_dict()
        quality["checksums"] = {name: file_checksum(match_dir / name) for name in ("match.xml", "events.xml", "positions.xml")}
        report_name = f"{match.match_id}-tracking.json" if include_tracking else f"{match.match_id}.json"
        write_quality_report(output_root / "quality" / report_name, quality)
        summaries.append({
            "match_id": match.match_id, "status": quality["status"], "raw_events": audit.raw_events,
            "parsed_events": audit.parsed_events, "legacy_valued_events": audit.valued_legacy_events,
            "shots": audit.parsed_type_counts.get("shot", 0), "passes": audit.parsed_type_counts.get("pass", 0),
            "carries": audit.parsed_type_counts.get("carry", 0), "tracking_audited": include_tracking,
        })
    payload = {"stage": "audit", "status": "complete", "matches": summaries}
    write_json(output_root / "audit-summary.json", payload)
    return payload


def features_stage(data_root: Path, output_root: Path, config: ZCPVConfig, match_id: str | None = None) -> dict:
    summaries = []
    for match_dir in complete_matches(data_root, match_id):
        result = extract_dfl_match(DFLAdapter(match_dir), config)
        destination = output_root / "features" / result.match_id
        write_jsonl(destination / "state_samples.jsonl", result.state_samples)
        write_jsonl(destination / "player_match_measurements.jsonl", result.measurements)
        write_jsonl(destination / "segments.jsonl", result.segments)
        write_json(destination / "summary.json", result.summary)
        summaries.append({"match_id": result.match_id, **result.summary})
    payload = {
        "stage": "features", "status": "complete", "source": "real_dfl_tracking",
        "sample_hz": config.sample_hz, "cache_key": config.hash, "matches": summaries,
        "note": "Features do not require xG labels. Target eligibility is recorded separately.",
    }
    write_json(output_root / "features-summary.json", payload)
    return payload


def _sample_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, np.nan) if row["features"].get(name) is not None else np.nan) for name in names] for row in rows], dtype=float)


def _match_order(output_root: Path, match_ids: set[str]) -> list[str]:
    dated = []
    for match_id in match_ids:
        rows = read_jsonl(output_root / "canonical" / match_id / "matches.jsonl")
        dated.append(((rows[0].get("kickoff") if rows else "") or "", match_id))
    return [match_id for _, match_id in sorted(dated)]


def state_stage(output_root: Path, config: ZCPVConfig) -> dict:
    rows = [row for path in output_root.glob("features/*/state_samples.jsonl") for row in read_jsonl(path)]
    if not rows:
        payload = {"stage": "state", "status": "missing_inputs", "reason": "run the features stage first"}
        write_json(output_root / "state-summary.json", payload); return payload
    missing_xg = sum(row.get("exclusion_reason") == "missing_xg_target" for row in rows)
    eligible = [row for row in rows if row.get("eligible") and row.get("reference_npxg_15s") is not None and row.get("opponent_npxg_15s") is not None]
    positives = sum((row.get("reference_npxg_15s") or 0) > 0 or (row.get("opponent_npxg_15s") or 0) > 0 for row in eligible)
    matches = _match_order(output_root, {row["match_id"] for row in eligible})
    deficiencies = []
    if missing_xg:
        deficiencies.append(f"{missing_xg} observation windows contain shots without provider/frozen xG")
    if positives < config.min_positive_windows:
        deficiencies.append(f"only {positives} positive labeled windows; need {config.min_positive_windows}")
    if len(matches) < 5:
        deficiencies.append(f"only {len(matches)} matches have eligible labels; need at least 5 for train/validation/test")
    if deficiencies:
        payload = {"stage": "state", "status": "insufficient_data", "reason": "; ".join(deficiencies), "rows": len(rows), "eligible_rows": len(eligible), "positive_rows": positives, "missing_xg_windows": missing_xg}
        write_json(output_root / "state-summary.json", payload); return payload
    train_end = max(1, int(len(matches) * 0.6)); validation_end = max(train_end + 1, int(len(matches) * 0.8))
    train_ids, validation_ids, test_ids = set(matches[:train_end]), set(matches[train_end:validation_end]), set(matches[validation_end:])
    if not validation_ids or not test_ids:
        payload = {"stage": "state", "status": "insufficient_data", "reason": "chronological validation/test partition is empty"}
        write_json(output_root / "state-summary.json", payload); return payload
    feature_sets = {
        "event_only": [name for name in EVENT_FEATURES if any(name in row["features"] for row in eligible)],
        "tracking_augmented": sorted({name for row in eligible for name, value in row["features"].items() if isinstance(value, (int, float))}),
    }
    results = {}; model_dir = output_root / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    for variant, names in feature_sets.items():
        train = [row for row in eligible if row["match_id"] in train_ids]
        validation = [row for row in eligible if row["match_id"] in validation_ids]
        for side, target in (("for", "reference_npxg_15s"), ("against", "opponent_npxg_15s")):
            fit = fit_state_model(_sample_matrix(train, names), np.asarray([row[target] for row in train]), _sample_matrix(validation, names), np.asarray([row[target] for row in validation]), names, config.random_seed, "canonical synchronized inputs", cutoff=matches[train_end], min_positive_windows=config.min_positive_windows)
            results[f"{variant}_{side}"] = fit.metadata()
            if fit.status == "available":
                joblib.dump(fit, model_dir / f"state-{variant}-{side}.joblib")
    status = "available" if all(result["status"] == "available" for result in results.values()) else "insufficient_data"
    payload = {"stage": "state", "status": status, "splits": {"train": sorted(train_ids), "validation": sorted(validation_ids), "test": sorted(test_ids)}, "identical_eligible_matches": True, "models": results}
    write_json(output_root / "state-summary.json", payload)
    return payload


def profiles_stage(output_root: Path, config: ZCPVConfig) -> dict:
    raw = [row for path in output_root.glob("features/*/player_match_measurements.jsonl") for row in read_jsonl(path)]
    if not raw:
        payload = {"stage": "profiles", "status": "missing_inputs", "reason": "run the features stage first"}
        write_json(output_root / "profiles-summary.json", payload); return payload
    measurements = [PlayerMatchMeasurement(**row) for row in raw]
    match_dates = sorted({(row.match_date, row.match_id) for row in measurements})
    output = []
    for cutoff, match_id in match_dates:
        profiles = build_lagged_profiles(measurements, cutoff, PROFILE_FEATURES, config.shrinkage_kappa, config.recency_half_life_matches)
        for profile in profiles.values():
            output.append({"target_match_id": match_id, **asdict(profile)})
    write_jsonl(output_root / "profiles" / "lagged_profiles.jsonl", output)
    earliest_dates = len({row.match_date for row in measurements})
    status = "available" if output else "insufficient_data"
    reason = None if output else f"all {len(match_dates)} supplied matches share the same date; no strictly earlier match is available for lagged profiles"
    payload = {"stage": "profiles", "status": status, "reason": reason, "measurement_rows": len(raw), "profile_rows": len(output), "distinct_dates": earliest_dates, "training_only_cutoffs": True}
    write_json(output_root / "profiles-summary.json", payload)
    return payload


def _impact_rows(output_root: Path) -> tuple[list[ImpactRow], list[str]]:
    profiles = read_jsonl(output_root / "profiles" / "lagged_profiles.jsonl")
    profile_map = {(row["target_match_id"], row["player_id"]): row for row in profiles}
    rows: list[ImpactRow] = []; deficiencies = []
    for segment_path in output_root.glob("features/*/segments.jsonl"):
        match_id = segment_path.parent.name
        match_rows = read_jsonl(output_root / "canonical" / match_id / "matches.jsonl")
        events = read_jsonl(output_root / "canonical" / match_id / "events.jsonl")
        if not match_rows:
            continue
        match = match_rows[0]
        shots = [event for event in events if event.get("event_type") == "shot" and not event.get("penalty")]
        if any(event.get("xg") is None for event in shots):
            deficiencies.append(f"{match_id}: {sum(event.get('xg') is None for event in shots)} non-penalty shots lack xG")
            continue
        for segment in read_jsonl(segment_path):
            if not segment.get("eligible_11v11"):
                continue
            for own_key, opponent_key, team_id in (("home_players", "away_players", match["home_team_id"]), ("away_players", "home_players", match["away_team_id"])):
                own, opponent = tuple(segment[own_key]), tuple(segment[opponent_key])
                own_profiles = {player: tuple(profile_map[(match_id, player)]["values"].get(name, 0.0) for name in PROFILE_FEATURES) for player in own if (match_id, player) in profile_map}
                opponent_profiles = {player: tuple(profile_map[(match_id, player)]["values"].get(name, 0.0) for name in PROFILE_FEATURES) for player in opponent if (match_id, player) in profile_map}
                if len(own_profiles) != 11 or len(opponent_profiles) != 11:
                    deficiencies.append(f"{match_id}: segment lacks complete historical profiles")
                    continue
                npxg = sum(float(event["xg"]) for event in shots if event.get("team_id") == team_id and event.get("period") == segment["period"] and segment["start_s"] < event.get("period_clock_s", 0) <= segment["end_s"])
                rows.append(ImpactRow(match_id, segment["end_s"] - segment["start_s"], npxg, own, opponent, own_profiles, opponent_profiles))
    return rows, sorted(set(deficiencies))


def impact_stage(output_root: Path, config: ZCPVConfig) -> dict:
    rows, deficiencies = _impact_rows(output_root)
    if deficiencies or not rows:
        reason = "; ".join(deficiencies[:8]) if deficiencies else "no eligible impact rows; run features and profiles first"
        payload = {"stage": "impact", "status": "insufficient_data", "reason": reason, "impact_rows": len(rows), "deficiencies": len(deficiencies)}
        write_json(output_root / "impact-summary.json", payload); return payload
    matches = _match_order(output_root, {row.match_id for row in rows})
    if len(matches) < 4:
        payload = {"stage": "impact", "status": "insufficient_data", "reason": f"only {len(matches)} matches have complete impact rows; need at least 4 for chronological training and holdout", "impact_rows": len(rows)}
        write_json(output_root / "impact-summary.json", payload); return payload
    cutoff = min(len(matches) - 1, max(3, int(len(matches) * 0.8)))
    train_matches, test_matches = set(matches[:cutoff]), set(matches[cutoff:])
    fit = fit_impact_model([row for row in rows if row.match_id in train_matches], PROFILE_FEATURES, config.impact_beta_l2, config.impact_player_l2)
    if fit.status == "available":
        joblib.dump(fit, output_root / "models" / "impact.joblib")
        write_jsonl(output_root / "impact" / "heldout_rows.jsonl", [asdict(row) for row in rows if row.match_id in test_matches])
    payload = {"stage": "impact", "status": fit.status, "reason": fit.reason, "train_matches": sorted(train_matches), "test_matches": sorted(test_matches), "impact_rows": len(rows)}
    write_json(output_root / "impact-summary.json", payload); return payload


def evaluate_stage(output_root: Path) -> dict:
    model_path = output_root / "models" / "impact.joblib"; heldout_path = output_root / "impact" / "heldout_rows.jsonl"
    if not model_path.exists() or not heldout_path.exists():
        impact = json.loads((output_root / "impact-summary.json").read_text(encoding="utf-8")) if (output_root / "impact-summary.json").exists() else {}
        payload = {"stage": "evaluate", "status": "insufficient_data", "reason": impact.get("reason", "fit the impact stage first")}
        write_json(output_root / "evaluate-summary.json", payload); return payload
    fit = joblib.load(model_path)
    raw = read_jsonl(heldout_path)
    rows = [ImpactRow(row["match_id"], row["duration_seconds"], row["npxg"], tuple(row["own_players"]), tuple(row["opponent_players"]), row["own_profiles"], row["opponent_profiles"], tuple(row.get("context", ()))) for row in raw]
    observed = [row.npxg / (row.duration_seconds / 5400.0) for row in rows]
    predicted = [fit.predict_row(row) for row in rows]
    payload = {"stage": "evaluate", "status": "available", "rows": len(rows), "rate_metrics": regression_metrics(observed, predicted), "bootstrap_unit": "match", "upstream_uncertainty_included": False}
    write_json(output_root / "evaluate-summary.json", payload); return payload


def smoke_stage(output_root: Path, config: ZCPVConfig) -> dict:
    rng = np.random.default_rng(config.random_seed)
    X = rng.normal(size=(180, 5)); y = np.where(rng.random(180) < .25, rng.uniform(.03, .5, 180), 0.0)
    state_fit = fit_state_model(X[:140], y[:140], X[140:], y[140:], [f"x{i}" for i in range(5)], config.random_seed, "synthetic fixture", min_positive_windows=10)
    players_a = tuple(f"A{i}" for i in range(11)); players_b = tuple(f"B{i}" for i in range(11))
    pa = {player: (float(index == 0), 0.0) for index, player in enumerate(players_a)}
    pb = {player: (0.0, float(index == 0)) for index, player in enumerate(players_b)}
    rows = [row for index in range(6) for row in (ImpactRow(f"M{index}", 900, .35, players_a, players_b, pa, pb), ImpactRow(f"M{index}", 900, .15, players_b, players_a, pb, pa))]
    impact_fit = fit_impact_model(rows[:8], ("progression", "availability"), config.impact_beta_l2, config.impact_player_l2)
    heldout = rows[8:]; predicted = [impact_fit.predict_row(row) for row in heldout]
    observed = [row.npxg / (row.duration_seconds / 5400) for row in heldout]
    payload = {"stage": "smoke", "status": "complete", "fixture": "synthetic_end_to_end_software_only", "state_model": state_fit.status, "impact_model": impact_fit.status, "heldout_metrics": regression_metrics(observed, predicted), "scientific_result": False}
    write_json(output_root / "smoke-summary.json", payload); return payload


def export_stage(output_root: Path, app_output: Path, config: ZCPVConfig, audit: dict | None) -> dict:
    summaries = {}
    for stage in ("features", "state", "profiles", "impact", "evaluate"):
        path = output_root / f"{stage}-summary.json"
        summaries[stage] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_run"}
    impact = summaries["impact"]
    available = impact.get("status") == "available" and summaries["evaluate"].get("status") == "available"
    reason = None if available else impact.get("reason") or summaries["state"].get("reason") or "required stages have not completed"
    players = None
    if available and (output_root / "models" / "impact.joblib").exists():
        fit = joblib.load(output_root / "models" / "impact.joblib")
        goalkeeper_ids = {
            row["player_id"] for path in output_root.glob("canonical/*/lineups.jsonl")
            for row in read_jsonl(path) if row.get("role") == "TW"
        }
        players = []
        for player_id in fit.player_ids:
            if player_id in goalkeeper_ids:
                continue
            offense, defense, net = fit.player_effect(player_id, fit.reference_profiles[player_id])
            minutes = 90.0 * max(fit.offensive_exposure.get(player_id, 0.0), fit.defensive_exposure.get(player_id, 0.0))
            players.append({"player_id": player_id, "o_pivot": offense, "d_pivot": defense, "net_pivot": net, "minutes": minutes, "model_version": config.model_version, "reliability": "low" if minutes < 900 else "research"})
    payload = research_export(
        output_root / "pivot-v1.json", model_version=config.model_version, config_hash=config.hash,
        cutoff=config.train_cutoff, status="available" if available else "insufficient_data", reason=reason,
        players=players, data_summary={"dfl_matches": len((audit or {}).get("matches", [])), "tracking_source": "DFL/IDSSE", "pipeline_stages": {name: value.get("status") for name, value in summaries.items()}},
    )
    app_output.parent.mkdir(parents=True, exist_ok=True)
    app_output.write_text("// Generated by models/scripts/train_zcpv.py.\nconst researchV1 = " + json.dumps(payload, ensure_ascii=False, indent=2) + " as const;\nexport default researchV1;\n", encoding="utf-8")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "models" / "data"
    parser = argparse.ArgumentParser(description="Run the auditable PIVOT v1 research pipeline.")
    parser.add_argument("stage", choices=STAGES); parser.add_argument("--data-root", type=Path, default=data_root / "raw" / "dfl"); parser.add_argument("--output-root", type=Path, default=data_root / "processed" / "pivot-v1"); parser.add_argument("--config", type=Path, default=repo_root / "models" / "config" / "zcpv_v1.json"); parser.add_argument("--app-output", type=Path, default=repo_root / "frontend" / "app" / "research-v1.generated.ts"); parser.add_argument("--match"); parser.add_argument("--include-tracking", action="store_true")
    args = parser.parse_args(); config = ZCPVConfig.load(args.config)
    stages = ("audit", "features", "state", "profiles", "impact", "evaluate", "export", "smoke") if args.stage == "all" else (args.stage,)
    results = {}; audit = None
    for stage in stages:
        if stage == "audit": audit = audit_stage(args.data_root, args.output_root, args.include_tracking, args.match); results[stage] = audit
        elif stage == "features": results[stage] = features_stage(args.data_root, args.output_root, config, args.match)
        elif stage == "state": results[stage] = state_stage(args.output_root, config)
        elif stage == "profiles": results[stage] = profiles_stage(args.output_root, config)
        elif stage == "impact": results[stage] = impact_stage(args.output_root, config)
        elif stage == "evaluate": results[stage] = evaluate_stage(args.output_root)
        elif stage == "export":
            if audit is None and (args.output_root / "audit-summary.json").exists(): audit = json.loads((args.output_root / "audit-summary.json").read_text(encoding="utf-8"))
            results[stage] = export_stage(args.output_root, args.app_output, config, audit)
        elif stage == "smoke": results[stage] = smoke_stage(args.output_root, config)
    manifest = {"model_version": config.model_version, "config_hash": config.hash, "generated_at": datetime.now(timezone.utc).isoformat(), "seed": config.random_seed, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "stages": {name: result.get("status") for name, result in results.items()}}
    write_json(args.output_root / "run-manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
