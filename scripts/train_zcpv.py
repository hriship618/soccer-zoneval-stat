from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from zcpv.adapters.dfl import DFLAdapter
from zcpv.config import ZCPVConfig
from zcpv.export import research_export
from zcpv.features.spatial import PlayerState, candidate_receivers, spatial_state_features
from zcpv.models.impact import ImpactRow, fit_impact_model
from zcpv.models.state_value import fit_state_model
from zcpv.profiles import PlayerMatchMeasurement, build_lagged_profiles
from zcpv.quality import build_quality_report, file_checksum, write_quality_report
from zcpv.schema import write_jsonl, write_schema
from zcpv.targets import Observation, label_states


STAGES = ("audit", "features", "state", "profiles", "impact", "evaluate", "export", "smoke", "all")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def complete_matches(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DFL input root does not exist: {root}")
    matches = [path for path in sorted(root.iterdir()) if path.is_dir() and all((path / name).exists() for name in ("match.xml", "events.xml", "positions.xml"))]
    if not matches:
        raise FileNotFoundError(f"No complete DFL matches in {root}; expected match.xml, events.xml and positions.xml")
    return matches


def audit_stage(data_root: Path, output_root: Path, include_tracking: bool = False, match_id: str | None = None) -> dict:
    summaries = []
    write_schema(output_root / "schema.json")
    matches = complete_matches(data_root)
    if match_id is not None:
        matches = [path for path in matches if path.name == match_id]
        if not matches:
            raise FileNotFoundError(f"DFL match {match_id} is not complete under {data_root}")
    for match_dir in matches:
        adapter = DFLAdapter(match_dir)
        match = adapter.match()
        events, audit = adapter.events()
        lineups = adapter.lineups()
        match_output = output_root / "canonical" / match.match_id
        write_jsonl(match_output / "matches.jsonl", [match])
        write_jsonl(match_output / "events.jsonl", events)
        write_jsonl(match_output / "lineups.jsonl", lineups)
        tracking_rows = None
        if include_tracking:
            tracking_rows = list(adapter.tracking(target_hz=1.0))
            write_jsonl(match_output / "tracking.jsonl", tracking_rows)
            audit.tracking = {"sample_hz": 1.0, "rows": len(tracking_rows)}
        quality = build_quality_report(match, events, lineups, tracking_rows)
        quality["ingestion_audit"] = audit.to_dict()
        quality["checksums"] = {name: file_checksum(match_dir / name) for name in ("match.xml", "events.xml", "positions.xml")}
        report_name = f"{match.match_id}-tracking.json" if include_tracking else f"{match.match_id}.json"
        write_quality_report(output_root / "quality" / report_name, quality)
        summaries.append({
            "match_id": match.match_id, "status": quality["status"],
            "raw_events": audit.raw_events, "parsed_events": audit.parsed_events,
            "legacy_valued_events": audit.valued_legacy_events,
            "shots": audit.parsed_type_counts.get("shot", 0),
            "passes": audit.parsed_type_counts.get("pass", 0),
            "carries": audit.parsed_type_counts.get("carry", 0),
            "tracking_audited": include_tracking,
        })
    payload = {"stage": "audit", "status": "complete", "matches": summaries}
    write_json(output_root / "audit-summary.json", payload)
    return payload


def features_stage(data_root: Path, output_root: Path, config: ZCPVConfig) -> dict:
    # The provider has tracking, but no validated xG targets.  Record the
    # feature contract and a deterministic smoke calculation without implying a fit.
    players = [
        PlayerState("carrier", "A", 50, 34), PlayerState("receiver", "A", 65, 30, 1, 0),
        PlayerState("defender", "B", 58, 33, 0.5, 0), PlayerState("goalkeeper", "B", 100, 34, goalkeeper=True),
    ]
    options = candidate_receivers((50, 34), "carrier", "A", players, config)
    state = spatial_state_features((50, 34), (2, 0), "A", players)
    payload = {
        "stage": "features", "status": "infrastructure_ready",
        "sample_hz": config.sample_hz, "features": sorted(state),
        "receiver_proxy": asdict(options[0]) if options else None,
        "receiver_proxy_status": "heuristic_not_calibrated",
        "cache_key": config.hash,
        "note": "Real DFL feature extraction is available through DFLAdapter.tracking; production targets remain blocked without provider or frozen external xG.",
    }
    write_json(output_root / "features-summary.json", payload)
    return payload


def unavailable_stage(name: str, output_root: Path, reason: str) -> dict:
    payload = {"stage": name, "status": "insufficient_data", "reason": reason}
    write_json(output_root / f"{name}-summary.json", payload)
    return payload


def smoke_stage(output_root: Path, config: ZCPVConfig) -> dict:
    rng = np.random.default_rng(config.random_seed)
    X = rng.normal(size=(180, 5))
    y = np.where(rng.random(180) < 0.25, rng.uniform(0.03, 0.5, 180), 0.0)
    state_fit = fit_state_model(X[:140], y[:140], X[140:], y[140:], [f"x{i}" for i in range(5)], config.random_seed, "synthetic invariant fixture", min_positive_windows=10)
    feature_names = ("progression", "availability")
    rows = []
    players_a = tuple(f"A{i}" for i in range(11)); players_b = tuple(f"B{i}" for i in range(11))
    profiles_a = {player: (1.0 if player == "A0" else 0.0, 0.0) for player in players_a}
    profiles_b = {player: (0.0, 1.0 if player == "B0" else 0.0) for player in players_b}
    for match_index in range(4):
        rows.append(ImpactRow(f"M{match_index}", 900, 0.35, players_a, players_b, profiles_a, profiles_b))
        rows.append(ImpactRow(f"M{match_index}", 900, 0.15, players_b, players_a, profiles_b, profiles_a))
    impact_fit = fit_impact_model(rows, feature_names, config.impact_beta_l2, config.impact_player_l2)
    measurements = [
        PlayerMatchMeasurement("A0", f"M{i}", f"2022-01-{i + 1:02d}", "FW", 90, {"progression": float(i), "availability": .5}, {"progression": 5, "availability": 90})
        for i in range(3)
    ]
    profiles = build_lagged_profiles(measurements, "2022-01-04", feature_names, config.shrinkage_kappa, config.recency_half_life_matches)
    payload = {
        "stage": "smoke", "status": "complete", "fixture": "synthetic_software_invariant_only",
        "state_model": state_fit.status, "impact_model": impact_fit.status,
        "profile_players": len(profiles), "scientific_result": False,
    }
    write_json(output_root / "smoke-summary.json", payload)
    return payload


def export_stage(output_root: Path, app_output: Path, config: ZCPVConfig, audit: dict | None) -> dict:
    reason = "The seven DFL matches lack provider xG and are too small for credible chronological player-impact estimation. Infrastructure and audits are runnable; no v1 player ratings are published."
    payload = research_export(
        output_root / "zcpv-v1.json", model_version=config.model_version, config_hash=config.hash,
        cutoff=config.train_cutoff, status="insufficient_data", reason=reason, players=None,
        data_summary={"dfl_matches": len((audit or {}).get("matches", [])), "tracking_source": "DFL/IDSSE", "xg_target": "unavailable"},
    )
    app_output.parent.mkdir(parents=True, exist_ok=True)
    app_output.write_text("// Generated by scripts/train_zcpv.py.\nconst researchV1 = " + json.dumps(payload, ensure_ascii=False, indent=2) + " as const;\nexport default researchV1;\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the auditable ZCPV v1 research pipeline.")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/dfl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/zcpv-v1"))
    parser.add_argument("--config", type=Path, default=Path("config/zcpv_v1.json"))
    parser.add_argument("--app-output", type=Path, default=Path("app/research-v1.generated.ts"))
    parser.add_argument("--match", help="Limit audit materialization to one match ID")
    parser.add_argument("--include-tracking", action="store_true", help="Write 1 Hz canonical tracking JSONL (large and ignored by Git)")
    args = parser.parse_args()
    config = ZCPVConfig.load(args.config)
    stages = ("audit", "features", "state", "profiles", "impact", "evaluate", "export", "smoke") if args.stage == "all" else (args.stage,)
    results: dict[str, dict] = {}
    audit = None
    reason = "Provider/frozen external xG targets and sufficient chronological matches are unavailable for a credible DFL fit."
    for stage in stages:
        if stage == "audit":
            audit = audit_stage(args.data_root, args.output_root, args.include_tracking, args.match)
            results[stage] = audit
        elif stage == "features":
            results[stage] = features_stage(args.data_root, args.output_root, config)
        elif stage in {"state", "profiles", "impact", "evaluate"}:
            results[stage] = unavailable_stage(stage, args.output_root, reason)
        elif stage == "export":
            if audit is None and (args.output_root / "audit-summary.json").exists():
                audit = json.loads((args.output_root / "audit-summary.json").read_text(encoding="utf-8"))
            results[stage] = export_stage(args.output_root, args.app_output, config, audit)
        elif stage == "smoke":
            results[stage] = smoke_stage(args.output_root, config)
    manifest = {
        "model_version": config.model_version, "config_hash": config.hash,
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": config.random_seed,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "stages": {name: result.get("status") for name, result in results.items()},
    }
    write_json(args.output_root / "run-manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
