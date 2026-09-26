from dataclasses import replace
import json

from scripts.train_zcpv import profiles_stage, state_stage
from zcpv.config import ZCPVConfig


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_suitable_artifacts_execute_state_and_profile_stages(tmp_path):
    for match_index in range(5):
        match_id = f"M{match_index}"
        write_jsonl(tmp_path / "canonical" / match_id / "matches.jsonl", [{"match_id": match_id, "kickoff": f"2022-01-{match_index + 1:02d}T12:00:00"}])
        samples = []
        measurements = []
        for sample_index in range(12):
            samples.append({
                "observation_id": f"{match_id}:{sample_index}", "match_id": match_id,
                "features": {"event_period_fraction": sample_index / 12, "event_score_difference": 0.0, "event_set_piece": 0.0, "event_last_x": 50.0, "event_last_y": 34.0, "event_seconds_since_action": 1.0, "tracking_ball_speed": float(sample_index)},
                "reference_npxg_15s": 0.2 if sample_index % 4 == 0 else 0.0,
                "opponent_npxg_15s": 0.1 if sample_index % 5 == 0 else 0.0,
                "eligible": True, "exclusion_reason": None,
            })
        for player_index in range(2):
            measurements.append({
                "player_id": f"P{player_index}", "match_id": match_id, "match_date": f"2022-01-{match_index + 1:02d}",
                "role": "MF", "minutes": 90.0,
                "features": {name: float(match_index + player_index) for name in ("progression", "receiving_availability", "option_quality", "pressure", "pressure_episodes", "lane_coverage", "danger_lane_coverage", "transition_protection")},
                "exposures": {name: 10.0 for name in ("progression", "receiving_availability", "option_quality", "pressure", "pressure_episodes", "lane_coverage", "danger_lane_coverage", "transition_protection")},
            })
        write_jsonl(tmp_path / "features" / match_id / "state_samples.jsonl", samples)
        write_jsonl(tmp_path / "features" / match_id / "player_match_measurements.jsonl", measurements)
    config = replace(ZCPVConfig(), min_positive_windows=2)
    assert state_stage(tmp_path, config)["status"] == "available"
    profile_result = profiles_stage(tmp_path, config)
    assert profile_result["status"] == "available"
    assert profile_result["profile_rows"] > 0
