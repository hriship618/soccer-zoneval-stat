from zcpv.config import ZCPVConfig
from zcpv.features.spatial import (
    PlayerState,
    SpatialSnapshot,
    candidate_receivers,
    lane_coverage,
    pressure_measurements,
    transition_protection_episodes,
)


def test_receiver_requires_lane_interception_margin():
    players = [
        PlayerState("carrier", "A", 10, 10),
        PlayerState("receiver", "A", 30, 10),
        PlayerState("blocker", "B", 20, 10),
    ]
    option = candidate_receivers((10, 10), "carrier", "A", players, ZCPVConfig())[0]
    assert not option.feasible
    assert option.heuristic


def test_pressure_episode_duration_stable_when_frames_are_duplicated():
    timestamps = [0.0, 1.0, 2.0]
    carrier = [(0.0, 0.0)] * 3
    defenders = [{"d": (-2.0 + t * 0.2, 0.0, 0.2, 0.0)} for t in timestamps]
    base = pressure_measurements(timestamps, carrier, defenders)[0]
    duplicated_times = [value for t in timestamps for value in (t, t + 0.01)]
    duplicated_carrier = [(0.0, 0.0)] * len(duplicated_times)
    duplicated_defenders = [{"d": (-2.0 + t * 0.2, 0.0, 0.2, 0.0)} for t in duplicated_times]
    duplicated = pressure_measurements(duplicated_times, duplicated_carrier, duplicated_defenders)[0]
    assert abs(base["duration_s"] - duplicated["duration_s"]) < 0.02


def test_open_play_offside_and_restart_exception():
    players = [
        PlayerState("carrier", "A", 60, 34), PlayerState("receiver", "A", 90, 34),
        PlayerState("d1", "B", 80, 20), PlayerState("d2", "B", 75, 40),
    ]
    open_play = candidate_receivers((60, 34), "carrier", "A", players)[0]
    throw_in = candidate_receivers((60, 34), "carrier", "A", players, restart_context="throw_in")[0]
    assert not open_play.onside and open_play.ineligibility_reason == "offside"
    assert throw_in.onside


def test_offside_is_mirrored_for_left_attacking_team():
    players = [
        PlayerState("carrier", "A", 45, 34), PlayerState("receiver", "A", 15, 34),
        PlayerState("d1", "B", 25, 20), PlayerState("d2", "B", 30, 40),
    ]
    option = candidate_receivers((45, 34), "carrier", "A", players, attacking_direction=-1)[0]
    assert not option.onside


def test_attacker_pulling_away_is_not_pressure():
    episodes = pressure_measurements(
        [0.0, 1.0], [(0.0, 0.0), (1.0, 0.0)],
        [{"d": (-2.0, 0.0, 1.0, 0.0)}, {"d": (-1.0, 0.0, 1.0, 0.0)}],
        carrier_velocities=[(2.0, 0.0), (2.0, 0.0)],
    )
    assert episodes == []


def test_lane_coverage_separates_raw_and_danger_weights():
    result = lane_coverage((20, 34), [PlayerState("r", "A", 90, 34)], [PlayerState("d", "B", 50, 34)])
    assert result["d"]["raw_covered_lanes"] == 1
    assert 0 < result["d"]["danger_weighted_coverage"] < 1


def test_transition_episode_stops_at_next_possession_change():
    players = (PlayerState("a", "A", 40, 34), PlayerState("b", "B", 60, 34))
    snapshots = [
        SpatialSnapshot(1, 0, "A", True, (50, 34), players),
        SpatialSnapshot(1, 1, "B", True, (51, 34), players),
        SpatialSnapshot(1, 2, "B", True, (52, 34), players),
        SpatialSnapshot(1, 3, "A", True, (53, 34), players),
    ]
    episodes = transition_protection_episodes(snapshots)
    assert episodes[0]["end_s"] == 2
    assert episodes[0]["stop_reason"] == "possession_change"
