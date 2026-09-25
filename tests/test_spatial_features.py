from zcpv.config import ZCPVConfig
from zcpv.features.spatial import PlayerState, candidate_receivers, pressure_measurements


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
