from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from typing import Any

from .adapters.dfl import DFLAdapter
from .config import ZCPVConfig
from .features.spatial import (
    PlayerState,
    SpatialSnapshot,
    candidate_receivers,
    lane_coverage,
    spatial_state_features,
    transition_protection_episodes,
)
from .profiles import PlayerMatchMeasurement
from .schema import StateSample, TrackingRecord
from .segments import Segment, build_segments
from .targets import Observation, label_states


EVENT_FEATURES = (
    "event_period_fraction", "event_score_difference", "event_set_piece",
    "event_last_x", "event_last_y", "event_seconds_since_action",
)


@dataclass
class MatchFeatureResult:
    match_id: str
    state_samples: list[StateSample]
    measurements: list[PlayerMatchMeasurement]
    segments: list[Segment]
    summary: dict[str, Any]


def _reference_coordinates(row: TrackingRecord, attacks_right: bool, length: float, width: float) -> tuple[float, float, float, float]:
    if row.x_m is None or row.y_m is None:
        return float("nan"), float("nan"), 0.0, 0.0
    vx, vy = row.vx_mps or 0.0, row.vy_mps or 0.0
    if attacks_right:
        return row.x_m, row.y_m, vx, vy
    return length - row.x_m, width - row.y_m, -vx, -vy


def _event_context(events, period: int, timestamp: float, reference_team: str, attacks_right: bool, length: float, width: float) -> dict[str, float | str]:
    prior = [event for event in events if event.period == period and event.period_clock_s <= timestamp]
    last = prior[-1] if prior else None
    goals_for = sum(event.outcome == "goal" and event.team_id == reference_team for event in prior)
    goals_against = sum(event.outcome == "goal" and event.team_id not in {None, reference_team} for event in prior)
    x = last.end_x_m if last and last.end_x_m is not None else last.start_x_m if last else None
    y = last.end_y_m if last and last.end_y_m is not None else last.start_y_m if last else None
    if x is not None and not attacks_right:
        x, y = length - x, width - (y or 0.0)
    restart_context = str(last.outcome) if last and last.event_type == "restart" and timestamp - last.period_clock_s <= 5.0 else "open_play"
    return {
        "event_period_fraction": min(timestamp / 3600.0, 1.5),
        "event_score_difference": float(goals_for - goals_against),
        "event_set_piece": float(restart_context != "open_play"),
        "restart_context": restart_context,
        "event_last_x": float(x or 0.0), "event_last_y": float(y or 0.0),
        "event_seconds_since_action": min(30.0, timestamp - last.period_clock_s) if last else 30.0,
    }


def extract_dfl_match(adapter: DFLAdapter, config: ZCPVConfig) -> MatchFeatureResult:
    """Connect canonical DFL inputs to observations and player-match evidence."""
    match = adapter.match()
    events, _ = adapter.events()
    lineups = adapter.lineups()
    player_meta = adapter.players()
    directions, _ = adapter.attacking_right()
    grouped: dict[tuple[int, str], list[TrackingRecord]] = defaultdict(list)
    for row in adapter.tracking(target_hz=config.sample_hz):
        grouped[(row.period, row.source_timestamp)].append(row)

    observations: list[Observation] = []
    snapshots: list[SpatialSnapshot] = []
    measurement_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    measurement_exposure: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pressure_last_seen: dict[tuple[str, int], float] = {}
    period_ends: dict[int, float] = defaultdict(float)
    dt = 1.0 / config.sample_hz
    unknown_possession = 0
    for (period, _), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ball = next((row for row in rows if row.is_ball), None)
        if ball is None or ball.x_m is None or ball.y_m is None:
            continue
        period_ends[period] = max(period_ends[period], ball.timestamp_s)
        code = None
        try:
            code = int(float(ball.provider_possession_code or ""))
        except ValueError:
            pass
        reference_team = match.home_team_id if code == 1 else match.away_team_id if code == 2 else None
        opponent_team = match.away_team_id if reference_team == match.home_team_id else match.home_team_id if reference_team == match.away_team_id else None
        if reference_team is None:
            unknown_possession += 1
        attacks_right = directions.get((reference_team or match.home_team_id, period), True)
        bx, by, bvx, bvy = _reference_coordinates(ball, attacks_right, match.pitch_length_m, match.pitch_width_m)
        players = []
        for row in rows:
            if row.is_ball or row.team_id is None or not row.active:
                continue
            x, y, vx, vy = _reference_coordinates(row, attacks_right, match.pitch_length_m, match.pitch_width_m)
            players.append(PlayerState(
                row.entity_id, row.team_id, x, y, vx, vy, True,
                player_meta.get(row.entity_id, {}).get("role") == "TW",
            ))
        snapshot = SpatialSnapshot(period, ball.timestamp_s, reference_team, bool(ball.ball_in_play), (bx, by), tuple(players))
        snapshots.append(snapshot)
        if not ball.ball_in_play or reference_team is None or opponent_team is None:
            continue
        possessor = min((player for player in players if player.team_id == reference_team), key=lambda player: hypot(player.x - bx, player.y - by), default=None)
        event_context = _event_context(events, period, ball.timestamp_s, reference_team, attacks_right, match.pitch_length_m, match.pitch_width_m)
        tracking_features = spatial_state_features((bx, by), (bvx, bvy), reference_team, players, score_difference=int(event_context["event_score_difference"]), restart_context=str(event_context["restart_context"]))
        features = dict(event_context)
        features.update({f"tracking_{name}": value for name, value in tracking_features.items()})
        options = candidate_receivers((bx, by), possessor.player_id if possessor else "", reference_team, players, config, restart_context=str(event_context["restart_context"]))
        features["tracking_feasible_receivers"] = float(sum(option.feasible for option in options))
        features["tracking_mean_interception_margin"] = float(sum(option.interception_margin_s for option in options) / len(options)) if options else 0.0
        observations.append(Observation(
            f"{match.match_id}:{period}:{ball.timestamp_s:.3f}", match.match_id, period, ball.timestamp_s,
            reference_team, opponent_team, True, features,
        ))
        for option in options:
            measurement_exposure[option.player_id]["receiving_availability"] += dt
            measurement_exposure[option.player_id]["option_quality"] += dt
            measurement_sums[option.player_id]["receiving_availability"] += dt * float(option.feasible)
            measurement_sums[option.player_id]["option_quality"] += dt * max(-5.0, min(5.0, option.interception_margin_s))
        if possessor is not None:
            for defender in (player for player in players if player.team_id == opponent_team):
                dx, dy = possessor.x - defender.x, possessor.y - defender.y
                distance = hypot(dx, dy)
                closing = 0.0 if distance == 0 else ((defender.vx - possessor.vx) * dx + (defender.vy - possessor.vy) * dy) / distance
                measurement_exposure[defender.player_id]["pressure"] += dt
                pressing = distance <= config.pressure_radius_m and closing > 0
                measurement_sums[defender.player_id]["pressure"] += dt * float(pressing)
                measurement_exposure[defender.player_id]["pressure_episodes"] += dt
                if pressing:
                    pressure_key = (defender.player_id, period)
                    previous = pressure_last_seen.get(pressure_key)
                    if previous is None or ball.timestamp_s - previous > config.pressure_merge_gap_seconds + dt:
                        measurement_sums[defender.player_id]["pressure_episodes"] += 1.0
                    pressure_last_seen[pressure_key] = ball.timestamp_s
        receivers = [player for player in players if player.team_id == reference_team and (possessor is None or player.player_id != possessor.player_id)]
        defenders = [player for player in players if player.team_id == opponent_team]
        for player_id, values in lane_coverage((bx, by), receivers, defenders).items():
            measurement_sums[player_id]["lane_coverage"] += values["raw_covered_lanes"]
            measurement_sums[player_id]["danger_lane_coverage"] += values["danger_weighted_coverage"]
            measurement_exposure[player_id]["lane_coverage"] += values["threat_exposure"]
            measurement_exposure[player_id]["danger_lane_coverage"] += values["danger_exposure"]

    for event in events:
        if event.actor_id is None or event.event_type not in {"pass", "carry"} or event.start_x_m is None or event.end_x_m is None:
            continue
        attacks_right = directions.get((event.team_id or "", event.period), True)
        advancement = (event.end_x_m - event.start_x_m) * (1 if attacks_right else -1) / match.pitch_length_m
        measurement_sums[event.actor_id]["progression"] += advancement if event.outcome in {"successfullyCompleted", "successful"} else -max(0.0, advancement)
        measurement_exposure[event.actor_id]["progression"] += 1.0

    transition_episodes = transition_protection_episodes(snapshots, config.transition_window_seconds)
    for episode in transition_episodes:
        for player_id, values in episode["players"].items():
            samples = max(1.0, values["samples"])
            measurement_sums[player_id]["transition_protection"] += values["goal_side"] / samples
            measurement_exposure[player_id]["transition_protection"] += 1.0

    samples = label_states(observations, events, {(match.match_id, period): end for period, end in period_ends.items()}, config.target_horizon_seconds)
    lineup_by_player = {row.player_id: row for row in lineups}
    measurements = []
    feature_names = ("progression", "receiving_availability", "option_quality", "pressure", "pressure_episodes", "lane_coverage", "danger_lane_coverage", "transition_protection")
    for player_id, lineup in lineup_by_player.items():
        values = {}
        exposures = {}
        for name in feature_names:
            exposure = measurement_exposure[player_id].get(name, 0.0)
            exposures[name] = exposure
            values[name] = measurement_sums[player_id].get(name, 0.0) / exposure if exposure > 0 else None
        measurements.append(PlayerMatchMeasurement(
            player_id, match.match_id, match.kickoff[:10], lineup.role,
            max(0.0, lineup.end_s - lineup.start_s) / 60.0, values, exposures,
        ))
    segments = build_segments(match.match_id, match.home_team_id, match.away_team_id, lineups, dict(period_ends), events)
    return MatchFeatureResult(
        match.match_id, samples, measurements, segments,
        {
            "tracking_rows": sum(len(rows) for rows in grouped.values()), "observations": len(observations),
            "eligible_targets": sum(sample.eligible for sample in samples),
            "missing_xg_targets": sum(sample.exclusion_reason == "missing_xg_target" for sample in samples),
            "unknown_possession_frames": unknown_possession, "player_measurements": len(measurements),
            "transition_episodes": len(transition_episodes), "segments": len(segments),
            "feature_sets": {"event_only": list(EVENT_FEATURES), "tracking_augmented": sorted(samples[0].features) if samples else []},
        },
    )
