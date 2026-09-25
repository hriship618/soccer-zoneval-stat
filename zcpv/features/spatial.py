from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot
from typing import Iterable, Sequence

import numpy as np

from ..config import ZCPVConfig
from ..zones import ZoneGrid


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    team_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    active: bool = True
    goalkeeper: bool = False


@dataclass(frozen=True)
class ReceiverOption:
    player_id: str
    distance_m: float
    ball_arrival_s: float
    receiver_arrival_s: float
    opponent_intercept_s: float
    interception_margin_s: float
    nearest_pressure_m: float
    destination_angle_rad: float
    feasible: bool
    heuristic: bool = True


def _distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    t = 0.0 if denominator == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    qx, qy = ax + t * dx, ay + t * dy
    return hypot(px - qx, py - qy), t


def candidate_receivers(
    ball_xy: tuple[float, float],
    possessor_id: str,
    reference_team_id: str,
    players: Sequence[PlayerState],
    config: ZCPVConfig = ZCPVConfig(),
) -> list[ReceiverOption]:
    """Ground-pass feasibility using receiver and opponent arrival times.

    These outputs are documented heuristics, not calibrated completion
    probabilities.
    """
    bx, by = ball_xy
    teammates = [p for p in players if p.active and p.team_id == reference_team_id and p.player_id != possessor_id]
    opponents = [p for p in players if p.active and p.team_id != reference_team_id]
    options = []
    for receiver in teammates:
        distance = hypot(receiver.x - bx, receiver.y - by)
        ball_arrival = distance / config.pass_ball_speed_mps
        projected_x = receiver.x + receiver.vx * ball_arrival
        projected_y = receiver.y + receiver.vy * ball_arrival
        receiver_arrival = hypot(projected_x - receiver.x, projected_y - receiver.y) / config.receiver_speed_mps
        intercept_times = []
        pressures = []
        for opponent in opponents:
            lane_distance, fraction = _distance_to_segment(opponent.x, opponent.y, bx, by, projected_x, projected_y)
            lane_ball_time = ball_arrival * fraction
            intercept_times.append(hypot(lane_distance, 0.0) / config.receiver_speed_mps - lane_ball_time)
            pressures.append(hypot(opponent.x - projected_x, opponent.y - projected_y))
        opponent_margin = min(intercept_times, default=float("inf"))
        nearest_pressure = min(pressures, default=float("inf"))
        margin = opponent_margin - config.interception_margin_seconds
        options.append(ReceiverOption(
            player_id=receiver.player_id, distance_m=distance, ball_arrival_s=ball_arrival,
            receiver_arrival_s=receiver_arrival, opponent_intercept_s=opponent_margin + ball_arrival,
            interception_margin_s=margin, nearest_pressure_m=nearest_pressure,
            destination_angle_rad=atan2(projected_y - by, projected_x - bx),
            feasible=receiver_arrival <= ball_arrival and margin > 0.0,
        ))
    return options


def spatial_state_features(
    ball_xy: tuple[float, float],
    ball_velocity: tuple[float, float],
    reference_team_id: str,
    players: Sequence[PlayerState],
    *,
    score_difference: int = 0,
    restart_context: str = "open_play",
) -> dict[str, float]:
    active = [p for p in players if p.active]
    own = [p for p in active if p.team_id == reference_team_id]
    opponents = [p for p in active if p.team_id != reference_team_id]
    bx, by = ball_xy

    def dimensions(group: Sequence[PlayerState]) -> tuple[float, float]:
        if not group:
            return 0.0, 0.0
        return max(p.y for p in group) - min(p.y for p in group), max(p.x for p in group) - min(p.x for p in group)

    own_width, own_depth = dimensions(own)
    opponent_width, opponent_depth = dimensions(opponents)
    nearest = min((hypot(p.x - bx, p.y - by) for p in opponents), default=float("nan"))
    grid = ZoneGrid()
    occupancy = np.zeros((2, 12), dtype=float)
    for player in active:
        occupancy[0 if player.team_id == reference_team_id else 1, int(grid.index(player.x, player.y))] += 1
    result = {
        "ball_x": bx, "ball_y": by, "ball_speed": hypot(*ball_velocity),
        "team_width": own_width, "team_depth": own_depth,
        "opponent_width": opponent_width, "opponent_depth": opponent_depth,
        "nearest_opponent": nearest,
        "players_behind_ball": float(sum(p.x < bx for p in own)),
        "numerical_advantage": float(len(own) - len(opponents)),
        "score_difference": float(score_difference),
        "set_piece": float(restart_context != "open_play"),
        "own_goalkeeper_x": next((p.x for p in own if p.goalkeeper), float("nan")),
        "opponent_goalkeeper_x": next((p.x for p in opponents if p.goalkeeper), float("nan")),
    }
    for team_index, name in enumerate(("own", "opponent")):
        for zone in range(12):
            result[f"{name}_zone_{zone}_occupancy"] = occupancy[team_index, zone]
    return result


def pressure_measurements(
    timestamps: Sequence[float],
    carrier_positions: Sequence[tuple[float, float]],
    defender_positions: Sequence[dict[str, tuple[float, float, float, float]]],
    radius_m: float = 3.0,
    merge_gap_s: float = 1.0,
) -> list[dict]:
    """Build contiguous close-and-closing pressure episodes."""
    active: dict[str, dict] = {}
    complete: list[dict] = []
    for timestamp, carrier, defenders in zip(timestamps, carrier_positions, defender_positions):
        seen = set()
        for defender_id, (x, y, vx, vy) in defenders.items():
            dx, dy = carrier[0] - x, carrier[1] - y
            distance = hypot(dx, dy)
            closing = 0.0 if distance == 0 else (vx * dx + vy * dy) / distance
            if distance <= radius_m and closing > 0:
                seen.add(defender_id)
                episode = active.get(defender_id)
                if episode is None or timestamp - episode["end_s"] > merge_gap_s:
                    if episode is not None:
                        complete.append(episode)
                    episode = {"defender_id": defender_id, "start_s": timestamp, "end_s": timestamp, "samples": 0, "min_distance_m": distance}
                    active[defender_id] = episode
                episode["end_s"] = timestamp
                episode["samples"] += 1
                episode["min_distance_m"] = min(episode["min_distance_m"], distance)
        for defender_id in list(active):
            if defender_id not in seen and timestamp - active[defender_id]["end_s"] > merge_gap_s:
                complete.append(active.pop(defender_id))
    complete.extend(active.values())
    for episode in complete:
        episode["duration_s"] = episode["end_s"] - episode["start_s"]
    return complete


def lane_coverage(
    ball_xy: tuple[float, float],
    receivers: Sequence[PlayerState],
    defenders: Sequence[PlayerState],
    lane_radius_m: float = 1.5,
) -> dict[str, dict[str, float]]:
    """Split each covered lane equally among qualifying defenders."""
    totals = {defender.player_id: {"covered_lanes": 0.0, "threat_exposure": 0.0} for defender in defenders}
    for receiver in receivers:
        qualifiers = []
        for defender in defenders:
            distance, fraction = _distance_to_segment(defender.x, defender.y, *ball_xy, receiver.x, receiver.y)
            if distance <= lane_radius_m and 0.05 < fraction < 0.95:
                qualifiers.append(defender)
        for defender in defenders:
            totals[defender.player_id]["threat_exposure"] += 1.0
        for defender in qualifiers:
            totals[defender.player_id]["covered_lanes"] += 1.0 / len(qualifiers)
    return totals


def transition_protection(
    ball_xy: tuple[float, float],
    defenders: Sequence[PlayerState],
    threats: Sequence[PlayerState],
) -> dict[str, dict[str, float]]:
    output = {}
    for defender in defenders:
        nearest = min((hypot(defender.x - threat.x, defender.y - threat.y) for threat in threats), default=float("nan"))
        output[defender.player_id] = {
            "goal_side": float(defender.x <= min((threat.x for threat in threats), default=105.0)),
            "nearest_threat_m": nearest,
            "threat_exposure": float(len(threats)),
            "ball_distance_m": hypot(defender.x - ball_xy[0], defender.y - ball_xy[1]),
        }
    return output
