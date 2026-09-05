from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .zones import ZoneGrid


@dataclass(frozen=True)
class PitchControlConfig:
    reaction_time_s: float = 0.7
    max_speed_mps: float = 7.0
    velocity_weight: float = 0.45
    control_lambda: float = 1.35
    grid_x: int = 32
    grid_y: int = 21


def grid_points(config: PitchControlConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 105.0, config.grid_x, dtype=np.float32)
    ys = np.linspace(0.0, 68.0, config.grid_y, dtype=np.float32)
    x, y = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack((x.ravel(), y.ravel())), ZoneGrid().index(x.ravel(), y.ravel())


def interception_times(positions: np.ndarray, velocities: np.ndarray, points: np.ndarray, config: PitchControlConfig) -> np.ndarray:
    """Reaction-adjusted time to intercept for [frame, player, grid-cell]."""
    projected = positions[:, :, None, :] + config.reaction_time_s * config.velocity_weight * velocities[:, :, None, :]
    distance = np.linalg.norm(points[None, None, :, :] - projected, axis=-1)
    return config.reaction_time_s + distance / config.max_speed_mps


def leave_one_out_zone_values(
    positions: np.ndarray,
    velocities: np.ndarray,
    teams: np.ndarray,
    zone_values: np.ndarray,
    config: PitchControlConfig = PitchControlConfig(),
) -> np.ndarray:
    """Return weighted team-control loss as [frame, player, zone].

    The algebraic leave-one-out form computes every counterfactual without
    materializing 23 full pitch-control surfaces.
    """
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    teams = np.asarray(teams, dtype=np.int8)
    zone_values = np.asarray(zone_values, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[2] != 2 or velocities.shape != positions.shape:
        raise ValueError("positions and velocities must have shape [frames, players, 2]")
    if teams.shape != (positions.shape[1],) or set(np.unique(teams)) - {0, 1}:
        raise ValueError("teams must be a player-length 0/1 vector")
    if zone_values.shape not in {(12,), (2, 12)}:
        raise ValueError("zone_values must contain 12 values or have shape [2, 12]")

    points, zone_ids = grid_points(config)
    active = np.isfinite(positions).all(axis=-1)
    safe_positions = np.nan_to_num(positions, nan=0.0)
    safe_velocities = np.nan_to_num(velocities, nan=0.0)
    weights = np.exp(-config.control_lambda * interception_times(safe_positions, safe_velocities, points, config))
    weights *= active[:, :, None]
    team_sum = np.stack((weights[:, teams == 0].sum(axis=1), weights[:, teams == 1].sum(axis=1)), axis=1)
    total = np.maximum(team_sum.sum(axis=1), 1e-12)
    full_control = team_sum / total[:, None, :]
    result = np.zeros((positions.shape[0], positions.shape[1], 12), dtype=np.float32)
    for player in range(positions.shape[1]):
        own = teams[player]
        removed = weights[:, player]
        without_total = np.maximum(total - removed, 1e-12)
        without_own = (team_sum[:, own] - removed) / without_total
        delta = np.maximum(0.0, full_control[:, own] - without_own)
        for zone in range(12):
            cells = zone_ids == zone
            weight = zone_values[zone] if zone_values.ndim == 1 else zone_values[own, zone]
            result[:, player, zone] = delta[:, cells].mean(axis=1) * weight
    return result


def naive_leave_one_out_zone_values(positions: np.ndarray, velocities: np.ndarray, teams: np.ndarray, zone_values: np.ndarray, config: PitchControlConfig = PitchControlConfig()) -> np.ndarray:
    """Readable recomputation baseline used for correctness and speedup tests."""
    points, zone_ids = grid_points(config)
    times = interception_times(np.asarray(positions), np.asarray(velocities), points, config)
    weights = np.exp(-config.control_lambda * times)
    active = np.isfinite(positions).all(axis=-1)
    weights = np.nan_to_num(weights, nan=0.0) * active[:, :, None]
    out = np.zeros((len(positions), positions.shape[1], 12), dtype=np.float32)
    for frame in range(len(positions)):
        for player in range(positions.shape[1]):
            own = teams[player]
            full = weights[frame, teams == own].sum(axis=0) / weights[frame].sum(axis=0)
            keep = np.arange(positions.shape[1]) != player
            reduced = weights[frame, keep]
            reduced_teams = teams[keep]
            without = reduced[reduced_teams == own].sum(axis=0) / reduced.sum(axis=0)
            delta = np.maximum(0.0, full - without)
            for zone in range(12):
                out[frame, player, zone] = delta[zone_ids == zone].mean() * zone_values[zone]
    return out
