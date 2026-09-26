from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ZCPVConfig:
    """Single versioned configuration for the v1 research pipeline."""

    model_version: str = "pivot-v1-research"
    random_seed: int = 618
    sample_hz: float = 1.0
    target_horizon_seconds: float = 15.0
    comparison_horizons_seconds: tuple[float, ...] = (10.0, 20.0)
    pass_ball_speed_mps: float = 14.0
    receiver_speed_mps: float = 7.0
    interception_margin_seconds: float = 0.25
    pressure_radius_m: float = 3.0
    pressure_merge_gap_seconds: float = 1.0
    pressure_outcome_seconds: float = 3.0
    transition_window_seconds: float = 5.0
    recency_half_life_matches: float = 8.0
    shrinkage_kappa: float = 10.0
    min_positive_windows: int = 25
    impact_beta_l2: float = 10.0
    impact_player_l2: float = 25.0
    train_cutoff: str | None = None
    validation_cutoff: str | None = None
    features: tuple[str, ...] = field(default_factory=lambda: (
        "ball_x", "ball_y", "ball_speed", "team_width", "team_depth",
        "opponent_width", "opponent_depth", "nearest_opponent",
        "players_behind_ball", "numerical_advantage", "score_difference",
        "set_piece", "own_goalkeeper_x", "opponent_goalkeeper_x", "zone_control",
    ))

    @classmethod
    def load(cls, path: Path) -> "ZCPVConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name in ("comparison_horizons_seconds", "features"):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()[:16]
