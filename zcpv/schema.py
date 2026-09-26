from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"


class PossessionState(str, Enum):
    HOME = "home"
    AWAY = "away"
    CONTESTED = "contested"
    UNKNOWN = "unknown"
    INTERRUPTION = "interruption"


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    competition: str
    season: str
    kickoff: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    pitch_length_m: float
    pitch_width_m: float
    source: str


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    match_id: str
    period: int
    period_clock_s: float
    source_timestamp: str
    team_id: str | None
    actor_id: str | None
    recipient_id: str | None
    event_type: str
    outcome: str | None
    start_x_m: float | None
    start_y_m: float | None
    end_x_m: float | None
    end_y_m: float | None
    penalty: bool
    xg: float | None
    xg_provenance: str | None
    possession_id: str | None
    live_play: bool


@dataclass(frozen=True)
class TrackingRecord:
    match_id: str
    period: int
    timestamp_s: float
    source_timestamp: str
    entity_id: str
    team_id: str | None
    x_m: float | None
    y_m: float | None
    attacking_x_m: float | None
    attacking_y_m: float | None
    vx_mps: float | None
    vy_mps: float | None
    active: bool
    quality: str
    is_ball: bool = False
    provider_possession_code: str | None = None
    ball_in_play: bool | None = None


@dataclass(frozen=True)
class LineupRecord:
    match_id: str
    player_id: str
    team_id: str
    start_s: float
    end_s: float
    role: str
    started: bool
    substitution_on_s: float | None = None
    substitution_off_s: float | None = None
    dismissal_s: float | None = None


@dataclass(frozen=True)
class StateSample:
    observation_id: str
    match_id: str
    period: int
    timestamp_s: float
    reference_team_id: str
    opponent_team_id: str
    features: dict[str, float | int | str | bool | None]
    reference_npxg_15s: float | None
    opponent_npxg_15s: float | None
    eligible: bool
    exclusion_reason: str | None = None


TABLE_TYPES = {
    "matches": MatchRecord,
    "events": EventRecord,
    "tracking": TrackingRecord,
    "lineups": LineupRecord,
    "state_samples": StateSample,
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, (datetime, Enum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def schema_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": {
            name: [{"name": f.name, "type": str(f.type)} for f in fields(record_type)]
            for name, record_type in TABLE_TYPES.items()
        },
        "coordinates": {
            "physical": "metres from the home-left pitch corner: x 0..length, y 0..width",
            "attacking": "physical coordinates rotated 180 degrees when the row's team attacks right-to-left",
        },
        "time": "continuous seconds within each period; source timestamp retained separately",
    }


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Dependency-free canonical interchange writer.

    Parquet is intentionally optional: JSONL keeps the audit command runnable in
    a clean environment while preserving types in ``schema.json``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_manifest(), indent=2), encoding="utf-8")
