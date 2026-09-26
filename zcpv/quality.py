from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from .schema import EventRecord, LineupRecord, MatchRecord, TrackingRecord


def file_checksum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def coordinate_round_trip(x: float, y: float, length: float, width: float) -> tuple[float, float]:
    return length - (length - x), width - (width - y)


def build_quality_report(
    match: MatchRecord,
    events: Iterable[EventRecord],
    lineups: Iterable[LineupRecord],
    tracking: Iterable[TrackingRecord] | None = None,
) -> dict:
    events = list(events)
    lineups = list(lineups)
    player_ids = {row.player_id for row in lineups}
    team_ids = {match.home_team_id, match.away_team_id}
    errors: list[str] = []
    warnings: list[str] = []
    event_counts = Counter(row.event_type for row in events)
    invalid_actors = [row.event_id for row in events if row.actor_id and row.actor_id not in player_ids]
    invalid_teams = [row.event_id for row in events if row.team_id and row.team_id not in team_ids]
    if invalid_actors:
        errors.append(f"{len(invalid_actors)} events reference actors outside the lineup table")
    if invalid_teams:
        errors.append(f"{len(invalid_teams)} events reference unknown teams")
    intervals: dict[str, float] = defaultdict(float)
    for row in lineups:
        if row.end_s < row.start_s:
            errors.append(f"negative lineup interval for {row.player_id}")
        intervals[row.team_id] += max(0.0, row.end_s - row.start_s)
    if event_counts.get("shot", 0) and not any(row.xg is not None for row in events if row.event_type == "shot"):
        warnings.append("provider xG is unavailable; production xG fitting is blocked")
    tracking_summary = None
    if tracking is not None:
        rows = list(tracking)
        missing = sum(not row.active for row in rows)
        speeds = [((row.vx_mps or 0.0) ** 2 + (row.vy_mps or 0.0) ** 2) ** 0.5 for row in rows if not row.is_ball and row.vx_mps is not None and row.vy_mps is not None]
        ball_speeds = [((row.vx_mps or 0.0) ** 2 + (row.vy_mps or 0.0) ** 2) ** 0.5 for row in rows if row.is_ball and row.vx_mps is not None and row.vy_mps is not None]
        impossible = sum(speed > 14.0 for speed in speeds)
        if impossible:
            warnings.append(f"{impossible} sampled velocities exceed 14 m/s and require review")
        tracking_summary = {
            "rows": len(rows), "missing_positions": missing,
            "missing_fraction": missing / max(1, len(rows)),
            "player_velocity_rows": len(speeds), "player_velocity_over_14_mps": impossible,
            "ball_velocity_rows": len(ball_speeds),
        }
    return {
        "schema_version": "1.0.0",
        "match_id": match.match_id,
        "status": "failed" if errors else "passed_with_warnings" if warnings else "passed",
        "event_counts": dict(sorted(event_counts.items())),
        "events": len(events),
        "lineups": len(lineups),
        "team_player_seconds": {team: round(seconds, 3) for team, seconds in intervals.items()},
        "invalid_actor_event_ids": invalid_actors[:20],
        "invalid_team_event_ids": invalid_teams[:20],
        "tracking": tracking_summary,
        "errors": errors,
        "warnings": warnings,
    }


def write_quality_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
