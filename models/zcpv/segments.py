from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import EventRecord, LineupRecord


@dataclass(frozen=True)
class Segment:
    match_id: str
    period: int
    start_s: float
    end_s: float
    home_players: tuple[str, ...]
    away_players: tuple[str, ...]
    eligible_11v11: bool
    exclusion_reason: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.end_s - self.start_s


def build_segments(
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    lineups: Iterable[LineupRecord],
    period_ends_s: dict[int, float],
    events: Iterable[EventRecord] = (),
) -> list[Segment]:
    """Split at period boundaries and simultaneous lineup changes."""
    lineups = list(lineups)
    event_times: dict[int, set[float]] = {}
    for event in events:
        if event.event_type in {"substitution", "dismissal"}:
            event_times.setdefault(event.period, set()).add(event.period_clock_s)
    output = []
    absolute_offset = 0.0
    for period, period_end in sorted(period_ends_s.items()):
        period_start_abs = absolute_offset
        period_end_abs = absolute_offset + period_end
        boundaries = {period_start_abs, period_end_abs}
        for row in lineups:
            if period_start_abs < row.start_s < period_end_abs:
                boundaries.add(row.start_s)
            if period_start_abs < row.end_s < period_end_abs:
                boundaries.add(row.end_s)
        boundaries.update(period_start_abs + value for value in event_times.get(period, set()) if 0 < value < period_end)
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            midpoint = (start + end) / 2
            active = [row for row in lineups if row.start_s <= midpoint < row.end_s]
            home = tuple(sorted(row.player_id for row in active if row.team_id == home_team_id))
            away = tuple(sorted(row.player_id for row in active if row.team_id == away_team_id))
            eligible = len(home) == 11 and len(away) == 11
            output.append(Segment(
                match_id=match_id, period=period, start_s=start - period_start_abs, end_s=end - period_start_abs,
                home_players=home, away_players=away, eligible_11v11=eligible,
                exclusion_reason=None if eligible else f"lineup_size:{len(home)}v{len(away)}",
            ))
        absolute_offset = period_end_abs
    return output
