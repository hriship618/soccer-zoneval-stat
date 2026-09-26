from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import PossessionState


@dataclass(frozen=True)
class PossessionFrame:
    period: int
    timestamp_s: float
    state: PossessionState
    team_id: str | None = None


@dataclass(frozen=True)
class PossessionEpisode:
    episode_id: str
    period: int
    start_s: float
    end_s: float
    state: PossessionState
    team_id: str | None


def segment_possessions(frames: Iterable[PossessionFrame], gap_seconds: float = 2.0) -> list[PossessionEpisode]:
    """Group stable team, contested, unknown and interruption runs.

    Consecutive unknown frames remain one unknown episode; they are never
    converted into a new possession per frame.
    """
    ordered = sorted(frames, key=lambda frame: (frame.period, frame.timestamp_s))
    if not ordered:
        return []
    episodes: list[PossessionEpisode] = []
    start = previous = ordered[0]
    sequence = 0
    for frame in ordered[1:]:
        changed = (
            frame.period != previous.period
            or frame.timestamp_s - previous.timestamp_s > gap_seconds
            or frame.state != previous.state
            or frame.team_id != previous.team_id
            or frame.state == PossessionState.INTERRUPTION
        )
        if changed:
            episodes.append(PossessionEpisode(f"pos-{sequence}", start.period, start.timestamp_s, previous.timestamp_s, start.state, start.team_id))
            sequence += 1
            start = frame
        previous = frame
    episodes.append(PossessionEpisode(f"pos-{sequence}", start.period, start.timestamp_s, previous.timestamp_s, start.state, start.team_id))
    return episodes
