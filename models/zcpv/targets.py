from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import EventRecord, StateSample


@dataclass(frozen=True)
class Observation:
    observation_id: str
    match_id: str
    period: int
    timestamp_s: float
    possession_team_id: str | None
    opponent_team_id: str | None
    live_play: bool
    features: dict[str, float | int | str | bool | None]
    possession_known: bool = True


def label_states(
    observations: Iterable[Observation],
    events: Iterable[EventRecord],
    period_ends_s: dict[tuple[str, int], float],
    horizon_s: float = 15.0,
    synchronization_guard_s: float = 0.25,
    require_xg: bool = True,
) -> list[StateSample]:
    """Label fixed-reference-team future non-penalty xG.

    Reference identity is copied from the observation and never changes after a
    turnover.  Windows are right-open at t and closed at t+horizon.
    """
    shots: dict[tuple[str, int], list[EventRecord]] = {}
    for event in events:
        if event.event_type == "shot" and not event.penalty:
            shots.setdefault((event.match_id, event.period), []).append(event)
    output = []
    for observation in observations:
        reason = None
        if not observation.live_play:
            reason = "not_live_play"
        elif not observation.possession_known or not observation.possession_team_id:
            reason = "unknown_or_contested_possession"
        elif not observation.opponent_team_id:
            reason = "unknown_opponent"
        end = period_ends_s.get((observation.match_id, observation.period))
        if reason is None and (end is None or observation.timestamp_s + horizon_s > end + 1e-9):
            reason = "right_censored"
        candidates = shots.get((observation.match_id, observation.period), [])
        if reason is None and any(abs(shot.period_clock_s - observation.timestamp_s) <= synchronization_guard_s for shot in candidates):
            reason = "ambiguous_shot_synchronization"
        future = [shot for shot in candidates if observation.timestamp_s < shot.period_clock_s <= observation.timestamp_s + horizon_s]
        if reason is None and require_xg and any(shot.xg is None for shot in future):
            reason = "missing_xg_target"
        reference, opponent = 0.0, 0.0
        if reason is None:
            for shot in future:
                value = float(shot.xg or 0.0)
                if shot.team_id == observation.possession_team_id:
                    reference += value
                elif shot.team_id == observation.opponent_team_id:
                    opponent += value
        output.append(StateSample(
            observation_id=observation.observation_id,
            match_id=observation.match_id,
            period=observation.period,
            timestamp_s=observation.timestamp_s,
            reference_team_id=observation.possession_team_id or "",
            opponent_team_id=observation.opponent_team_id or "",
            features=dict(observation.features),
            reference_npxg_15s=reference if reason is None else None,
            opponent_npxg_15s=opponent if reason is None else None,
            eligible=reason is None,
            exclusion_reason=reason,
        ))
    return output
