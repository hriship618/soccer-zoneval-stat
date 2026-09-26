from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET

from ..schema import EventRecord, LineupRecord, MatchRecord, TrackingRecord


PERIOD_BY_SECTION = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtraTime": 3, "secondHalfExtraTime": 4}


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _child(event: ET.Element, tag: str) -> ET.Element | None:
    return next((node for node in event.iter() if node.tag == tag), None)


def _first_present(*nodes: ET.Element | None) -> ET.Element | None:
    for node in nodes:
        if node is not None:
            return node
    return None


@dataclass
class DFLAudit:
    match_id: str
    raw_events: int = 0
    parsed_events: int = 0
    valued_legacy_events: int = 0
    raw_type_counts: Counter[str] = field(default_factory=Counter)
    parsed_type_counts: Counter[str] = field(default_factory=Counter)
    exclusions: Counter[str] = field(default_factory=Counter)
    alignment_samples: list[dict] = field(default_factory=list)
    direction_source: dict[str, str] = field(default_factory=dict)
    tracking: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "raw_events": self.raw_events,
            "parsed_events": self.parsed_events,
            "valued_legacy_events": self.valued_legacy_events,
            "raw_type_counts": dict(sorted(self.raw_type_counts.items())),
            "parsed_type_counts": dict(sorted(self.parsed_type_counts.items())),
            "exclusions": dict(sorted(self.exclusions.items())),
            "alignment_samples": self.alignment_samples,
            "direction_source": self.direction_source,
            "tracking": self.tracking,
        }


class DFLAdapter:
    """Normalize one DFL/IDSSE match without assigning scientific value.

    The adapter retains provider timestamps and physical coordinates.  It never
    treats the legacy hand-written xG formula as a target.
    """

    source = "DFL/IDSSE open data (CC BY 4.0)"

    def __init__(self, match_dir: Path):
        self.match_dir = Path(match_dir)
        self.metadata_path = self.match_dir / "match.xml"
        self.events_path = self.match_dir / "events.xml"
        self.tracking_path = self.match_dir / "positions.xml"
        missing = [path.name for path in (self.metadata_path, self.events_path, self.tracking_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete DFL match {self.match_dir.name}; missing: {', '.join(missing)}")
        self._metadata_root = ET.parse(self.metadata_path).getroot()
        self._events_root: ET.Element | None = None
        self._period_starts: dict[int, datetime] | None = None

    @property
    def events_root(self) -> ET.Element:
        if self._events_root is None:
            self._events_root = ET.parse(self.events_path).getroot()
        return self._events_root

    def _general(self) -> ET.Element:
        node = self._metadata_root.find("./MatchInformation/General")
        if node is None:
            raise ValueError(f"{self.metadata_path} has no MatchInformation/General")
        return node

    def match(self) -> MatchRecord:
        info = self._metadata_root.find("MatchInformation")
        general = self._general()
        environment = info.find("Environment") if info is not None else None
        if environment is None:
            raise ValueError(f"{self.metadata_path} has no Environment pitch dimensions")
        return MatchRecord(
            match_id=(general.get("MatchId") or "").replace("DFL-MAT-", ""),
            competition=general.get("CompetitionName") or "unknown",
            season=general.get("Season") or "unknown",
            kickoff=general.get("KickoffTime") or general.get("PlannedKickoffTime") or "",
            home_team_id=general.get("HomeTeamId") or "",
            home_team_name=general.get("HomeTeamName") or "",
            away_team_id=general.get("GuestTeamId") or "",
            away_team_name=general.get("GuestTeamName") or "",
            pitch_length_m=float(environment.get("PitchX") or 105.0),
            pitch_width_m=float(environment.get("PitchY") or 68.0),
            source=self.source,
        )

    def players(self) -> dict[str, dict]:
        output: dict[str, dict] = {}
        for team in self._metadata_root.findall("./MatchInformation/Teams/Team"):
            team_id = team.get("TeamId") or ""
            for player in team.findall("./Players/Player"):
                player_id = player.get("PersonId") or ""
                output[player_id] = {
                    "player_id": player_id,
                    "team_id": team_id,
                    "role": player.get("PlayingPosition") or "SUB",
                    "started": player.get("Starting") == "true",
                    "name": player.get("Shortname") or player_id,
                }
        return output

    def period_starts(self) -> dict[int, datetime]:
        if self._period_starts is not None:
            return self._period_starts
        starts: dict[int, datetime] = {}
        for event in self.events_root.findall("Event"):
            kickoff = _first_present(_child(event, "KickOff"), _child(event, "Kickoff"))
            section = kickoff.get("GameSection") if kickoff is not None else None
            timestamp = _dt(event.get("EventTime"))
            if section in PERIOD_BY_SECTION and timestamp is not None:
                starts.setdefault(PERIOD_BY_SECTION[section], timestamp)
        self._period_starts = starts
        return starts

    def _period_and_clock(self, timestamp: datetime) -> tuple[int, float]:
        starts = self.period_starts()
        candidates = [(period, start) for period, start in starts.items() if timestamp >= start]
        if not candidates:
            return 1, 0.0
        period, start = max(candidates, key=lambda item: item[1])
        return period, max(0.0, (timestamp - start).total_seconds())

    def attacking_right(self) -> tuple[dict[tuple[str, int], bool], dict[str, str]]:
        """Read provider TeamLeft directions and use shot medians only as fallback."""
        directions: dict[tuple[str, int], bool] = {}
        sources: dict[str, str] = {}
        for event in self.events_root.findall("Event"):
            kickoff = _first_present(_child(event, "KickOff"), _child(event, "Kickoff"))
            if kickoff is None:
                continue
            section = kickoff.get("GameSection")
            if section not in PERIOD_BY_SECTION:
                continue
            period = PERIOD_BY_SECTION[section]
            left, right = kickoff.get("TeamLeft"), kickoff.get("TeamRight")
            if left:
                directions[(left, period)] = True
                sources[f"{left}:{period}"] = "provider_team_left"
            if right:
                directions[(right, period)] = False
                sources[f"{right}:{period}"] = "provider_team_right"
        match = self.match()
        for period in self.period_starts():
            for team_id in (match.home_team_id, match.away_team_id):
                if (team_id, period) not in directions:
                    previous = directions.get((team_id, period - 1), period % 2 == 1)
                    directions[(team_id, period)] = not previous if period > 1 else previous
                    sources[f"{team_id}:{period}"] = "inferred_alternating_fallback"
        return directions, sources

    def events(self) -> tuple[list[EventRecord], DFLAudit]:
        match = self.match()
        players = self.players()
        records: list[EventRecord] = []
        audit = DFLAudit(match_id=match.match_id)
        directions, direction_sources = self.attacking_right()
        audit.direction_source = direction_sources
        supported = {
            "ShotAtGoal": "shot", "Pass": "pass", "Run": "carry",
            "TacklingGame": "tackle", "BallClaiming": "recovery",
            "Foul": "foul", "Substitution": "substitution",
            "Caution": "card",
            "KickOff": "restart", "CornerKick": "restart", "FreeKick": "restart",
            "ThrowIn": "restart", "GoalKick": "restart", "FinalWhistle": "interruption",
        }
        for event in self.events_root.findall("Event"):
            audit.raw_events += 1
            tags = [node.tag for node in event.iter() if node is not event]
            primary_tag = next((tag for tag in supported if tag in tags), tags[0] if tags else "empty")
            audit.raw_type_counts[primary_tag] += 1
            kind = supported.get(primary_tag)
            if kind is None:
                audit.exclusions[f"unsupported:{primary_tag}"] += 1
                continue
            timestamp = _dt(event.get("EventTime"))
            if timestamp is None:
                audit.exclusions["missing_timestamp"] += 1
                continue
            period, clock = self._period_and_clock(timestamp)
            actor_node = _first_present(_child(event, "ShotAtGoal"), _child(event, "Play"), _child(event, "TacklingGame"), _child(event, primary_tag))
            actor = None if actor_node is None else (actor_node.get("Player") or actor_node.get("Winner") or actor_node.get("PlayerIn"))
            team = None if actor_node is None else (actor_node.get("Team") or actor_node.get("WinnerTeam"))
            recipient = None if actor_node is None else actor_node.get("Recipient")
            if actor and actor not in players:
                audit.exclusions["invalid_actor"] += 1
                actor = None
            if team and team not in {match.home_team_id, match.away_team_id}:
                audit.exclusions["invalid_team"] += 1
                team = None
            coords: list[float | None] = []
            for name in ("X-Source-Position", "Y-Source-Position", "X-Position", "Y-Position"):
                try:
                    coords.append(float(event.get(name)))
                except (TypeError, ValueError):
                    coords.append(None)
            if kind in {"pass", "carry", "shot", "tackle", "recovery"} and coords[0] is None:
                audit.exclusions[f"missing_location:{kind}"] += 1
                continue
            outcome = None
            if actor_node is not None:
                outcome = actor_node.get("Evaluation") or actor_node.get("WinnerResult")
                if kind == "restart":
                    outcome = {
                        "GoalKick": "goal_kick", "ThrowIn": "throw_in", "CornerKick": "corner_kick",
                        "FreeKick": "free_kick", "KickOff": "kickoff",
                    }.get(primary_tag, "restart")
                if kind == "card":
                    outcome = actor_node.get("CardColor") or actor_node.get("CardRating")
                    if outcome and outcome.lower() in {"red", "yellowred", "yellow-red"}:
                        kind = "dismissal"
            if _child(event, "SuccessfulShot") is not None:
                outcome = "goal"
            penalty = _child(event, "Penalty") is not None or (actor_node is not None and actor_node.get("Penalty") == "true")
            possession = None if actor_node is None else actor_node.get("BallPossessionPhase")
            record = EventRecord(
                event_id=event.get("EventId") or f"{match.match_id}:{audit.raw_events}",
                match_id=match.match_id,
                period=period,
                period_clock_s=clock,
                source_timestamp=timestamp.isoformat(),
                team_id=team,
                actor_id=actor,
                recipient_id=recipient,
                event_type=kind,
                outcome=outcome,
                start_x_m=coords[0], start_y_m=coords[1], end_x_m=coords[2], end_y_m=coords[3],
                penalty=penalty,
                xg=None,
                xg_provenance=None,
                possession_id=possession,
                live_play=kind not in {"restart", "interruption", "substitution"},
            )
            records.append(record)
            audit.parsed_events += 1
            audit.parsed_type_counts[kind] += 1
            if kind in {"pass", "shot", "tackle", "carry"}:
                audit.valued_legacy_events += 1
            if kind in {"pass", "shot"} and len(audit.alignment_samples) < 12:
                audit.alignment_samples.append({
                    "event_id": record.event_id, "period": period, "clock_s": round(clock, 3),
                    "type": kind, "actor_id": actor, "team_id": team,
                    "start": [coords[0], coords[1]], "direction": "right" if directions.get((team or "", period)) else "left",
                })
        return records, audit

    def lineups(self) -> list[LineupRecord]:
        players = self.players()
        substitution_events: list[tuple[int, float, str | None, str | None]] = []
        dismissal_events: list[tuple[int, float, str | None]] = []
        ends = {period: 0.0 for period in self.period_starts()}
        for event in self.events_root.findall("Event"):
            timestamp = _dt(event.get("EventTime"))
            if timestamp is None:
                continue
            period, clock = self._period_and_clock(timestamp)
            ends[period] = max(ends.get(period, 0.0), clock)
            node = _child(event, "Substitution")
            if node is not None:
                substitution_events.append((period, clock, node.get("PlayerIn"), node.get("PlayerOut")))
            caution = _child(event, "Caution")
            if caution is not None and (caution.get("CardColor") or "").lower() in {"red", "yellowred", "yellow-red"}:
                dismissal_events.append((period, clock, caution.get("Player")))
        period_offsets: dict[int, float] = {}
        match_end = 0.0
        for period in sorted(ends):
            period_offsets[period] = match_end
            match_end += max(45 * 60, ends.get(period, 0.0))
        substitutions: dict[str, dict[str, float]] = {player_id: {} for player_id in players}
        for period, clock, player_in, player_out in substitution_events:
            absolute = period_offsets[period] + clock
            if player_in in substitutions:
                substitutions[player_in]["on"] = absolute
            if player_out in substitutions:
                substitutions[player_out]["off"] = absolute
        dismissals: dict[str, float] = {}
        for period, clock, player_id in dismissal_events:
            if player_id in substitutions:
                absolute = period_offsets[period] + clock
                dismissals[player_id] = absolute
                substitutions[player_id]["off"] = min(substitutions[player_id].get("off", absolute), absolute)
        rows = []
        for player_id, player in players.items():
            change = substitutions[player_id]
            start = 0.0 if player["started"] else change.get("on")
            if start is None:
                continue
            end = change.get("off", match_end)
            rows.append(LineupRecord(
                match_id=self.match().match_id, player_id=player_id, team_id=player["team_id"],
                start_s=float(start), end_s=float(max(start, end)), role=player["role"], started=player["started"],
                substitution_on_s=change.get("on"), substitution_off_s=change.get("off") if player_id not in dismissals else None,
                dismissal_s=dismissals.get(player_id),
            ))
        return rows

    def tracking(self, target_hz: float = 1.0) -> Iterator[TrackingRecord]:
        """Yield sampled canonical rows using actual timestamps for velocity.

        Frames are grouped by entity in the source, so the iterator keeps only
        one previous sample per entity.  Period changes and gaps invalidate
        velocity rather than producing a cross-period spike.
        """
        match = self.match()
        players = self.players()
        directions, _ = self.attacking_right()
        previous: dict[tuple[str, int], tuple[datetime, float, float]] = {}
        interval = 1.0 / target_hz
        for _, frame_set in ET.iterparse(self.tracking_path, events=("end",)):
            if frame_set.tag != "FrameSet":
                continue
            section = frame_set.get("GameSection")
            period = PERIOD_BY_SECTION.get(section or "")
            if period is None:
                frame_set.clear()
                continue
            entity = frame_set.get("PersonId") or frame_set.get("TeamId") or "unknown"
            is_ball = (frame_set.get("TeamId") or "").lower() == "ball"
            team_id = players.get(entity, {}).get("team_id")
            last_emitted: datetime | None = None
            for frame in frame_set:
                timestamp = _dt(frame.get("T"))
                if timestamp is None or (last_emitted is not None and (timestamp - last_emitted).total_seconds() < interval - 1e-4):
                    continue
                last_emitted = timestamp
                try:
                    x, y = float(frame.get("X")) + match.pitch_length_m / 2, float(frame.get("Y")) + match.pitch_width_m / 2
                except (TypeError, ValueError):
                    x = y = None
                start = self.period_starts().get(period)
                clock = 0.0 if start is None else max(0.0, (timestamp - start).total_seconds())
                vx = vy = None
                key = (entity, period)
                prior = previous.get(key)
                if prior is not None and x is not None and y is not None:
                    dt = (timestamp - prior[0]).total_seconds()
                    if 0 < dt <= interval * 2.5:
                        vx, vy = (x - prior[1]) / dt, (y - prior[2]) / dt
                if x is not None and y is not None:
                    previous[key] = (timestamp, x, y)
                attacks_right = directions.get((team_id or "", period), True)
                ax = x if attacks_right or x is None else match.pitch_length_m - x
                ay = y if attacks_right or y is None else match.pitch_width_m - y
                quality = "observed" if x is not None and y is not None else "missing_position"
                yield TrackingRecord(
                    match_id=match.match_id, period=period, timestamp_s=clock,
                    source_timestamp=timestamp.isoformat(), entity_id=entity, team_id=team_id,
                    x_m=x, y_m=y, attacking_x_m=ax, attacking_y_m=ay,
                    vx_mps=vx, vy_mps=vy, active=x is not None and y is not None,
                    quality=quality, is_ball=is_ball,
                    provider_possession_code=frame.get("BallPossession") if is_ball else None,
                    ball_in_play=(frame.get("BallStatus") == "1") if is_ball and frame.get("BallStatus") is not None else None,
                )
            frame_set.clear()
