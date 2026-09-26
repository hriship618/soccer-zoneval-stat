from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, pi, sqrt
from pathlib import Path

import numpy as np


ACTION_TYPES = (
    "Pass", "Carry", "Shot", "Dribble", "Ball Recovery", "Interception",
    "Duel", "Clearance", "Miscontrol", "Dispossessed", "Foul Won", "Other",
)
TYPE_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}
MOVE_TYPES = {"Pass", "Carry", "Dribble"}
INCLUDED_RAW_TYPES = {
    "Pass", "Carry", "Shot", "Dribble", "Ball Recovery", "Interception",
    "Duel", "Clearance", "Miscontrol", "Dispossessed", "Foul Won",
    "Goal Keeper", "Block", "50/50",
}


@dataclass(frozen=True)
class SBAction:
    match_id: int | str
    index: int
    period: int
    minute: int
    second: int
    possession: int
    team_id: int
    team_name: str
    player_id: int | None
    player_name: str
    action_type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    successful: bool
    goal: bool
    shot_xg: float
    score_for: int
    score_against: int


def _location(event: dict, key: str = "location") -> tuple[float, float] | None:
    value = event.get(key)
    if not value or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _action_type(raw_type: str) -> str:
    return raw_type if raw_type in TYPE_INDEX else "Other"


def _end_location(event: dict, raw_type: str, start: tuple[float, float]) -> tuple[float, float]:
    detail = event.get(raw_type.lower().replace(" ", "_"), {})
    end = detail.get("end_location") if isinstance(detail, dict) else None
    if end and len(end) >= 2:
        return float(end[0]), float(end[1])
    return start


def _successful(event: dict, raw_type: str) -> bool:
    detail = event.get(raw_type.lower().replace(" ", "_"), {})
    outcome = detail.get("outcome", {}).get("name") if isinstance(detail, dict) else None
    if raw_type in {"Pass", "Dribble", "Interception", "Duel"}:
        return outcome in {None, "Complete", "Success", "Won", "Success In Play", "Success Out"}
    if raw_type in {"Miscontrol", "Dispossessed"}:
        return False
    return True


def load_match(path: Path) -> list[SBAction]:
    events = json.loads(path.read_text(encoding="utf-8"))
    match_id = int(path.stem)
    scores: dict[int, int] = {}
    actions: list[SBAction] = []
    for event in events:
        period = int(event.get("period", 1))
        # Period 5 is a penalty shootout, not ordinary match play.
        if period > 4:
            continue
        start = _location(event)
        team = event.get("team") or {}
        raw_type = (event.get("type") or {}).get("name", "Other")
        if start is None or not team.get("id") or raw_type not in INCLUDED_RAW_TYPES:
            continue
        team_id = int(team["id"])
        scores.setdefault(team_id, 0)
        opponents = [score for candidate, score in scores.items() if candidate != team_id]
        score_against = opponents[0] if opponents else 0
        shot = event.get("shot") or {}
        goal = raw_type == "Shot" and (shot.get("outcome") or {}).get("name") == "Goal"
        player = event.get("player") or {}
        action_type = _action_type(raw_type)
        end_x, end_y = _end_location(event, raw_type, start)
        actions.append(SBAction(
            match_id=match_id,
            index=int(event.get("index", len(actions))),
            period=period,
            minute=int(event.get("minute", 0)),
            second=int(event.get("second", 0)),
            possession=int(event.get("possession", 0)),
            team_id=team_id,
            team_name=str(team.get("name", "Unknown")),
            player_id=int(player["id"]) if player.get("id") is not None else None,
            player_name=str(player.get("name", "Unknown")),
            action_type=action_type,
            start_x=start[0], start_y=start[1], end_x=end_x, end_y=end_y,
            successful=_successful(event, raw_type), goal=goal,
            shot_xg=float(shot.get("statsbomb_xg", 0.0)),
            score_for=scores[team_id], score_against=score_against,
        ))
        if goal:
            scores[team_id] += 1
    return actions


def load_competition(root: Path) -> tuple[list[SBAction], list[dict]]:
    matches = json.loads((root / "matches.json").read_text(encoding="utf-8"))
    actions: list[SBAction] = []
    for match in matches:
        actions.extend(load_match(root / "events" / f"{match['match_id']}.json"))
    return actions, matches


def split_match_ids(matches: list[dict], test_fraction: float = 0.2) -> tuple[set[int], set[int], str]:
    ordered = sorted(matches, key=lambda row: (row["match_date"], row["kick_off"], row["match_id"]))
    test_count = max(1, round(len(ordered) * test_fraction))
    train, test = ordered[:-test_count], ordered[-test_count:]
    return {int(row["match_id"]) for row in train}, {int(row["match_id"]) for row in test}, test[0]["match_date"]


def split_match_ids_three_way(matches: list[dict], validation_fraction: float = 0.2, test_fraction: float = 0.2) -> tuple[set[int], set[int], set[int], str, str]:
    """Chronological, match-disjoint train/validation/test split."""
    ordered = sorted(matches, key=lambda row: (row["match_date"], row["kick_off"], row["match_id"]))
    validation_count = max(1, round(len(ordered) * validation_fraction))
    test_count = max(1, round(len(ordered) * test_fraction))
    if validation_count + test_count >= len(ordered):
        raise ValueError("not enough matches for three non-empty partitions")
    train = ordered[:-(validation_count + test_count)]
    validation = ordered[-(validation_count + test_count):-test_count]
    test = ordered[-test_count:]
    return (
        {int(row["match_id"]) for row in train},
        {int(row["match_id"]) for row in validation},
        {int(row["match_id"]) for row in test},
        validation[0]["match_date"], test[0]["match_date"],
    )


def _goal_geometry(x: float, y: float) -> tuple[float, float]:
    dx, dy = max(0.001, 1.0 - x), abs(0.5 - y)
    distance = sqrt(dx * dx + dy * dy)
    angle = abs(atan2(0.0915 * dx, dx * dx + dy * dy - (0.0915 / 2) ** 2)) / pi
    return distance, angle


def shot_features(actions: list[SBAction]) -> np.ndarray:
    """Provider-neutral shot geometry used to transfer World Cup xG to DFL."""
    rows = []
    for action in actions:
        distance, angle = _goal_geometry(action.end_x / 120, action.end_y / 80)
        rows.append((action.end_x / 120, action.end_y / 80, distance, angle))
    return np.asarray(rows, dtype=np.float32).reshape((-1, 4))


def _state_feature_row(match_actions: list[SBAction], index: int, lags: int = 3) -> np.ndarray:
    """Encode one completed-action state in the provider-neutral event schema."""
    current = match_actions[index]
    width_per_lag = 9 + len(ACTION_TYPES)
    row = np.zeros(4 + width_per_lag * lags, dtype=np.float32)
    row[0] = min(current.minute, 130) / 130
    row[1] = current.period / 5
    row[2] = np.clip(current.score_for - current.score_against, -4, 4) / 4
    row[3] = 1.0 if current.possession else 0.0
    offset = 4
    for lag in range(lags):
        if index - lag < 0 or match_actions[index - lag].period != current.period:
            offset += width_per_lag
            continue
        action = match_actions[index - lag]
        sx, sy = action.start_x / 120, action.start_y / 80
        ex, ey = action.end_x / 120, action.end_y / 80
        distance, angle = _goal_geometry(ex, ey)
        numeric = (
            sx, sy, ex, ey, ex - sx, ey - sy, float(action.successful),
            float(action.team_id == current.team_id), distance,
        )
        row[offset:offset + 9] = numeric
        row[offset + 9 + TYPE_INDEX[action.action_type]] = 1.0
        if action.action_type == "Shot":
            row[offset + 8] = distance - angle
        offset += width_per_lag
    return row


def action_state_features(actions: list[SBAction], lags: int = 3) -> tuple[np.ndarray, list[SBAction]]:
    """Encode every action without requiring future labels.

    This is the shared representation used both for StatsBomb training states
    and synchronized DFL inference states. Coordinates must already be rotated
    so the acting team attacks left-to-right and scaled to 120 by 80.
    """
    by_match: dict[int | str, list[SBAction]] = {}
    for action in actions:
        by_match.setdefault(action.match_id, []).append(action)
    rows: list[np.ndarray] = []
    ordered: list[SBAction] = []
    for match_actions in by_match.values():
        match_actions.sort(key=lambda action: action.index)
        for index, action in enumerate(match_actions):
            rows.append(_state_feature_row(match_actions, index, lags))
            ordered.append(action)
    width = 4 + (9 + len(ACTION_TYPES)) * lags
    return np.asarray(rows, dtype=np.float32).reshape((-1, width)), ordered


def event_model_values(actions: list[SBAction], score_model, concede_model, lags: int = 3) -> list[dict]:
    """Return post-state probabilities and action deltas in the actor's frame.

    The pre-state is the immediately preceding same-period state. If that
    action belonged to the opponent, its net forecast is sign-flipped before
    comparison. No future action or label is used during scoring.
    """
    features, ordered = action_state_features(actions, lags=lags)
    if not ordered:
        return []
    score = score_model.predict_proba(features)[:, 1]
    concede = concede_model.predict_proba(features)[:, 1]
    output: list[dict] = []
    previous_by_match: dict[int | str, tuple[SBAction, float]] = {}
    for index, action in enumerate(ordered):
        post_net = float(score[index] - concede[index])
        previous = previous_by_match.get(action.match_id)
        if previous is None or previous[0].period != action.period:
            pre_net = 0.0
        else:
            pre_net = previous[1] if previous[0].team_id == action.team_id else -previous[1]
        output.append({
            "action": action,
            "score_probability": float(score[index]),
            "concede_probability": float(concede[index]),
            "net_state_value": post_net,
            "event_value": post_net - pre_net,
        })
        previous_by_match[action.match_id] = (action, post_net)
    return output


def state_features(actions: list[SBAction], horizon: int = 10, lags: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SBAction]]:
    """Post-action forecasts from the completed current action.

    Features include the current action endpoint. Labels begin strictly with
    the next action and require ``horizon`` subsequent actions in the same
    period; end-of-period observations are therefore explicitly censored.
    Histories and labels never cross period boundaries.
    """
    by_match: dict[int | str, list[SBAction]] = {}
    for action in actions:
        by_match.setdefault(action.match_id, []).append(action)
    features: list[np.ndarray] = []
    scores: list[int] = []
    concedes: list[int] = []
    ordered_actions: list[SBAction] = []
    for match_actions in by_match.values():
        match_actions.sort(key=lambda action: action.index)
        for index, current in enumerate(match_actions):
            future = match_actions[index + 1:index + 1 + horizon]
            if len(future) < horizon or any(action.period != current.period for action in future):
                continue
            row = _state_feature_row(match_actions, index, lags)
            scores.append(int(any(action.goal and action.team_id == current.team_id for action in future)))
            concedes.append(int(any(action.goal and action.team_id != current.team_id for action in future)))
            features.append(row)
            ordered_actions.append(current)
    return np.asarray(features), np.asarray(scores, dtype=np.int8), np.asarray(concedes, dtype=np.int8), ordered_actions


def fit_xt(actions: list[SBAction], grid_x: int = 16, grid_y: int = 12, tolerance: float = 1e-9) -> np.ndarray:
    zones = grid_x * grid_y
    transitions = np.zeros((zones, zones), dtype=np.float64)
    shots = np.zeros(zones, dtype=np.float64)
    goals = np.zeros(zones, dtype=np.float64)
    turnovers = np.zeros(zones, dtype=np.float64)

    def zone(x: float, y: float) -> int:
        x_index = min(grid_x - 1, max(0, int(x / 120 * grid_x)))
        y_index = min(grid_y - 1, max(0, int(y / 80 * grid_y)))
        return x_index * grid_y + y_index

    for action in actions:
        origin = zone(action.start_x, action.start_y)
        if action.action_type == "Shot":
            shots[origin] += 1
            goals[origin] += int(action.goal)
        elif action.action_type in MOVE_TYPES:
            if action.successful:
                transitions[origin, zone(action.end_x, action.end_y)] += 1
            else:
                turnovers[origin] += 1
    totals = transitions.sum(axis=1) + shots + turnovers
    safe = np.where(totals == 0, 1.0, totals)
    move_probability = transitions / safe[:, None]
    immediate_goal = goals / safe
    values = immediate_goal.copy()
    for _ in range(10_000):
        updated = immediate_goal + move_probability @ values
        if np.max(np.abs(updated - values)) < tolerance:
            return updated.reshape(grid_x, grid_y)
        values = updated
    raise RuntimeError("xT value iteration did not converge")


def xt_action_value(action: SBAction, grid: np.ndarray) -> float:
    grid_x, grid_y = grid.shape
    zone = lambda x, y: (min(grid_x - 1, max(0, int(x / 120 * grid_x))), min(grid_y - 1, max(0, int(y / 80 * grid_y))))
    origin = grid[zone(action.start_x, action.start_y)]
    if action.action_type == "Shot":
        return action.shot_xg - origin
    if action.action_type in MOVE_TYPES:
        return float(grid[zone(action.end_x, action.end_y)] - origin) if action.successful else float(-origin)
    return 0.0


def collapse_xt_grid(grid: np.ndarray, target_x: int = 4, target_y: int = 3) -> np.ndarray:
    """Area-average a fine xT surface into a coarser tactical grid."""
    grid = np.asarray(grid, dtype=np.float64)
    source_x, source_y = grid.shape
    if source_x % target_x or source_y % target_y:
        raise ValueError("source grid dimensions must be divisible by target dimensions")
    x_factor, y_factor = source_x // target_x, source_y // target_y
    return grid.reshape(target_x, x_factor, target_y, y_factor).mean(axis=(1, 3)).reshape(-1)
