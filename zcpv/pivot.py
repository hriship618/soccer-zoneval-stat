from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .adapters.dfl import DFLAdapter
from .dfl import read_metadata, read_tracking
from .pitch_control import PitchControlConfig, pitch_control_counterfactuals
from .statsbomb import SBAction, event_model_values, shot_features, state_features, xt_action_value


DFL_ACTION_TYPES = {
    "pass": "Pass",
    "carry": "Carry",
    "shot": "Shot",
    "tackle": "Duel",
    "recovery": "Ball Recovery",
}


@dataclass
class DFLEventState:
    match_id: str
    action_index: int
    event_id: str
    player_id: str
    team_id: str
    team_index: int
    period: int
    timestamp: float
    event_value: float
    xt_value: float
    score_probability: float
    concede_probability: float
    control_advantage: float
    player_losses: dict[str, tuple[int, float]]
    score_label: int | None
    concede_label: int | None


@dataclass
class DFLMatchResult:
    match_id: str
    metadata: dict[str, Any]
    events: list[DFLEventState]
    players: list[dict[str, Any]]
    actions: list[SBAction]
    quality: dict[str, Any]


def _successful(outcome: str | None) -> bool:
    normalized = (outcome or "").lower()
    return normalized not in {"unsuccessful", "failed", "lost"} and "unsuccess" not in normalized


def dfl_common_actions(match_dir: Path) -> tuple[list[SBAction], dict[int, Any], DFLAdapter]:
    """Map provider events into the exact World Cup event feature space."""
    adapter = DFLAdapter(match_dir)
    match = adapter.match()
    events, _ = adapter.events()
    directions, _ = adapter.attacking_right()
    actions: list[SBAction] = []
    source_by_index: dict[int, Any] = {}
    scores = {match.home_team_id: 0, match.away_team_id: 0}
    possession = 0
    previous_team: str | None = None
    for event in sorted(events, key=lambda row: (row.period, row.period_clock_s, row.event_id)):
        action_type = DFL_ACTION_TYPES.get(event.event_type)
        if action_type is None or event.actor_id is None or event.team_id is None or event.start_x_m is None:
            continue
        if event.team_id != previous_team:
            possession += 1
            previous_team = event.team_id
        right = directions.get((event.team_id, event.period), True)
        start_x = event.start_x_m if right else match.pitch_length_m - event.start_x_m
        start_y = event.start_y_m if right else match.pitch_width_m - event.start_y_m
        end_x_raw = event.end_x_m if event.end_x_m is not None else event.start_x_m
        end_y_raw = event.end_y_m if event.end_y_m is not None else event.start_y_m
        end_x = end_x_raw if right else match.pitch_length_m - end_x_raw
        end_y = end_y_raw if right else match.pitch_width_m - end_y_raw
        opponent = match.away_team_id if event.team_id == match.home_team_id else match.home_team_id
        goal = event.event_type == "shot" and event.outcome == "goal"
        index = len(actions)
        actions.append(SBAction(
            match_id=match.match_id,
            index=index,
            period=event.period,
            minute=int((event.period - 1) * 45 + event.period_clock_s // 60),
            second=int(event.period_clock_s % 60),
            possession=possession,
            team_id=event.team_id,
            team_name=match.home_team_name if event.team_id == match.home_team_id else match.away_team_name,
            player_id=event.actor_id,
            player_name=adapter.players().get(event.actor_id, {}).get("name", event.actor_id),
            action_type=action_type,
            start_x=float(np.clip(start_x / match.pitch_length_m * 120.0, 0.0, 120.0)),
            start_y=float(np.clip(start_y / match.pitch_width_m * 80.0, 0.0, 80.0)),
            end_x=float(np.clip(end_x / match.pitch_length_m * 120.0, 0.0, 120.0)),
            end_y=float(np.clip(end_y / match.pitch_width_m * 80.0, 0.0, 80.0)),
            successful=_successful(event.outcome),
            goal=goal,
            shot_xg=0.0,
            score_for=scores[event.team_id],
            score_against=scores[opponent],
        ))
        source_by_index[index] = event
        if goal:
            scores[event.team_id] += 1
    return actions, source_by_index, adapter


def _absolute_threat_surface(zone_values: np.ndarray, attacks_right: bool) -> np.ndarray:
    output = np.zeros(12, dtype=np.float32)
    for absolute in range(12):
        x_band, y_band = divmod(absolute, 3)
        relative = absolute if attacks_right else (3 - x_band) * 3 + (2 - y_band)
        output[absolute] = zone_values[relative]
    return output


def _nearest_frame(
    timestamps: np.ndarray,
    segments: np.ndarray,
    live: np.ndarray,
    segment: str,
    target: float,
    tolerance_s: float,
) -> int | None:
    candidates = np.flatnonzero((segments == segment) & live)
    if not len(candidates):
        return None
    local_times = timestamps[candidates]
    offset = int(np.searchsorted(local_times, target))
    choices = [value for value in (offset - 1, offset) if 0 <= value < len(candidates)]
    chosen = min(choices, key=lambda value: abs(local_times[value] - target))
    frame = int(candidates[chosen])
    return frame if abs(timestamps[frame] - target) <= tolerance_s else None


def process_dfl_match(
    match_dir: Path,
    score_model,
    concede_model,
    shot_xg_model,
    xt_grid: np.ndarray,
    zone_values: np.ndarray,
    *,
    tracking_hz: int = 5,
    chunk_size: int = 256,
) -> DFLMatchResult:
    actions, source_by_index, adapter = dfl_common_actions(match_dir)
    shot_indices = [index for index, action in enumerate(actions) if action.action_type == "Shot"]
    if shot_indices:
        predicted_xg = shot_xg_model.predict(shot_features([actions[index] for index in shot_indices]))
        for index, xg in zip(shot_indices, predicted_xg):
            actions[index] = replace(actions[index], shot_xg=float(np.clip(xg, 0.001, 0.999)))
    scored = event_model_values(actions, score_model, concede_model)
    _, score_labels, concede_labels, labeled_actions = state_features(actions)
    labels = {
        action.index: (int(score_labels[index]), int(concede_labels[index]))
        for index, action in enumerate(labeled_actions)
    }
    metadata = read_metadata(match_dir / "match.xml")
    (
        positions, velocities, teams, active, live, possessions, segments,
        frame_numbers, timestamps, source_hz,
    ) = read_tracking(match_dir / "positions.xml", metadata, tracking_hz)
    player_ids = [player.id for player in metadata["players"]]
    team_index_by_id = {player.team_id: player.team_index for player in metadata["players"]}
    directions, _ = adapter.attacking_right()
    aligned: list[dict[str, Any]] = []
    alignment_errors = 0
    for result in scored:
        action = result["action"]
        source = source_by_index[action.index]
        segment = {1: "firstHalf", 2: "secondHalf", 3: "firstHalfExtraTime", 4: "secondHalfExtraTime"}.get(action.period)
        if segment is None:
            continue
        target = datetime.fromisoformat(source.source_timestamp).timestamp()
        frame = _nearest_frame(timestamps, segments, live, segment, target, max(0.6, 1.5 / tracking_hz))
        if frame is None or action.team_id not in team_index_by_id:
            alignment_errors += 1
            continue
        aligned.append({
            "result": result,
            "source": source,
            "frame": frame,
            "team_index": team_index_by_id[action.team_id],
            "attacks_right": directions.get((action.team_id, action.period), True),
        })

    event_states: list[DFLEventState | None] = [None] * len(aligned)
    groups: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for index, row in enumerate(aligned):
        groups[(row["team_index"], row["attacks_right"])].append(index)
    config = PitchControlConfig(grid_x=32, grid_y=21)
    for (actor_team, attacks_right), event_indices in groups.items():
        surface = _absolute_threat_surface(zone_values, attacks_right)
        for start in range(0, len(event_indices), chunk_size):
            batch_indices = event_indices[start:start + chunk_size]
            frames = np.asarray([aligned[index]["frame"] for index in batch_indices], dtype=int)
            full, losses = pitch_control_counterfactuals(
                positions[frames], velocities[frames], teams, surface, config,
            )
            for local, event_index in enumerate(batch_indices):
                row = aligned[event_index]
                result = row["result"]
                action = result["action"]
                opponent_team = 1 - actor_team
                control_advantage = float(full[local, actor_team].sum() - full[local, opponent_team].sum())
                player_losses = {
                    player_id: (int(teams[player]), float(losses[local, player].sum()))
                    for player, player_id in enumerate(player_ids)
                    if active[row["frame"], player]
                }
                label = labels.get(action.index)
                event_states[event_index] = DFLEventState(
                    match_id=metadata["match_id"],
                    action_index=action.index,
                    event_id=row["source"].event_id,
                    player_id=str(action.player_id),
                    team_id=str(action.team_id),
                    team_index=actor_team,
                    period=action.period,
                    timestamp=float(row["source"].period_clock_s),
                    event_value=float(result["event_value"]),
                    xt_value=float(xt_action_value(action, xt_grid)),
                    score_probability=float(result["score_probability"]),
                    concede_probability=float(result["concede_probability"]),
                    control_advantage=control_advantage,
                    player_losses=player_losses,
                    score_label=None if label is None else label[0],
                    concede_label=None if label is None else label[1],
                )

    minutes = active.sum(axis=0) / tracking_hz / 60.0
    players = []
    for index, player in enumerate(metadata["players"]):
        players.append({
            "player_id": player.id,
            "name": player.name,
            "team_id": player.team_id,
            "team": player.team_name,
            "team_index": player.team_index,
            "role": player.role,
            "number": player.number,
            "minutes": float(minutes[index]),
        })
    quality = {
        "common_actions": len(actions),
        "aligned_events": len(aligned),
        "alignment_failures": alignment_errors,
        "alignment_rate": len(aligned) / max(1, len(scored)),
        "tracking_hz": tracking_hz,
        "source_hz": source_hz,
        "tracking_frames": len(positions),
        "live_frames": int(live.sum()),
        "labeled_events": sum(event.score_label is not None for event in event_states if event is not None),
    }
    return DFLMatchResult(
        metadata["match_id"], metadata,
        [event for event in event_states if event is not None], players, actions, quality,
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _fusion_features(
    events: list[DFLEventState],
    target: str,
    advantages: np.ndarray | None = None,
    *,
    include_control: bool = True,
) -> np.ndarray:
    probabilities = np.asarray([
        event.score_probability if target == "score" else event.concede_probability
        for event in events
    ])
    event_logit = _logit(probabilities).reshape(-1, 1)
    if not include_control:
        return event_logit
    control = np.asarray([event.control_advantage for event in events]) if advantages is None else advantages
    return np.column_stack((event_logit[:, 0], control))


def fit_fusion(events: list[DFLEventState], target: str, *, include_control: bool = True):
    label_name = f"{target}_label"
    eligible = [event for event in events if getattr(event, label_name) is not None]
    labels = np.asarray([getattr(event, label_name) for event in eligible], dtype=np.int8)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"DFL training fold has only one {target} class")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=1_000, random_state=42),
    )
    return model.fit(_fusion_features(eligible, target, include_control=include_control), labels)


def _control_coefficient(model) -> dict[str, float | str]:
    """Expose the control coefficient on standardized and original scales."""
    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["logisticregression"]
    standardized = float(estimator.coef_[0, 1])
    original = standardized / float(scaler.scale_[1])
    return {
        "standardized": standardized,
        "original_control_units": original,
        "sign": "positive" if original > 0 else "negative" if original < 0 else "zero",
    }


def _safe_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int | None]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    output: dict[str, float | int | None] = {
        "events": int(len(labels)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": None,
    }
    if len(np.unique(labels)) == 2:
        output["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return output


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def exposure_reliability(matches: int, minutes: float, prior_matches: float = 4.0) -> tuple[float, float]:
    """Conservative reliability from match-equivalent exposure.

    Correlated frames within a match cannot make that match worth more than one
    independent exposure. Partial matches count in proportion to 90 minutes,
    and a four-match prior keeps one-match estimates strongly shrunk.
    """
    if matches < 0 or minutes < 0 or prior_matches <= 0:
        raise ValueError("matches/minutes must be non-negative and prior_matches positive")
    effective_matches = min(float(matches), minutes / 90.0)
    return effective_matches, effective_matches / (effective_matches + prior_matches)


def leave_one_match_out(matches: list[DFLMatchResult]) -> tuple[list[dict], dict, dict]:
    """Cross-fit fusion and return leakage-safe player-event contributions."""
    contributions: list[dict] = []
    evaluation_rows: list[dict] = []
    fold_details = []
    for held_match in matches:
        train_events = [event for match in matches if match.match_id != held_match.match_id for event in match.events]
        score_model = fit_fusion(train_events, "score", include_control=True)
        concede_model = fit_fusion(train_events, "concede", include_control=True)
        score_event_only = fit_fusion(train_events, "score", include_control=False)
        concede_event_only = fit_fusion(train_events, "concede", include_control=False)
        held_events = held_match.events
        score_full = score_model.predict_proba(_fusion_features(held_events, "score"))[:, 1]
        concede_full = concede_model.predict_proba(_fusion_features(held_events, "concede"))[:, 1]
        score_event = score_event_only.predict_proba(
            _fusion_features(held_events, "score", include_control=False)
        )[:, 1]
        concede_event = concede_event_only.predict_proba(
            _fusion_features(held_events, "concede", include_control=False)
        )[:, 1]
        train_labeled = [event for event in train_events if event.score_label is not None]
        score_prevalence = float(np.mean([event.score_label for event in train_labeled]))
        concede_prevalence = float(np.mean([event.concede_label for event in train_labeled]))
        fold_details.append({
            "held_out_match": held_match.match_id,
            "training_matches": len(matches) - 1,
            "training_events": len(train_labeled),
            "held_out_events": sum(event.score_label is not None for event in held_events),
            "control_coefficients": {
                "score": _control_coefficient(score_model),
                "concede": _control_coefficient(concede_model),
            },
        })
        for event_index, event in enumerate(held_events):
            if event.score_label is not None:
                evaluation_rows.append({
                    "match_id": event.match_id,
                    "score_label": event.score_label,
                    "concede_label": event.concede_label,
                    "pivot_score": float(score_full[event_index]),
                    "pivot_concede": float(concede_full[event_index]),
                    "event_score": float(score_event[event_index]),
                    "event_concede": float(concede_event[event_index]),
                    "constant_score": score_prevalence,
                    "constant_concede": concede_prevalence,
                })
            player_ids = list(event.player_losses)
            if not player_ids:
                continue
            advantages = np.asarray([
                event.control_advantage - loss if team_index == event.team_index else event.control_advantage + loss
                for team_index, loss in event.player_losses.values()
            ])
            score_without = score_model.predict_proba(_fusion_features([event] * len(player_ids), "score", advantages))[:, 1]
            concede_without = concede_model.predict_proba(_fusion_features([event] * len(player_ids), "concede", advantages))[:, 1]
            full_net = score_full[event_index] - concede_full[event_index]
            without_net = score_without - concede_without
            for player_offset, player_id in enumerate(player_ids):
                player_team, raw_loss = event.player_losses[player_id]
                actor_frame_effect = float(full_net - without_net[player_offset])
                spatial = actor_frame_effect if player_team == event.team_index else -actor_frame_effect
                on_ball = event.event_value if player_id == event.player_id else 0.0
                xt = event.xt_value if player_id == event.player_id else 0.0
                contributions.append({
                    "match_id": event.match_id,
                    "event_id": event.event_id,
                    "player_id": player_id,
                    "team_index": player_team,
                    "actor": player_id == event.player_id,
                    "event_value": on_ball,
                    "xt_value": xt,
                    "spatial_counterfactual": spatial,
                    "raw_control_loss": raw_loss,
                    "pivot_contribution": on_ball + spatial,
                })

    labels_score = np.asarray([row["score_label"] for row in evaluation_rows])
    labels_concede = np.asarray([row["concede_label"] for row in evaluation_rows])
    coefficient_stability = {}
    for target in ("score", "concede"):
        values = np.asarray([
            fold["control_coefficients"][target]["original_control_units"]
            for fold in fold_details
        ])
        coefficient_stability[target] = {
            "positive_folds": int((values > 0).sum()),
            "negative_folds": int((values < 0).sum()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "stable_sign": bool((values > 0).all() or (values < 0).all()),
        }
    evaluation = {
        "strategy": "leave-one-match-out cross-fitting across all seven DFL matches",
        "folds": fold_details,
        "control_coefficient_stability": coefficient_stability,
        "score": {
            "constant": _safe_metrics(labels_score, np.asarray([row["constant_score"] for row in evaluation_rows])),
            "event_model_alone": _safe_metrics(labels_score, np.asarray([row["event_score"] for row in evaluation_rows])),
            "event_plus_pitch_control": _safe_metrics(labels_score, np.asarray([row["pivot_score"] for row in evaluation_rows])),
        },
        "concede": {
            "constant": _safe_metrics(labels_concede, np.asarray([row["constant_concede"] for row in evaluation_rows])),
            "event_model_alone": _safe_metrics(labels_concede, np.asarray([row["event_concede"] for row in evaluation_rows])),
            "event_plus_pitch_control": _safe_metrics(labels_concede, np.asarray([row["pivot_concede"] for row in evaluation_rows])),
        },
    }
    return contributions, evaluation, {"folds": fold_details}


def aggregate_rankings(matches: list[DFLMatchResult], contributions: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    player_meta: dict[str, dict] = {}
    minutes: dict[str, float] = defaultdict(float)
    match_counts: dict[str, set[str]] = defaultdict(set)
    for match in matches:
        for player in match.players:
            player_meta[player["player_id"]] = player
            minutes[player["player_id"]] += player["minutes"]
            if player["minutes"] > 0:
                match_counts[player["player_id"]].add(match.match_id)
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    player_match: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in contributions:
        player_id = row["player_id"]
        totals[player_id]["samples"] += 1
        for name in ("event_value", "xt_value", "spatial_counterfactual", "pivot_contribution"):
            totals[player_id][name] += row[name]
            player_match[(player_id, row["match_id"])][name] += row[name]
        player_match[(player_id, row["match_id"])]["samples"] += 1

    eligible_ids = [
        player_id for player_id, values in totals.items()
        if minutes[player_id] >= 45 and len(match_counts[player_id]) >= 1 and values["samples"] > 0
    ]
    rating_group = {
        player_id: "goalkeeper" if player_meta[player_id]["role"] == "TW" else "outfield"
        for player_id in eligible_ids
    }
    shrunk: dict[str, float] = {}
    group_reference: dict[str, dict[str, float]] = {}
    for group in ("outfield", "goalkeeper"):
        members = [player_id for player_id in eligible_ids if rating_group[player_id] == group]
        total_contribution = sum(totals[player_id]["pivot_contribution"] for player_id in members)
        total_samples = sum(totals[player_id]["samples"] for player_id in members)
        global_per_100 = 100 * total_contribution / max(1, total_samples)
        raw_rates = {
            player_id: 100 * totals[player_id]["pivot_contribution"] / totals[player_id]["samples"]
            for player_id in members
        }
        for player_id in members:
            matches_played = len(match_counts[player_id])
            effective_matches, reliability = exposure_reliability(matches_played, minutes[player_id])
            shrunk[player_id] = reliability * raw_rates[player_id] + (1 - reliability) * global_per_100
        # Keep the reference scale fixed to the raw-rate pool. Standardizing on
        # the shrunken distribution would expand its compressed variance and
        # undo the uncertainty adjustment in the displayed rating.
        raw_scale = float(np.std(list(raw_rates.values()))) if len(raw_rates) > 1 else 1.0
        group_reference[group] = {
            "center": global_per_100,
            "scale": max(raw_scale, 1e-12),
            "prior": global_per_100,
        }
    rankings = []
    for player_id in eligible_ids:
        values = totals[player_id]
        meta = player_meta[player_id]
        player_minutes = minutes[player_id]
        sample_count = int(values["samples"])
        matches_played = len(match_counts[player_id])
        effective_matches, reliability = exposure_reliability(matches_played, player_minutes)
        reference = group_reference[rating_group[player_id]]
        rating = float(np.clip(50 + 10 * (shrunk[player_id] - reference["center"]) / reference["scale"], 0, 100))
        rankings.append({
            "player_id": player_id,
            "name": meta["name"],
            "team": meta["team"],
            "role": meta["role"],
            "matches": matches_played,
            "minutes": round(player_minutes, 1),
            "event_samples": sample_count,
            "pivot_rating": round(rating, 2),
            "rating_group": rating_group[player_id],
            "raw_pivot_per_100": round(100 * values["pivot_contribution"] / sample_count, 5),
            "shrunk_pivot_per_100": round(shrunk[player_id], 5),
            "event_value_per_90": round(90 * values["event_value"] / max(player_minutes, 1), 5),
            "spatial_value_per_100_events": round(100 * values["spatial_counterfactual"] / sample_count, 5),
            "xt_per_90": round(90 * values["xt_value"] / max(player_minutes, 1), 5),
            "effective_matches": round(effective_matches, 4),
            "reliability": round(reliability, 4),
        })
    rankings_by_group = {
        "outfield": [row for row in rankings if row["rating_group"] == "outfield"],
        "goalkeepers": [row for row in rankings if row["rating_group"] == "goalkeeper"],
    }
    for group in rankings_by_group.values():
        group.sort(key=lambda row: (-row["pivot_rating"], -row["minutes"], row["name"]))
        for rank, row in enumerate(group, 1):
            row["rank"] = rank

    goals_by_match_team: dict[tuple[str, int], int] = defaultdict(int)
    for match in matches:
        team_index_by_id = {player["team_id"]: player["team_index"] for player in match.players}
        for action in match.actions:
            if action.goal:
                goals_by_match_team[(match.match_id, team_index_by_id[str(action.team_id)])] += 1
    team_rows = []
    for match in matches:
        match_contributions = [row for row in contributions if row["match_id"] == match.match_id]
        for team_index in (0, 1):
            team_player_ids = {player["player_id"] for player in match.players if player["team_index"] == team_index}
            rows = [row for row in match_contributions if row["player_id"] in team_player_ids]
            raw = sum(row["event_value"] for row in rows)
            xt = sum(row["xt_value"] for row in rows)
            pivot = sum(row["pivot_contribution"] for row in rows)
            adjusted = sum(
                90 * player_match[(player_id, match.match_id)]["event_value"] /
                max(1.0, next(player["minutes"] for player in match.players if player["player_id"] == player_id))
                for player_id in team_player_ids if player_match[(player_id, match.match_id)]["samples"] > 0
            )
            goal_diff = goals_by_match_team[(match.match_id, team_index)] - goals_by_match_team[(match.match_id, 1 - team_index)]
            team_rows.append({"goal_diff": goal_diff, "raw": raw, "adjusted": adjusted, "xt": xt, "pivot": pivot})
    target = [row["goal_diff"] for row in team_rows]
    baseline_evaluation = {
        "unit": "14 team-match observations",
        "target": "non-shootout goal difference",
        "pearson_correlation": {
            "raw_on_ball_event_value": _pearson([row["raw"] for row in team_rows], target),
            "minutes_adjusted_event_value": _pearson([row["adjusted"] for row in team_rows], target),
            "statsbomb_xt": _pearson([row["xt"] for row in team_rows], target),
            "pivot": _pearson([row["pivot"] for row in team_rows], target),
        },
        "warning": "Descriptive only: seven matches and 14 paired team observations are too small for strong comparative claims.",
    }
    return rankings_by_group, baseline_evaluation
