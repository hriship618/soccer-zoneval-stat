import json

from zcpv.statsbomb import SBAction, load_match, split_match_ids_three_way, state_features


def action(index: int, period: int, *, goal: bool = False, team: int = 1, kind: str = "Pass") -> SBAction:
    return SBAction(1, index, period, 0, index, 1, team, str(team), index, str(index), kind, 10, 10, 20, 20, True, goal, 0.2 if kind == "Shot" else 0.0, 0, 0)


def test_post_action_label_excludes_current_shot_and_period_crossing():
    actions = [action(0, 1, goal=True, kind="Shot"), action(1, 1)]
    actions += [action(i, 1) for i in range(2, 11)]
    actions += [action(11, 2, goal=True, kind="Shot")] + [action(i, 2) for i in range(12, 23)]
    _, scores, _, ordered = state_features(actions, horizon=10, lags=2)
    assert ordered[0].index == 0
    assert scores[0] == 0  # the completed current shot is not a future outcome
    first_half_last_eligible = next(i for i, item in enumerate(ordered) if item.index == 0)
    assert scores[first_half_last_eligible] == 0  # second-half goal cannot label period one


def test_shootout_events_are_excluded(tmp_path):
    payload = [
        {"index": 0, "period": 1, "minute": 1, "second": 0, "possession": 1, "location": [10, 10], "team": {"id": 1, "name": "A"}, "player": {"id": 1, "name": "P"}, "type": {"name": "Pass"}, "pass": {"end_location": [20, 20]}},
        {"index": 1, "period": 5, "minute": 120, "second": 0, "possession": 2, "location": [100, 40], "team": {"id": 2, "name": "B"}, "player": {"id": 2, "name": "Q"}, "type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}, "statsbomb_xg": 0.76}},
    ]
    path = tmp_path / "123.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_match(path)
    assert len(loaded) == 1 and loaded[0].period == 1


def test_three_way_match_splits_are_disjoint():
    matches = [{"match_id": i, "match_date": f"2022-01-{i:02d}", "kick_off": "12:00"} for i in range(1, 11)]
    train, validation, test, _, _ = split_match_ids_three_way(matches)
    assert train and validation and test
    assert not (train & validation or train & test or validation & test)
