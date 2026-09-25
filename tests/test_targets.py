from zcpv.schema import EventRecord
from zcpv.targets import Observation, label_states


def shot(team: str, second: float, xg: float = 0.2) -> EventRecord:
    return EventRecord(
        event_id=f"s-{team}-{second}", match_id="M", period=1, period_clock_s=second,
        source_timestamp="2022-01-01T00:00:00+00:00", team_id=team, actor_id="p",
        recipient_id=None, event_type="shot", outcome=None, start_x_m=90, start_y_m=34,
        end_x_m=None, end_y_m=None, penalty=False, xg=xg, xg_provenance="fixture",
        possession_id=None, live_play=True,
    )


def test_turnover_does_not_change_reference_team():
    observation = Observation("o", "M", 1, 10, "A", "B", True, {})
    result = label_states([observation], [shot("B", 20)], {("M", 1): 90})[0]
    assert result.reference_team_id == "A"
    assert result.reference_npxg_15s == 0
    assert result.opponent_npxg_15s == 0.2


def test_penalties_ignored_and_censored_windows_excluded():
    penalty = shot("A", 20)
    penalty = EventRecord(**{**penalty.__dict__, "penalty": True})
    first, second = label_states(
        [Observation("a", "M", 1, 10, "A", "B", True, {}), Observation("b", "M", 1, 80, "A", "B", True, {})],
        [penalty], {("M", 1): 90},
    )
    assert first.reference_npxg_15s == 0
    assert not second.eligible and second.exclusion_reason == "right_censored"
