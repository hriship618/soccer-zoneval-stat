from zcpv.possession import PossessionFrame, segment_possessions
from zcpv.schema import PossessionState


def test_unknown_run_is_one_episode_not_one_per_frame():
    frames = [PossessionFrame(1, float(second), PossessionState.UNKNOWN) for second in range(4)]
    episodes = segment_possessions(frames)
    assert len(episodes) == 1
    assert episodes[0].start_s == 0 and episodes[0].end_s == 3


def test_contested_and_interruption_are_not_team_possessions():
    frames = [
        PossessionFrame(1, 0, PossessionState.HOME, "A"),
        PossessionFrame(1, 1, PossessionState.CONTESTED),
        PossessionFrame(1, 2, PossessionState.INTERRUPTION),
        PossessionFrame(1, 3, PossessionState.AWAY, "B"),
    ]
    episodes = segment_possessions(frames)
    assert [episode.state for episode in episodes] == [PossessionState.HOME, PossessionState.CONTESTED, PossessionState.INTERRUPTION, PossessionState.AWAY]
