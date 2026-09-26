import numpy as np

from zcpv.models.impact import ImpactRow, fit_impact_model


def fixture_rows(duplicate=False):
    a = tuple(f"A{i}" for i in range(11)); b = tuple(f"B{i}" for i in range(11))
    pa = {p: (1.0 if p == "A0" else 0.0,) for p in a}
    pb = {p: (-1.0 if p == "B0" else 0.0,) for p in b}
    rows = []
    for i in range(4):
        duration = 450 if duplicate else 900
        repeats = 2 if duplicate else 1
        for _ in range(repeats):
            rows.extend([
                ImpactRow(f"m{i}", duration, 0.30 / repeats, a, b, pa, pb),
                ImpactRow(f"m{i}", duration, 0.10 / repeats, b, a, pb, pa),
            ])
    return rows


def test_impact_fit_exports_offense_defense_and_net():
    fit = fit_impact_model(fixture_rows(), ["progression"], beta_l2=1, player_l2=10)
    assert fit.status == "available"
    offense, defense, net = fit.player_effect("A0", (1.0,))
    assert abs(offense + defense - net) < 1e-12
    unseen_offense, unseen_defense, _ = fit.player_effect("unseen", None)
    assert unseen_offense == 0 and unseen_defense == 0


def test_subdividing_only_one_segment_preserves_effects_and_predictions():
    rows = fixture_rows()
    first = rows[0]
    halves = [
        ImpactRow(first.match_id, first.duration_seconds / 2, first.npxg / 2, first.own_players, first.opponent_players, first.own_profiles, first.opponent_profiles, first.context),
        ImpactRow(first.match_id, first.duration_seconds / 2, first.npxg / 2, first.own_players, first.opponent_players, first.own_profiles, first.opponent_profiles, first.context),
    ]
    whole = fit_impact_model(rows, ["progression"], beta_l2=1, player_l2=10)
    split = fit_impact_model(halves + rows[1:], ["progression"], beta_l2=1, player_l2=10)
    for player in whole.player_ids:
        profile = whole.reference_profiles[player]
        assert np.allclose(whole.player_effect(player, profile), split.player_effect(player, profile), atol=1e-10)
    assert np.allclose([whole.predict_row(row) for row in rows], [split.predict_row(row) for row in rows], atol=1e-10)


def test_complete_effect_centering_is_zero_and_preserves_predictions():
    rows = fixture_rows()
    fit = fit_impact_model(rows, ["progression"], beta_l2=1, player_l2=10)
    offense, defense = [], []
    for player in fit.player_ids:
        effect = fit.player_effect(player, fit.reference_profiles[player])
        offense.append(effect[0]); defense.append(effect[1])
    assert abs(np.average(offense, weights=[fit.offensive_exposure[p] for p in fit.player_ids])) < 1e-12
    assert abs(np.average(defense, weights=[fit.defensive_exposure[p] for p in fit.player_ids])) < 1e-12
    for row in rows:
        raw = fit.uncentered_intercept
        for player in row.own_players:
            vector = (np.asarray(row.own_profiles[player]) - fit.profile_mean) / fit.profile_scale
            raw += float(vector @ fit.offensive_profile_coef + fit.offensive_residuals[player])
        for player in row.opponent_players:
            vector = (np.asarray(row.opponent_profiles[player]) - fit.profile_mean) / fit.profile_scale
            raw -= float(vector @ fit.defensive_profile_coef + fit.defensive_residuals[player])
        assert abs(raw - fit.predict_row(row)) < 1e-10
