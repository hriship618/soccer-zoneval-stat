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


def test_duplicate_identical_rate_segments_preserve_fit():
    whole = fit_impact_model(fixture_rows(), ["progression"], beta_l2=1, player_l2=10)
    split = fit_impact_model(fixture_rows(duplicate=True), ["progression"], beta_l2=1, player_l2=10)
    assert abs(whole.intercept - split.intercept) < 1e-9
