from zcpv.profiles import PlayerMatchMeasurement, build_lagged_profiles


def test_future_matches_cannot_change_earlier_profile():
    early = PlayerMatchMeasurement("p", "m1", "2022-01-01", "MF", 90, {"progression": 1.0}, {"progression": 10})
    future = PlayerMatchMeasurement("p", "m2", "2022-03-01", "MF", 90, {"progression": 999.0}, {"progression": 10})
    one = build_lagged_profiles([early], "2022-02-01", ["progression"], shrinkage_kappa=1)
    two = build_lagged_profiles([early, future], "2022-02-01", ["progression"], shrinkage_kappa=1)
    assert one["p"].values == two["p"].values


def test_negative_measurements_are_preserved():
    row = PlayerMatchMeasurement("p", "m", "2022-01-01", "MF", 90, {"residual": -2.0}, {"residual": 20})
    profile = build_lagged_profiles([row], "2022-02-01", ["residual"], shrinkage_kappa=1)["p"]
    assert profile.values["residual"] < 0
