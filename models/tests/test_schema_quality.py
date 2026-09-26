import pytest

from zcpv.quality import coordinate_round_trip
from zcpv.schema import schema_manifest


def test_coordinate_transform_round_trip():
    assert coordinate_round_trip(17.2, 61.3, 105, 68) == pytest.approx((17.2, 61.3))


def test_canonical_tables_are_versioned():
    manifest = schema_manifest()
    assert manifest["schema_version"] == "1.0.0"
    assert {"matches", "events", "tracking", "lineups", "state_samples"} <= manifest["tables"].keys()
