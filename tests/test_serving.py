"""Serving smoke tests. Skipped automatically where serving deps are absent."""
import pytest

pytest.importorskip("fastapi")


def test_feature_vector_length_matches_schema():
    from driveguard.features.rolling import feature_columns
    from driveguard.serving.app import DayReading, PredictRequest, _feature_vector
    req = PredictRequest(model="ST12000NM0007", capacity_bytes=12_000_000_000_000,
                         drive_age_days=300,
                         history=[DayReading(smart_5_raw=float(i)) for i in range(30)])
    x = _feature_vector(req, {"model_code_map": {}})
    assert len(x) == len(feature_columns())
    assert not (x != x).any()  # no NaNs


def test_health_ok():
    from fastapi.testclient import TestClient

    from driveguard.serving.app import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
