"""Smoke test: config loads and has the expected top-level sections."""
from driveguard.config import load_config


def test_config_loads_with_expected_sections():
    cfg = load_config()
    for section in ["data", "labels", "split", "features", "models", "evaluation", "mlops"]:
        assert section in cfg, f"missing config section: {section}"


def test_time_split_is_used():
    cfg = load_config()
    assert cfg["split"]["strategy"] == "time", "must use time-based split to avoid leakage"
