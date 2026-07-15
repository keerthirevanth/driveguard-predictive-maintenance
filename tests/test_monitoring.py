"""Monitoring tests. Retrain-decision logic is pure-python (CI-safe); PSI needs numpy."""
import pytest

from driveguard.monitoring.drift import check_retrain


def test_retrain_fires_on_high_drift():
    d = {"drifted_share": 0.6, "n_drifted": 3, "n_features": 5}
    assert check_retrain(d)["retrain"] is True


def test_retrain_fires_on_perf_drop():
    d = {"drifted_share": 0.1, "n_drifted": 1, "n_features": 7}
    out = check_retrain(d, pr_auc_current=0.10, pr_auc_reference=0.164)
    assert out["retrain"] is True and out["reasons"]


def test_no_retrain_when_stable():
    d = {"drifted_share": 0.14, "n_drifted": 1, "n_features": 7}
    assert check_retrain(d, pr_auc_current=0.16, pr_auc_reference=0.164)["retrain"] is False


def test_psi_zero_for_identical():
    np = pytest.importorskip("numpy")
    x = np.random.default_rng(0).normal(size=5000)
    from driveguard.monitoring.drift import psi
    assert psi(x, x.copy()) < 0.01
