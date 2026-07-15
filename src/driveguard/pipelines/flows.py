"""Phase 4 - orchestration: the closed MLOps loop.

  monitor drift  ->  decide  ->  (if triggered) retrain + re-save models

Uses Prefect decorators when Prefect is installed (real flow with a UI / scheduling), and
falls back to plain function calls otherwise, so the loop always runs. Schedule the flow
(cron / Prefect deployment / Airflow) to close the loop automatically.
"""
from __future__ import annotations

from pathlib import Path

from driveguard.config import PROJECT_ROOT, load_config
from driveguard.models import finalize
from driveguard.monitoring import drift

try:  # real orchestration if available; no-op decorators otherwise
    from prefect import flow, task
except Exception:  # pragma: no cover
    def task(fn=None, **_):
        return fn if fn else (lambda f: f)

    def flow(fn=None, **_):
        return fn if fn else (lambda f: f)


@task
def monitor(cfg: dict, root: Path) -> dict:
    return drift.run(cfg, root)


@task
def retrain(cfg: dict, root: Path) -> dict:
    return finalize.run(cfg, root)


@flow(name="driveguard-mlops-loop")
def mlops_loop(retrain_if_needed: bool = True) -> dict:
    cfg, root = load_config(), PROJECT_ROOT
    report = monitor(cfg, root)
    decision = report["decision"]
    retrained = False
    if decision["retrain"] and retrain_if_needed:
        retrain(cfg, root)
        retrained = True
    return {"retrain_triggered": decision["retrain"], "retrained": retrained,
            "reasons": decision["reasons"],
            "drift_share": report["drift"]["drifted_share"],
            "n_drifted": report["drift"]["n_drifted"]}


if __name__ == "__main__":
    import json

    print(json.dumps(mlops_loop(retrain_if_needed=False), indent=2))
