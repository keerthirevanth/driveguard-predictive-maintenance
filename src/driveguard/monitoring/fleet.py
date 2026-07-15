"""Score a sample of the current fleet for the dashboard's fleet-health view.

Builds rolling features for a sample of drives from the newest quarter (memory-safe, reusing
finalize's drive-pruned frame), takes each drive's latest snapshot, and batch-scores it with
the production models -> failure probability, RUL, and alert level per drive.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl


def score_fleet(cfg: dict, root: Path, n_drives: int = 3000) -> pl.DataFrame:
    import lightgbm as lgb
    import pandas as pd

    from driveguard.models.finalize import _training_frame

    store = root / "models_store"
    meta = json.loads((store / "serving_meta.json").read_text())
    feats, code_map = meta["feature_cols"], meta["model_code_map"]
    booster = lgb.Booster(model_file=str(store / "classifier.txt"))
    aft = pickle.load(open(store / "rul_weibull.pkl", "rb"))
    cal = pickle.load(open(store / "calibrator.pkl", "rb")) if (store / "calibrator.pkl").exists() else None
    op = meta.get("operating_points", {})
    thr_warn = (op.get("fpr_1pct") or {}).get("threshold", 1.0)
    thr_crit = (op.get("fpr_0.1pct") or {}).get("threshold", 1.0)

    df = _training_frame(cfg, root, quarters=cfg["split"]["test_quarters"], n_healthy=n_drives)
    latest = df.sort("date").group_by("serial_number").last()  # newest snapshot per drive
    enc = latest.with_columns(
        pl.col("model").replace_strict(code_map, default=-1).cast(pl.Int32).alias("_code")
    ).with_columns([pl.col(c).fill_null(0.0) for c in feats if c != "model"])

    cols = []
    for c in feats:
        cols.append(pl.col("_code") if c == "model" else pl.col(c))
    X = enc.select(cols).to_numpy().astype(np.float32)

    raw = booster.predict(X)
    prob = cal.predict(raw) if cal is not None else raw
    rul = np.clip(aft.predict_median(
        pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])).to_numpy(), 1, 400)
    alert = np.where(raw >= thr_crit, "critical",
                     np.where(raw >= thr_warn, "warning",
                              np.where(rul < 60, "watch", "ok")))

    return latest.select(
        "serial_number", "model",
        pl.col("capacity_gb").round(0),
        pl.col("drive_age_days"),
    ).with_columns(
        failure_probability=pl.Series(np.round(prob, 4)),
        rul_days=pl.Series(np.round(rul, 0)),
        alert_level=pl.Series(alert),
    ).sort("failure_probability", descending=True)


if __name__ == "__main__":
    from driveguard.config import PROJECT_ROOT, load_config

    fleet = score_fleet(load_config(), PROJECT_ROOT, n_drives=2000)
    print("scored", fleet.height, "drives")
    print(fleet["alert_level"].value_counts())
    print(fleet.head(5))
