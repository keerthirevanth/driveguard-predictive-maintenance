"""Milestone 5 - train and save the production models locally.

Trains the two models the API serves, on rolling features, and writes artifacts to
models_store/:
  - classifier.txt        LightGBM failure classifier (tuned config from best_params.json)
  - rul_weibull.pkl       Weibull AFT remaining-useful-life model (emits days)
  - serving_meta.json     feature order, model->code map, SMART cols, window (for inference)

Memory-safe: we keep all failing drives + a sample of healthy drives and prune to them with
a streaming inner-join BEFORE computing rolling features, so this runs on a low-RAM machine.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from driveguard.features.build_features import BASE_FEATURES, SMART_BIG5, _QUARTER_RANGE
from driveguard.features.rolling import MAX_LOOKBACK, _rolling_exprs, feature_columns

CAT = "model"
HORIZON = 30


def _training_frame(cfg: dict, root: Path, quarters: list[str] | None = None,
                    n_healthy: int = 10_000, seed: int = 42) -> pl.DataFrame:
    interim_glob = str(root / cfg["data"]["interim_dir"] / "*.parquet")
    summary_path = str(root / cfg["data"]["processed_dir"] / "drive_summary.parquet")
    quarters = quarters or cfg["split"]["train_quarters"]
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    lo_d, hi_d = pl.lit(lo).str.to_date(), pl.lit(hi).str.to_date()
    lookback = pl.lit(lo).str.to_date() - pl.duration(days=MAX_LOOKBACK.days)

    summ = pl.read_parquet(summary_path).select(
        "serial_number", "first_date", "last_date", "event")
    active = summ.filter((pl.col("first_date") <= hi_d) & (pl.col("last_date") >= lo_d))
    ev = active.filter(pl.col("event") == 1)
    cen = active.filter(pl.col("event") == 0)
    if cen.height > n_healthy:
        cen = cen.sample(n=n_healthy, seed=seed)
    keep = pl.concat([ev, cen]).select("serial_number").lazy()

    df = (
        pl.scan_parquet(interim_glob)
        .select(["date", "serial_number", CAT, "capacity_bytes", *SMART_BIG5])
        .filter(pl.col("date").is_between(lookback, hi_d))
        .join(keep, on="serial_number", how="inner")            # prune first -> low memory
        .with_columns([pl.col(c).cast(pl.Float32) for c in SMART_BIG5])
        .sort("serial_number", "date")
        .with_columns([pl.col(c).forward_fill().over("serial_number").fill_null(0).alias(c)
                       for c in SMART_BIG5])
        .with_columns(_rolling_exprs(SMART_BIG5))
        .with_columns([(pl.col(c) - pl.col(f"{c}_rmean_30")).cast(pl.Float32).alias(f"{c}_dev30")
                       for c in SMART_BIG5])
        .join(summ.lazy(), on="serial_number", how="left")
        .with_columns(
            (pl.col("date") - pl.col("first_date")).dt.total_days().alias("drive_age_days"),
            (pl.col("capacity_bytes") / 1e9).alias("capacity_gb"),
            (pl.col("last_date") - pl.col("date")).dt.total_days().alias("rul_time_days"),
        )
        .filter(pl.col("date").is_between(lo_d, hi_d))
        .with_columns(
            pl.when((pl.col("event") == 1) & (pl.col("rul_time_days") <= HORIZON)).then(1)
            .when(pl.col("rul_time_days") >= HORIZON).then(0).otherwise(None).alias("y")
        )
        .collect(engine="streaming")
    )
    return df


def run(cfg: dict, root: Path) -> dict:
    feats = feature_columns()                       # model, capacity, age, big5, rolling...
    best = json.load(open(root / "config" / "best_params.json"))["params"]["lightgbm"]
    df = _training_frame(cfg, root)

    code_map = {m: i for i, m in enumerate(sorted(df[CAT].unique().drop_nulls().to_list()))}
    enc = df.with_columns(pl.col(CAT).replace_strict(code_map, default=-1).cast(pl.Int32))
    # rolling std is null for a drive's first day in a window; fill 0 to match serving
    # (numpy std of one value = 0) and because lifelines rejects NaNs.
    enc = enc.with_columns([pl.col(c).fill_null(0.0) for c in feats if c != CAT])

    # --- classifier (drop censored rows) ---
    clf_df = enc.filter(pl.col("y").is_not_null())
    Xc = clf_df.select(feats).to_numpy().astype(np.float32)
    yc = clf_df["y"].to_numpy().astype(np.int8)
    spw = float((yc == 0).sum() / max(yc.sum(), 1))
    clf = lgb.LGBMClassifier(scale_pos_weight=spw, n_jobs=-1, random_state=42,
                             verbose=-1, **best)
    clf.fit(Xc, yc, categorical_feature=[0])

    # --- Weibull RUL (all rows; duration + event) ---
    from lifelines import WeibullAFTFitter
    surv = enc.select([*feats, "event",
                       pl.col("rul_time_days").clip(1).alias("duration")]).to_pandas()
    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(surv.rename(columns={c: f"f{i}" for i, c in enumerate(feats)}),
            duration_col="duration", event_col="event")

    # --- calibration + operating thresholds on the VALIDATION quarter (never test) ---
    # scale_pos_weight inflates raw scores, so: (a) isotonic-calibrate them into honest
    # probabilities, (b) derive alert thresholds from validation FPR budgets instead of
    # a meaningless 0.5 cut.
    from sklearn.isotonic import IsotonicRegression

    from driveguard.evaluation.metrics import recall_at_fpr

    vdf = _training_frame(cfg, root, quarters=cfg["split"]["val_quarters"], n_healthy=8_000)
    venc = vdf.with_columns(pl.col(CAT).replace_strict(code_map, default=-1).cast(pl.Int32))
    venc = venc.with_columns([pl.col(c).fill_null(0.0) for c in feats if c != CAT])
    venc = venc.filter(pl.col("y").is_not_null())
    Xv = venc.select(feats).to_numpy().astype(np.float32)
    yv = venc["y"].to_numpy().astype(np.int8)
    sv = clf.predict_proba(Xv)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(sv, yv)
    op = {"fpr_1pct": recall_at_fpr(yv, sv, 0.01),
          "fpr_0.1pct": recall_at_fpr(yv, sv, 0.001)}

    store = root / "models_store"
    store.mkdir(exist_ok=True)
    clf.booster_.save_model(str(store / "classifier.txt"))
    with open(store / "rul_weibull.pkl", "wb") as f:
        pickle.dump(aft, f)
    with open(store / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    meta = {"feature_cols": feats, "model_code_map": code_map, "cat_index": 0,
            "smart_cols": SMART_BIG5, "base_features": BASE_FEATURES, "horizon": HORIZON,
            "windows": [7, 14, 30], "n_train_rows": int(df.height),
            "class_pos": int(yc.sum()), "class_neg": int((yc == 0).sum()),
            "operating_points": op,
            "calibration": "isotonic, fit on validation quarter "
                           + ",".join(cfg["split"]["val_quarters"]),
            "val_pos": int(yv.sum()), "val_neg": int((yv == 0).sum())}
    (store / "serving_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    from driveguard.config import PROJECT_ROOT, load_config

    m = run(load_config(), PROJECT_ROOT)
    print(json.dumps({k: v for k, v in m.items() if k != "model_code_map"}, indent=2))
    print("saved to models_store/: classifier.txt, rul_weibull.pkl, serving_meta.json")
