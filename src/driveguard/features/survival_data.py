"""Milestone 4 - build the survival / RUL modelling tables.

Same rolling features as the classifier, but the target is the survival pair:
  - duration : rul_time_days = days from this drive-day to the drive's end
  - event    : 1 if that end is a failure, 0 if right-censored (still alive / left fleet)

Each drive-day is treated as a right-censored observation. We keep ALL failed-drive rows
and a hash-sampled fraction of censored rows (config ratio), so the tables are small enough
for classical survival models while retaining every event.

Reuses rolling.py for the feature engineering (look-back-bounded per split).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from driveguard.features.build_features import _QUARTER_RANGE
from driveguard.features.rolling import (
    MAX_LOOKBACK, SMART_BIG5, _rolling_exprs, feature_columns,
)


def _surv_split_lazy(interim_glob: str, summary_path: str, quarters: list[str]) -> pl.LazyFrame:
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    lo_d, hi_d = pl.lit(lo).str.to_date(), pl.lit(hi).str.to_date()
    lookback = pl.lit(lo).str.to_date() - pl.duration(days=MAX_LOOKBACK.days)

    ds = pl.read_parquet(summary_path).select("serial_number", "first_date", "last_date", "event")
    return (
        pl.scan_parquet(interim_glob)
        .select(["date", "serial_number", "model", "capacity_bytes", *SMART_BIG5])
        .filter(pl.col("date").is_between(lookback, hi_d))
        .with_columns([pl.col(c).cast(pl.Float32) for c in SMART_BIG5])
        .sort("serial_number", "date")
        .with_columns([
            pl.col(c).forward_fill().over("serial_number").fill_null(0).alias(c)
            for c in SMART_BIG5
        ])
        .with_columns(_rolling_exprs(SMART_BIG5))
        .with_columns([
            (pl.col(c) - pl.col(f"{c}_rmean_30")).cast(pl.Float32).alias(f"{c}_dev30")
            for c in SMART_BIG5
        ])
        .join(ds.lazy(), on="serial_number", how="left")
        .with_columns(
            (pl.col("date") - pl.col("first_date")).dt.total_days().alias("drive_age_days"),
            (pl.col("capacity_bytes") / 1e9).alias("capacity_gb"),
            (pl.col("last_date") - pl.col("date")).dt.total_days().cast(pl.Int32).alias("duration"),
            pl.col("event").cast(pl.Int8),
        )
        .filter(pl.col("date").is_between(lo_d, hi_d))
        .filter(pl.col("duration") >= 0)
    )


def _sink_split(lf: pl.LazyFrame, out_path: Path, keep: list[str], ratio: float | None) -> dict:
    pos = lf.select((pl.col("event") == 1).sum()).collect(engine="streaming").item()
    neg = lf.select((pl.col("event") == 0).sum()).collect(engine="streaming").item()
    sel = lf.select([*keep, "event", "duration"])
    if ratio is not None and neg > ratio * pos:
        frac = (ratio * pos) / neg
        rnd = ((pl.col("serial_number").cast(pl.String) + pl.col("date").cast(pl.String))
               .hash(seed=42) % 1_000_000) / 1_000_000
        sel = sel.filter((pl.col("event") == 1) | (rnd < frac))
    sel.sink_parquet(out_path, compression="zstd")
    return {"out": str(out_path), "orig_events": int(pos), "orig_censored": int(neg)}


def make_survival_dataset(cfg: dict, project_root: Path) -> dict:
    interim = project_root / cfg["data"]["interim_dir"]
    processed = project_root / cfg["data"]["processed_dir"]
    interim_glob = str(interim / "*.parquet")
    summary_path = str(processed / "drive_summary.parquet")
    keep = ["serial_number", "date", *feature_columns()]
    out_dir = processed / "survival_rolling"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split, quarters, ratio in [
        ("train", cfg["split"]["train_quarters"], float(cfg["data"]["healthy_downsample_ratio"])),
        ("val", cfg["split"]["val_quarters"], 20.0),
        ("test", cfg["split"]["test_quarters"], 20.0),
    ]:
        lf = _surv_split_lazy(interim_glob, summary_path, quarters)
        results[split] = _sink_split(lf, out_dir / f"{split}.parquet", keep, ratio)
    results["feature_cols"] = feature_columns()
    results["out_dir"] = str(out_dir)
    return results


if __name__ == "__main__":
    import json

    from driveguard.config import PROJECT_ROOT, load_config

    print(json.dumps(make_survival_dataset(load_config(), PROJECT_ROOT), indent=2))
