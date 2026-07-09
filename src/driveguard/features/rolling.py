"""Phase 1 (M3) - rolling / temporal feature engineering.

Adds per-drive trailing-window features on top of the big5 SMART values:
  for each SMART col: rolling mean & std over 7/14/30 days, plus deviation from the
  30-day mean (current - mean_30). These capture *degradation trajectory*, which a
  point-in-time snapshot misses - the expected lift over the Milestone 2 baseline.

Efficiency: rolling windows need look-back, but not the whole history. For each split we
scan only [split_start - max_window, split_end], compute rolling, then keep rows in the
split range. So each split processes ~one quarter instead of all 113M rows.

Outputs train/val/test Parquet under processed/features_rolling_N{horizon}/ with the same
contract as build_features (negative undersampling on train, natural dist on val/test).
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from driveguard.features.build_features import BASE_FEATURES, SMART_BIG5, _QUARTER_RANGE, build_split

WINDOWS = [7, 14, 30]
MAX_LOOKBACK = timedelta(days=max(WINDOWS))


def _rolling_exprs(cols: list[str]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for c in cols:
        for w in WINDOWS:
            exprs.append(
                pl.col(c).rolling_mean_by("date", window_size=f"{w}d")
                .over("serial_number").cast(pl.Float32).alias(f"{c}_rmean_{w}")
            )
            exprs.append(
                pl.col(c).rolling_std_by("date", window_size=f"{w}d")
                .over("serial_number").cast(pl.Float32).alias(f"{c}_rstd_{w}")
            )
    return exprs


def _split_lazy(interim_glob: str, summary_path: str, quarters: list[str], n: int) -> pl.LazyFrame:
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    lo_d = pl.lit(lo).str.to_date()
    hi_d = pl.lit(hi).str.to_date()
    lookback = (pl.lit(lo).str.to_date() - pl.duration(days=MAX_LOOKBACK.days))

    ds = pl.read_parquet(summary_path).select("serial_number", "first_date", "last_date", "event")
    lf = (
        pl.scan_parquet(interim_glob)
        .select(["date", "serial_number", "model", "capacity_bytes", *SMART_BIG5])
        .filter(pl.col("date").is_between(lookback, hi_d))  # quarter + look-back only
        .with_columns([pl.col(c).cast(pl.Float32) for c in SMART_BIG5])
        .sort("serial_number", "date")
        .with_columns(_rolling_exprs(SMART_BIG5))
        .with_columns([
            (pl.col(c) - pl.col(f"{c}_rmean_30")).cast(pl.Float32).alias(f"{c}_dev30")
            for c in SMART_BIG5
        ])
        .join(ds.lazy(), on="serial_number", how="left")
        .with_columns(
            (pl.col("date") - pl.col("first_date")).dt.total_days().alias("drive_age_days"),
            (pl.col("last_date") - pl.col("date")).dt.total_days().alias("rul_time_days"),
            (pl.col("capacity_bytes") / 1e9).alias("capacity_gb"),
        )
        .with_columns(
            pl.when((pl.col("event") == 1) & (pl.col("rul_time_days") <= n))
            .then(pl.lit(1, dtype=pl.Int8))
            .when(pl.col("rul_time_days") >= n)
            .then(pl.lit(0, dtype=pl.Int8))
            .otherwise(None)
            .alias("y")
        )
        .filter(pl.col("date").is_between(lo_d, hi_d))  # drop the look-back rows
        .drop_nulls("y")
    )
    return lf


def feature_columns() -> list[str]:
    roll = [f"{c}_rmean_{w}" for c in SMART_BIG5 for w in WINDOWS]
    roll += [f"{c}_rstd_{w}" for c in SMART_BIG5 for w in WINDOWS]
    roll += [f"{c}_dev30" for c in SMART_BIG5]
    return BASE_FEATURES + SMART_BIG5 + roll


def make_rolling_dataset(cfg: dict, project_root: Path, horizon: int) -> dict:
    interim = project_root / cfg["data"]["interim_dir"]
    processed = project_root / cfg["data"]["processed_dir"]
    interim_glob = str(interim / "*.parquet")
    summary_path = str(processed / "drive_summary.parquet")

    keep = ["serial_number", "date", *feature_columns()]
    out_dir = processed / f"features_rolling_N{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split, quarters, ratio in [
        ("train", cfg["split"]["train_quarters"], float(cfg["data"]["healthy_downsample_ratio"])),
        ("val", cfg["split"]["val_quarters"], None),
        ("test", cfg["split"]["test_quarters"], None),
    ]:
        lf = _split_lazy(interim_glob, summary_path, quarters, horizon)
        results[split] = build_split(lf, out_dir / f"{split}.parquet", keep, ratio)
    results["feature_cols"] = feature_columns()
    results["out_dir"] = str(out_dir)
    return results


if __name__ == "__main__":
    import json
    import sys

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(json.dumps(make_rolling_dataset(cfg, PROJECT_ROOT, n), indent=2))
