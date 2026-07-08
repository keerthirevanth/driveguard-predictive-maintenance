"""Phase 1 - feature engineering (v1: point-in-time, memory-safe).

Builds train/val/test feature tables for a given horizon N and feature set, written to
Parquet so downstream training/scoring never has to hold the full 113M rows in RAM.

v1 feature sets (config.features.feature_sets_to_compare):
  - big5 : Backblaze's flagged predictors (SMART 5/187/188/197/198, raw)
  - full : all 93 smart_*_raw columns
  Both always include base features: model (categorical), capacity_gb, drive_age_days.

Rolling/temporal features are intentionally deferred to Milestone 3 (they need per-drive
contiguous history and are much heavier). This is the honest point-in-time baseline set.

Labels + splits are re-derived here from drive_summary (small, broadcast join) rather than
re-reading labels.parquet, so the whole thing stays a single streaming pass.

Training negatives are undersampled deterministically (hash-based, reproducible) to
config.imbalance ratio; val/test keep the natural distribution for honest evaluation.
Censored rows (unknown label at horizon N) are dropped.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

SMART_BIG5 = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
BASE_FEATURES = ["model", "capacity_gb", "drive_age_days"]
_QUARTER_RANGE = {
    "2024-Q4": ("2024-10-01", "2024-12-31"),
    "2025-Q1": ("2025-01-01", "2025-03-31"),
    "2025-Q2": ("2025-04-01", "2025-06-30"),
    "2025-Q3": ("2025-07-01", "2025-09-30"),
}


def _smart_cols(feature_set: str, all_cols: list[str]) -> list[str]:
    if feature_set == "big5":
        return SMART_BIG5
    if feature_set == "full":
        return [c for c in all_cols if c.endswith("_raw")]
    raise ValueError(f"unknown feature_set: {feature_set}")


def _labeled_lazy(interim_glob: str, summary_path: str, smart_cols: list[str], n: int) -> pl.LazyFrame:
    ds = pl.read_parquet(summary_path).select("serial_number", "first_date", "last_date", "event")
    lf = (
        pl.scan_parquet(interim_glob)
        .select(["date", "serial_number", "model", "capacity_bytes", *smart_cols])
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
        .drop_nulls("y")  # drop censored (unknown outcome at horizon N)
    )
    # cast SMART to float32 to halve memory downstream
    return lf.with_columns([pl.col(c).cast(pl.Float32) for c in smart_cols])


def _range_filter(lf: pl.LazyFrame, quarters: list[str]) -> pl.LazyFrame:
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    return lf.filter(pl.col("date").is_between(pl.lit(lo).str.to_date(), pl.lit(hi).str.to_date()))


def _counts(lf: pl.LazyFrame) -> tuple[int, int]:
    row = lf.select(
        (pl.col("y") == 1).sum().alias("pos"), (pl.col("y") == 0).sum().alias("neg")
    ).collect(engine="streaming").to_dicts()[0]
    return int(row["pos"]), int(row["neg"])


def build_split(
    lf: pl.LazyFrame, out_path: Path, keep_cols: list[str],
    undersample_ratio: float | None = None,
) -> dict:
    """Sink one split to Parquet. If undersample_ratio set, keep all positives + a
    deterministic hash-sampled fraction of negatives to hit ~ratio:1."""
    pos, neg = _counts(lf)
    sel = lf.select([*keep_cols, "y"])
    if undersample_ratio is not None and neg > undersample_ratio * pos:
        frac = (undersample_ratio * pos) / neg
        rnd = (
            (pl.col("serial_number").cast(pl.String) + pl.col("date").cast(pl.String))
            .hash(seed=42) % 1_000_000
        ) / 1_000_000
        sel = sel.filter((pl.col("y") == 1) | (rnd < frac))
    sel.sink_parquet(out_path, compression="zstd")
    return {"out": str(out_path), "orig_pos": pos, "orig_neg": neg,
            "undersample_ratio": undersample_ratio}


def make_dataset(cfg: dict, project_root: Path, horizon: int, feature_set: str = "big5") -> dict:
    interim = project_root / cfg["data"]["interim_dir"]
    processed = project_root / cfg["data"]["processed_dir"]
    interim_glob = str(interim / "*.parquet")
    summary_path = str(processed / "drive_summary.parquet")

    all_cols = pl.scan_parquet(interim_glob).collect_schema().names()
    smart = _smart_cols(feature_set, all_cols)
    keep = ["serial_number", "date", *BASE_FEATURES, *smart]

    out_dir = processed / f"features_{feature_set}_N{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split, quarters, ratio in [
        ("train", cfg["split"]["train_quarters"], float(cfg["data"]["healthy_downsample_ratio"])),
        ("val", cfg["split"]["val_quarters"], None),
        ("test", cfg["split"]["test_quarters"], None),
    ]:
        lf = _range_filter(_labeled_lazy(interim_glob, summary_path, smart, horizon), quarters)
        results[split] = build_split(lf, out_dir / f"{split}.parquet", keep, ratio)
    results["feature_cols"] = BASE_FEATURES + smart
    results["out_dir"] = str(out_dir)
    return results


if __name__ == "__main__":
    import json
    import sys

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fs = sys.argv[2] if len(sys.argv) > 2 else "big5"
    print(json.dumps(make_dataset(cfg, PROJECT_ROOT, n, fs), indent=2))
