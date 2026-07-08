"""Phase 0 - reconstruct each drive's life history and build labels.

Because `failure=1` appears only on a drive's last operational day, we aggregate all
snapshots per `serial_number` (across all quarters) to find each drive's first/last day
and whether it ever failed. From that we derive, per drive-day:

  - rul_time_days : days from this day to the drive's end (failure OR last observation)
  - event         : 1 if the drive failed (uncensored), 0 if right-censored
  - will_fail_within_{N} : classification label per horizon N, with honest censoring:
        1  -> failed and end is within N days
        0  -> observed alive at least N days later
        null -> censored before N days elapsed (unknown; dropped for that horizon)

Outputs (data/processed/):
  - drive_summary.parquet : one row per drive (model, capacity, lifetime, event)
  - labels.parquet        : lean per-drive-day table (identity + date + labels)

Memory-safe: the per-drive summary is small (one row per serial); the big day-level join
is streamed to Parquet via polars' streaming engine.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def build_drive_summary(interim_glob: str | Path) -> pl.DataFrame:
    """One row per drive: first/last observed day, lifetime, and failure (event) flag."""
    lf = pl.scan_parquet(str(interim_glob))
    summary = (
        lf.group_by("serial_number")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.col("failure").max().cast(pl.Int8).alias("event"),
            pl.col("model").last().alias("model"),
            pl.col("capacity_bytes").last().alias("capacity_bytes"),
            pl.len().alias("n_days_observed"),
        )
        .with_columns(
            ((pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1)
            .alias("lifetime_days")
        )
        .collect(engine="streaming")
    )
    return summary


def build_labels(
    interim_glob: str | Path,
    out_path: str | Path,
    horizons_days: list[int],
    drive_summary: pl.DataFrame,
) -> str:
    """Stream the day-level labelled table to Parquet. Returns the output path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ends = drive_summary.select("serial_number", "last_date", "event").lazy()
    lf = (
        pl.scan_parquet(str(interim_glob))
        .select("date", "serial_number", "model", "capacity_bytes", "failure")
        .join(ends, on="serial_number", how="left")
        .with_columns(
            ((pl.col("last_date") - pl.col("date")).dt.total_days()).alias("rul_time_days")
        )
    )

    for n in horizons_days:
        lf = lf.with_columns(
            pl.when((pl.col("event") == 1) & (pl.col("rul_time_days") <= n))
            .then(pl.lit(1, dtype=pl.Int8))
            .when(pl.col("rul_time_days") >= n)
            .then(pl.lit(0, dtype=pl.Int8))
            .otherwise(None)  # censored before horizon -> unknown
            .alias(f"will_fail_within_{n}")
        )

    lf = lf.drop("last_date")
    lf.sink_parquet(out_path, compression="zstd")
    return str(out_path)


def run(interim_dir: str | Path, processed_dir: str | Path, horizons_days: list[int]) -> dict:
    """Build and persist drive_summary + labels; return a small EDA summary."""
    interim_dir, processed_dir = Path(interim_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    glob = interim_dir / "*.parquet"

    summary = build_drive_summary(glob)
    summary.write_parquet(processed_dir / "drive_summary.parquet", compression="zstd")
    labels_path = build_labels(glob, processed_dir / "labels.parquet", horizons_days, summary)

    n_drives = summary.height
    n_failed = int(summary["event"].sum())
    return {
        "drives": n_drives,
        "failed_drives": n_failed,
        "failed_pct": round(100 * n_failed / n_drives, 3),
        "labels_out": labels_path,
        "summary_out": str(processed_dir / "drive_summary.parquet"),
    }


if __name__ == "__main__":
    import json

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    res = run(
        PROJECT_ROOT / cfg["data"]["interim_dir"],
        PROJECT_ROOT / cfg["data"]["processed_dir"],
        cfg["labels"]["horizons_days"],
    )
    print(json.dumps(res, indent=2))
