"""Phase 0 - ingest Backblaze Drive Stats CSVs -> partitioned Parquet.

Real data source (verified 2026-07-08):
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
  Kaggle mirror: https://www.kaggle.com/datasets/backblaze/hard-drive-test-data

Each row = one drive-day. `failure=1` only on a drive's last operational day.

NOTE: This is a scaffold. Implementation is TODO in milestone 1. No data pulled yet.
"""
from __future__ import annotations

from pathlib import Path

# Core identity/label columns present in every Backblaze snapshot.
BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def download_quarters(quarters: list[str], raw_dir: str | Path) -> None:
    """Download the given quarters (e.g. '2025-Q1') to raw_dir.

    TODO(milestone-1): pull zipped quarterly CSVs from Backblaze or Kaggle CLI,
    verify checksums, unzip. Keep raw files out of git (see .gitignore / DVC).
    """
    raise NotImplementedError("Milestone 1: implement Backblaze/Kaggle download.")


def csv_to_parquet(raw_dir: str | Path, out_dir: str | Path) -> None:
    """Convert daily CSVs to partitioned Parquet (columnar, ~10x smaller).

    TODO(milestone-1): stream CSVs with polars/pyarrow, cast SMART cols, drop
    all-null SMART columns, partition by month, write Parquet.
    """
    raise NotImplementedError("Milestone 1: implement CSV->Parquet conversion.")


if __name__ == "__main__":
    print("Scaffold only. See PLAN.md milestone 1.")
