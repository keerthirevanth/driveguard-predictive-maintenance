"""Phase 0 - ingest Backblaze Drive Stats quarterly zips -> partitioned Parquet.

Real data source (verified 2026-07):
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
Quarterly zips: https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/<quarter>.zip

Facts confirmed against the actual files (not assumed):
  - Each row = one drive-day. `failure=1` only on a drive's last operational day.
  - Schema is 197 columns and identical across 2024-Q4 .. 2025-Q3:
      11 base/location cols + 93 smart_*_raw + 93 smart_*_normalized.
  - Some quarters are zipped on macOS (contain a subfolder + __MACOSX/._*/.DS_Store junk);
    others are flat. Both layouts are handled.

Design:
  - Stream one day-CSV at a time from inside the zip and append to a per-quarter Parquet
    file (pyarrow ParquetWriter). Memory stays ~one day; the ~12 GB/quarter of CSV is never
    fully unpacked to disk.
  - A fixed schema is enforced so every day's table matches (SMART cols -> Float64).
  - All columns are kept; null-heavy SMART columns cost almost nothing in Parquet and we
    avoid pruning modelling options this early.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

BASE_STR_COLS = [
    "serial_number", "model", "datacenter", "cluster_id",
    "vault_id", "pod_id", "pod_slot_num",
]
QUARTER_URLS = {
    "2024-Q4": "data_Q4_2024",
    "2025-Q1": "data_Q1_2025",
    "2025-Q2": "data_Q2_2025",
    "2025-Q3": "data_Q3_2025",
}
_B2_BASE = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"


def download_quarters(quarters: list[str], raw_dir: str | Path) -> list[Path]:
    """Download the given quarters' zips to raw_dir. Returns local paths.

    Reproducibility helper (the initial download was done via curl). Uses urllib so
    it works without extra deps; skips files already present.
    """
    import urllib.request

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for q in quarters:
        stem = QUARTER_URLS[q]
        dest = raw_dir / f"{stem}.zip"
        if not dest.exists():
            urllib.request.urlretrieve(f"{_B2_BASE}/{stem}.zip", dest)
        out.append(dest)
    return out


def _real_day_csvs(zf: zipfile.ZipFile) -> list[str]:
    """Day CSVs inside a quarter zip, excluding macOS junk (__MACOSX, ._*, .DS_Store)."""
    days = []
    for n in zf.namelist():
        base = n.rsplit("/", 1)[-1]
        if not base.endswith(".csv"):
            continue
        if "__MACOSX" in n or base.startswith("._"):
            continue
        days.append(n)
    return sorted(days, key=lambda n: n.rsplit("/", 1)[-1])


def _schema_overrides(header: list[str]) -> dict[str, type[pl.DataType]]:
    """Force a consistent dtype per column so every day's frame aligns."""
    overrides: dict[str, type[pl.DataType]] = {
        "date": pl.String,          # parsed to Date after read
        "capacity_bytes": pl.Int64,
        "failure": pl.Int8,
        "is_legacy_format": pl.String,
    }
    for c in BASE_STR_COLS:
        overrides[c] = pl.String
    for c in header:
        if c.endswith("_raw") or c.endswith("_normalized"):
            overrides[c] = pl.Float64   # nullable + handles large SMART values
    return overrides


def _read_day(zf: zipfile.ZipFile, name: str) -> pl.DataFrame:
    """Read one day-CSV from the zip into a typed polars DataFrame."""
    raw = zf.read(name)
    header = raw[: raw.find(b"\n")].decode().strip().split(",")
    df = pl.read_csv(
        io.BytesIO(raw),
        schema_overrides=_schema_overrides(header),
        infer_schema_length=0,   # trust our overrides, don't sniff
        null_values=[""],
    )
    return df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def quarter_to_parquet(zip_path: str | Path, out_path: str | Path) -> dict:
    """Stream a quarter zip -> single Parquet file. Returns a small summary."""
    zip_path, out_path = Path(zip_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    n_rows = n_fail = n_days = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for day in _real_day_csvs(zf):
                df = _read_day(zf, day)
                table = df.to_arrow()
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
                else:
                    table = table.cast(schema)   # enforce identical schema across days
                writer.write_table(table)
                n_rows += df.height
                n_fail += int(df["failure"].sum())
                n_days += 1
    finally:
        if writer is not None:
            writer.close()
    return {"quarter_zip": zip_path.name, "days": n_days, "rows": n_rows,
            "failures": n_fail, "out": str(out_path)}


def csv_to_parquet(raw_dir: str | Path, out_dir: str | Path) -> list[dict]:
    """Convert every quarter zip in raw_dir to a Parquet file in out_dir."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    summaries = []
    for zip_path in sorted(raw_dir.glob("*.zip")):
        out_path = out_dir / f"{zip_path.stem}.parquet"
        summaries.append(quarter_to_parquet(zip_path, out_path))
    return summaries


if __name__ == "__main__":
    import json

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    res = csv_to_parquet(PROJECT_ROOT / cfg["data"]["raw_dir"],
                         PROJECT_ROOT / cfg["data"]["interim_dir"])
    print(json.dumps(res, indent=2))
