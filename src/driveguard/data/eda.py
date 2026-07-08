"""Phase 0 - EDA on the labelled Backblaze data.

Computes the numbers that shape the modelling choices (imbalance magnitude per horizon,
failure trend over time, per-model failure rates, RUL distribution) and saves figures.
Everything is read with polars streaming / lazy scans so it stays memory-safe on 113M rows.

Outputs:
  reports/eda_summary.json        machine-readable stats
  reports/figures/*.png           figures for the README / writeup
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import polars as pl

HORIZONS = [7, 15, 30]


def _label_balance(labels_path: Path) -> dict:
    """Positives / negatives / censored-dropped and imbalance ratio per horizon."""
    lf = pl.scan_parquet(str(labels_path))
    aggs = []
    for n in HORIZONS:
        col = pl.col(f"will_fail_within_{n}")
        aggs += [
            (col == 1).sum().alias(f"pos_{n}"),
            (col == 0).sum().alias(f"neg_{n}"),
            col.is_null().sum().alias(f"cens_{n}"),
        ]
    row = lf.select(aggs).collect(engine="streaming").to_dicts()[0]
    out = {}
    for n in HORIZONS:
        pos, neg, cens = row[f"pos_{n}"], row[f"neg_{n}"], row[f"cens_{n}"]
        out[str(n)] = {
            "positives": int(pos),
            "negatives": int(neg),
            "censored_dropped": int(cens),
            "positive_rate_pct": round(100 * pos / (pos + neg), 5) if pos + neg else None,
            "imbalance_neg_per_pos": round(neg / pos, 1) if pos else None,
        }
    return out


def _failures_by_month(labels_path: Path, fig_dir: Path) -> None:
    lf = pl.scan_parquet(str(labels_path))
    by_month = (
        lf.group_by(pl.col("date").dt.truncate("1mo").alias("month"))
        .agg(pl.col("failure").sum().alias("failures"),
             pl.len().alias("drive_days"))
        .sort("month")
        .collect(engine="streaming")
    )
    m = by_month["month"].to_list()
    plt.figure(figsize=(9, 4))
    plt.bar([d.strftime("%Y-%m") for d in m], by_month["failures"].to_list(), color="#c0392b")
    plt.title("Drive failures per month (2024-Q4 .. 2025-Q3)")
    plt.ylabel("failures")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "failures_by_month.png", dpi=120)
    plt.close()


def _model_failure_rates(summary_path: Path, fig_dir: Path) -> list[dict]:
    s = pl.read_parquet(str(summary_path))
    by_model = (
        s.group_by("model")
        .agg(pl.len().alias("drives"), pl.col("event").sum().alias("failures"))
        .with_columns((100 * pl.col("failures") / pl.col("drives")).alias("failure_rate_pct"))
        .filter(pl.col("drives") >= 1000)  # stable estimate only
        .sort("failure_rate_pct", descending=True)
    )
    top = by_model.head(15)
    plt.figure(figsize=(9, 6))
    plt.barh(top["model"].to_list()[::-1], top["failure_rate_pct"].to_list()[::-1], color="#2c3e50")
    plt.title("Failure rate by drive model (>=1000 drives), top 15")
    plt.xlabel("failure rate (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "failure_rate_by_model_top15.png", dpi=120)
    plt.close()
    return by_model.head(15).to_dicts()


def _rul_distribution(labels_path: Path, fig_dir: Path) -> None:
    lf = pl.scan_parquet(str(labels_path))
    # days-to-failure for failed drives, capped at 180 for readability
    rul = (
        lf.filter((pl.col("event") == 1) & (pl.col("rul_time_days") <= 180))
        .select("rul_time_days")
        .collect(engine="streaming")["rul_time_days"]
        .to_list()
    )
    plt.figure(figsize=(9, 4))
    plt.hist(rul, bins=60, color="#e67e22")
    plt.title("Days-to-failure distribution (failed drives, <=180d of runway)")
    plt.xlabel("days before failure")
    plt.ylabel("drive-days")
    plt.tight_layout()
    plt.savefig(fig_dir / "rul_distribution_failed.png", dpi=120)
    plt.close()


def run(processed_dir: str | Path, reports_dir: str | Path) -> dict:
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels_path = processed_dir / "labels.parquet"
    summary_path = processed_dir / "drive_summary.parquet"

    s = pl.read_parquet(str(summary_path))
    total_dd = pl.scan_parquet(str(labels_path)).select(pl.len()).collect().item()

    summary = {
        "total_drive_days": int(total_dd),
        "unique_drives": int(s.height),
        "failed_drives": int(s["event"].sum()),
        "drive_level_failure_pct": round(100 * s["event"].sum() / s.height, 3),
        "drive_day_failure_pct": round(100 * s["event"].sum() / total_dd, 5),
        "median_lifetime_days_all": int(s["lifetime_days"].median()),
        "median_lifetime_days_failed": int(s.filter(pl.col("event") == 1)["lifetime_days"].median()),
        "label_balance_by_horizon": _label_balance(labels_path),
    }
    _failures_by_month(labels_path, fig_dir)
    summary["top_models_by_failure_rate"] = _model_failure_rates(summary_path, fig_dir)
    _rul_distribution(labels_path, fig_dir)

    (reports_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    from driveguard.config import PROJECT_ROOT

    res = run(PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "reports")
    print(json.dumps(res, indent=2))
