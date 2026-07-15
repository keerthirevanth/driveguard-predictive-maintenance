"""Phase 4 - drift monitoring + retrain trigger.

Feature drift is organic here: as the fleet ages and new drive models arrive, the SMART
distributions shift between the training window and the current window - the whole reason we
held out the newest quarter.

Core drift detection uses PSI (population stability index) + a KS two-sample test per
feature, which are dependency-light and version-stable. Evidently is used only for an
optional rich HTML report (its API changes across versions, so it must never be load-bearing).

A retrain is triggered when too many features drift, or when live performance (PR-AUC on a
freshly-labelled window) falls below the trained level by more than a configured margin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from driveguard.features.build_features import SMART_BIG5, _QUARTER_RANGE

# PSI: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant. PSI is the primary drift flag
# because it is robust to sample size; a KS p-value at tens of thousands of rows flags
# trivially small differences, so KS is reported as a diagnostic but does not drive the flag.
PSI_THRESHOLD = 0.1


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between reference and current, on reference quantiles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def cat_psi(ref: pl.Series, cur: pl.Series) -> float:
    """PSI for a categorical column, over the union of categories."""
    r = ref.value_counts(normalize=True)
    c = cur.value_counts(normalize=True)
    name = ref.name
    rd = dict(zip(r[name].to_list(), r["proportion"].to_list()))
    cd = dict(zip(c[name].to_list(), c["proportion"].to_list()))
    cats = set(rd) | set(cd)
    total = 0.0
    for k in cats:
        rp = max(rd.get(k, 0.0), 1e-6)
        cp = max(cd.get(k, 0.0), 1e-6)
        total += (cp - rp) * np.log(cp / rp)
    return float(total)


def compute_drift(ref: pl.DataFrame, cur: pl.DataFrame, numeric: list[str],
                  categorical: list[str] | None = None) -> dict:
    from scipy.stats import ks_2samp
    per_feature = []
    for f in numeric:
        r = ref[f].drop_nulls().to_numpy()
        c = cur[f].drop_nulls().to_numpy()
        if len(r) < 50 or len(c) < 50:
            continue
        p = psi(r, c)
        ks = ks_2samp(r, c)
        drifted = bool(p > PSI_THRESHOLD)  # PSI-driven; KS reported only
        per_feature.append({"feature": f, "type": "numeric", "psi": round(p, 4),
                            "ks_stat": round(float(ks.statistic), 4),
                            "ks_pvalue": float(ks.pvalue), "drifted": drifted})
    for f in categorical or []:
        p = cat_psi(ref[f], cur[f])
        n_new = len(set(cur[f].unique().to_list()) - set(ref[f].unique().to_list()))
        drifted = bool(p > PSI_THRESHOLD or n_new > 0)
        per_feature.append({"feature": f, "type": "categorical", "psi": round(p, 4),
                            "new_categories": n_new, "drifted": drifted})
    n_drift = sum(d["drifted"] for d in per_feature)
    return {"features": per_feature, "n_features": len(per_feature),
            "n_drifted": n_drift,
            "drifted_share": round(n_drift / max(len(per_feature), 1), 3),
            "ref_rows": ref.height, "cur_rows": cur.height}


def check_retrain(drift: dict, pr_auc_current: float | None = None,
                  pr_auc_reference: float | None = None,
                  drift_share_threshold: float = 0.5,
                  pr_auc_drop: float = 0.05) -> dict:
    reasons = []
    if drift["drifted_share"] >= drift_share_threshold:
        reasons.append(f"feature drift: {drift['n_drifted']}/{drift['n_features']} "
                       f"features drifted (>= {drift_share_threshold:.0%})")
    if pr_auc_current is not None and pr_auc_reference is not None:
        if pr_auc_reference - pr_auc_current >= pr_auc_drop:
            reasons.append(f"performance drop: PR-AUC {pr_auc_reference:.3f} -> "
                           f"{pr_auc_current:.3f} (>= {pr_auc_drop})")
    return {"retrain": bool(reasons), "reasons": reasons}


NUMERIC = ["capacity_gb", *SMART_BIG5]
CATEGORICAL = ["model"]


def _sample(interim_glob: str, quarters: list[str], n: int, seed: int = 42) -> pl.DataFrame:
    lo = min(_QUARTER_RANGE[q][0] for q in quarters)
    hi = max(_QUARTER_RANGE[q][1] for q in quarters)
    df = (pl.scan_parquet(interim_glob)
          .select(["date", "model", "capacity_bytes", *SMART_BIG5])
          .filter(pl.col("date").is_between(pl.lit(lo).str.to_date(), pl.lit(hi).str.to_date()))
          .with_columns((pl.col("capacity_bytes") / 1e9).alias("capacity_gb"))
          .select(["model", *NUMERIC])
          .collect(engine="streaming"))
    return df.sample(n=min(n, df.height), seed=seed)


def evidently_report(ref: pl.DataFrame, cur: pl.DataFrame, out_html: Path) -> bool:
    """Optional rich HTML drift report. Returns True if produced (never load-bearing)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        rep = Report(metrics=[DataDriftPreset()])
        snap = rep.run(reference_data=ref.to_pandas(), current_data=cur.to_pandas())
        snap.save_html(str(out_html))
        return True
    except Exception:
        try:  # older Evidently API
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            rep = Report(metrics=[DataDriftPreset()])
            rep.run(reference_data=ref.to_pandas(), current_data=cur.to_pandas())
            rep.save_html(str(out_html))
            return True
        except Exception:
            return False


def run(cfg: dict, project_root: Path) -> dict:
    interim_glob = str(project_root / cfg["data"]["interim_dir"] / "*.parquet")
    ref = _sample(interim_glob, cfg["split"]["train_quarters"], 60_000)
    cur = _sample(interim_glob, cfg["split"]["test_quarters"], 60_000)
    drift = compute_drift(ref, cur, NUMERIC, CATEGORICAL)

    reports = project_root / "reports"
    reports.mkdir(exist_ok=True)
    drift["evidently_html"] = evidently_report(ref, cur, reports / "drift_report.html")
    decision = check_retrain(drift)
    out = {"drift": drift, "decision": decision,
           "reference_quarters": cfg["split"]["train_quarters"],
           "current_quarters": cfg["split"]["test_quarters"]}
    (reports / "drift_report.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from driveguard.config import PROJECT_ROOT, load_config

    res = run(load_config(), PROJECT_ROOT)
    d = res["drift"]
    print(f"drift: {d['n_drifted']}/{d['n_features']} SMART features drifted "
          f"(share {d['drifted_share']}); evidently_html={d['evidently_html']}")
    for f in d["features"]:
        extra = (f"KS_p={f['ks_pvalue']:.2e}" if f["type"] == "numeric"
                 else f"new_models={f['new_categories']}")
        print(f"  {f['feature']:16s} [{f['type'][:3]}] PSI={f['psi']:.3f}  {extra}  "
              f"drifted={f['drifted']}")
    print("retrain:", res["decision"]["retrain"], "|", res["decision"]["reasons"])
