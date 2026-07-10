"""Milestone 4 - survival / RUL model bake-off (classical + forest).

Lineup: Cox PH, Weibull AFT (lifelines), Random Survival Forest (scikit-survival).
Each model exposes:
  risk(X)       -> higher = more likely to fail sooner (for the concordance index)
  pred_time(X)  -> point estimate of remaining days (for RUL MAE)

Classical survival models don't need millions of rows, so train/eval are subsampled
(all events kept + capped censored). Metric: concordance index (primary) + RUL MAE.
Sequence/deep RUL (LSTM/GRU/CNN, DeepSurv) is a separate next increment.
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np
import polars as pl

from driveguard.evaluation.metrics import evaluate_survival
from driveguard.models.train import CAT, _model_code_map

DROP = ("serial_number", "date", "event", "duration")


def _feature_cols(train_path: Path) -> list[str]:
    cols = pl.read_parquet(train_path, n_rows=1).columns
    feats = [c for c in cols if c not in DROP]
    return [CAT] + [c for c in feats if c != CAT]


def _load(path: Path, feats, code_map, cap_censored: int, seed=42):
    """Load a split: all events + capped censored, imputed to numpy + arrays."""
    lf = pl.scan_parquet(str(path))
    ev = lf.filter(pl.col("event") == 1).collect(engine="streaming")
    neg_total = lf.select((pl.col("event") == 0).sum()).collect(engine="streaming").item()
    frac = min(1.0, cap_censored / max(neg_total, 1))
    rnd = ((pl.col(CAT).cast(pl.String) + pl.col("date").cast(pl.String)).hash(seed)
           % 1_000_000) / 1_000_000
    cen = lf.filter((pl.col("event") == 0) & (rnd < frac)).collect(engine="streaming")
    df = pl.concat([ev, cen]).with_columns(
        pl.col(CAT).replace_strict(code_map, default=-1).cast(pl.Int32)
    )
    X = df.select(feats).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)  # classical models need dense input
    event = df["event"].to_numpy().astype(bool)
    dur = df["duration"].to_numpy().astype(float)
    dur = np.clip(dur, 1.0, None)  # survival models need strictly positive durations
    return X, event, dur, feats


# --- model wrappers ---

class _CoxPH:
    def fit(self, X, event, dur, feats):
        import pandas as pd
        from lifelines import CoxPHFitter
        d = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        d["duration"], d["event"] = dur, event.astype(int)
        self.f = CoxPHFitter(penalizer=0.1)
        self.f.fit(d, duration_col="duration", event_col="event")
        self.cols = [f"f{i}" for i in range(X.shape[1])]
        return self

    def _df(self, X):
        import pandas as pd
        return pd.DataFrame(X, columns=self.cols)

    def risk(self, X):
        return self.f.predict_partial_hazard(self._df(X)).to_numpy()

    def pred_time(self, X):
        return self.f.predict_expectation(self._df(X)).to_numpy()


class _WeibullAFT:
    def fit(self, X, event, dur, feats):
        import pandas as pd
        from lifelines import WeibullAFTFitter
        d = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        d["duration"], d["event"] = dur, event.astype(int)
        self.f = WeibullAFTFitter(penalizer=0.1)
        self.f.fit(d, duration_col="duration", event_col="event")
        self.cols = [f"f{i}" for i in range(X.shape[1])]
        return self

    def pred_time(self, X):
        import pandas as pd
        return self.f.predict_median(pd.DataFrame(X, columns=self.cols)).to_numpy()

    def risk(self, X):
        return -self.pred_time(X)  # shorter predicted time = higher risk


class _RSF:
    def fit(self, X, event, dur, feats):
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        y = Surv.from_arrays(event, dur)
        self.f = RandomSurvivalForest(n_estimators=200, min_samples_leaf=50,
                                      max_features="sqrt", n_jobs=-1, random_state=42)
        self.f.fit(X, y)
        return self

    def risk(self, X):
        return self.f.predict(X)  # risk score

    def pred_time(self, X):
        # expected time = integral of survival function over the time grid
        surv = self.f.predict_survival_function(X, return_array=True)
        t = self.f.unique_times_
        dt = np.diff(np.concatenate([[0.0], t]))
        return (surv * dt).sum(axis=1)


FACTORY = {"cox_ph": _CoxPH, "weibull_aft": _WeibullAFT, "random_survival_forest": _RSF}


def run_survival(feature_dir: str | Path, models: list[str],
                 train_cap: int = 120_000, eval_cap: int = 60_000,
                 mlflow_uri: str | None = None) -> list[dict]:
    feature_dir = Path(feature_dir)
    feats = _feature_cols(feature_dir / "train.parquet")
    code_map = _model_code_map(feature_dir / "train.parquet")

    Xtr, etr, dtr, _ = _load(feature_dir / "train.parquet", feats, code_map, train_cap)
    Xte, ete, dte, _ = _load(feature_dir / "test.parquet", feats, code_map, eval_cap)

    try:
        import mlflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("driveguard_survival")
        have_mlflow = True
    except Exception:
        have_mlflow = False

    results = []
    for name in models:
        t0 = _time.time()
        try:
            model = FACTORY[name]().fit(Xtr, etr, dtr, feats)
            risk = model.risk(Xte)
            try:
                ptime = model.pred_time(Xte)
            except Exception:
                ptime = None
            res = {"model": name, "train_rows": int(len(etr)),
                   "fit_sec": round(_time.time() - t0, 1),
                   "test": evaluate_survival(ete, dte, risk, ptime), "status": "ok"}
        except Exception as e:
            res = {"model": name, "status": "error", "error": str(e)}
        results.append(res)
        msg = (f"  c_index={res['test']['c_index']:.4f}" if res.get("status") == "ok"
               and res["test"].get("c_index") is not None else "")
        print(json.dumps({"model": name, "status": res.get("status")}) + msg, flush=True)
        if have_mlflow and res.get("status") == "ok":
            with mlflow.start_run(run_name=f"surv_{name}"):
                mlflow.log_param("model", name)
                if res["test"].get("c_index") is not None:
                    mlflow.log_metric("test_c_index", res["test"]["c_index"])
                if res["test"].get("rul_mae_days") is not None:
                    mlflow.log_metric("test_rul_mae_days", res["test"]["rul_mae_days"])
    return results


if __name__ == "__main__":
    import sys

    from driveguard.config import PROJECT_ROOT

    fdir = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "data" / "processed" / "survival_rolling")
    mdls = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FACTORY)
    board = run_survival(fdir, mdls, mlflow_uri=str(PROJECT_ROOT / "mlruns"))
    Path(PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "survival_leaderboard.json").write_text(json.dumps(board, indent=2))
