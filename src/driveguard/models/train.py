"""Phase 2/3 - model bake-off. Empirical benchmark, no pre-picked winner.

Trains a lineup on a prepared feature directory (train/val/test Parquet from
build_features / rolling), evaluates every model on the natural-distribution val and test
splits with the imbalance-aware metric bundle, and logs every run to MLflow.

Lineup: logreg, random_forest, lightgbm, xgboost, catboost, tabnet (GPU when available).
Imbalance handling: train negatives are already undersampled (config ratio); on top of that
GBDTs use scale_pos_weight and linear/forest models use class_weight='balanced'. Threshold
tuning is handled in evaluation (best F-beta). SMOTE/ADASYN are intentionally NOT used - at
tens of millions of rows they are infeasible and undersampling + class weights is the
correct, scalable choice (documented, not assumed).

Val/test are scored in batches so the full quarters never load at once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from driveguard.evaluation.metrics import evaluate_classification

DROP = ("serial_number", "date", "y")
CAT = "model"


def _feature_cols(train_path: Path) -> list[str]:
    cols = pl.read_parquet(train_path, n_rows=1).columns
    feats = [c for c in cols if c not in DROP]
    return [CAT] + [c for c in feats if c != CAT]  # model first


def _model_code_map(train_path: Path) -> dict:
    m = pl.read_parquet(train_path, columns=[CAT])[CAT].unique().to_list()
    return {v: i for i, v in enumerate(sorted(x for x in m if x is not None))}


def _prep(df: pl.DataFrame, feats: list[str], code_map: dict) -> tuple[np.ndarray, np.ndarray]:
    df = df.with_columns(
        pl.col(CAT).replace_strict(code_map, default=-1).cast(pl.Int32)
    )
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.int8)
    return X, y


def _batch_score(estimator, path: Path, feats: list[str], code_map: dict,
                 batch_rows: int = 1_000_000):
    ys, ss = [], []
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=[*feats, "y"]):
        X, y = _prep(pl.from_arrow(b), feats, code_map)
        ss.append(estimator.predict_proba(X)[:, 1].astype(np.float32))
        ys.append(y)
    return np.concatenate(ys), np.concatenate(ss)


# --- model factory: each returns a fitted estimator exposing predict_proba(X) ---

def _fit_lightgbm(Xtr, ytr, spw, cat_idx, params=None):
    import lightgbm as lgb
    p = dict(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
             n_jobs=-1, random_state=42, verbose=-1)
    p.update(params or {})
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr, ytr, categorical_feature=[cat_idx])
    return m


def _fit_xgboost(Xtr, ytr, spw, cat_idx, params=None):
    import xgboost as xgb
    p = dict(n_estimators=800, learning_rate=0.05, max_depth=8, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
             tree_method="hist", n_jobs=-1, random_state=42, eval_metric="aucpr")
    p.update(params or {})
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    return m


def _fit_catboost(Xtr, ytr, spw, cat_idx, params=None):
    # CatBoost rejects float-valued categorical columns, so pass the model column as a
    # string via a DataFrame (our X is all-float32). Numeric NaNs are handled natively.
    import pandas as pd
    from catboost import CatBoostClassifier

    def to_df(X):
        df = pd.DataFrame(X)
        df[cat_idx] = pd.Series(X[:, cat_idx]).fillna(-1).astype("int64").astype(str)
        return df

    p = dict(iterations=800, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
             scale_pos_weight=spw, random_seed=42, verbose=0)
    p.update(params or {})
    m = CatBoostClassifier(**p)
    m.fit(to_df(Xtr), ytr, cat_features=[cat_idx])

    class _Wrap:
        def predict_proba(self, X):
            return m.predict_proba(to_df(X))
    return _Wrap()


def _fit_sklearn_linear(Xtr, ytr, spw, cat_idx, params=None):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=-1),
    )
    m.fit(Xtr, ytr)
    return m


def _fit_random_forest(Xtr, ytr, spw, cat_idx, params=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    p = dict(n_estimators=300, max_depth=None, min_samples_leaf=50,
             max_features="sqrt", class_weight="balanced_subsample",
             n_jobs=-1, random_state=42)
    p.update(params or {})
    m = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(**p))
    m.fit(Xtr, ytr)
    return m


def _fit_tabnet(Xtr, ytr, spw, cat_idx, params=None):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xi = imp.transform(Xtr).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabNetClassifier(seed=42, device_name=dev, verbose=0)
    clf.fit(Xi, ytr, weights=1, max_epochs=40, patience=8, batch_size=16384)
    # wrap to apply the same imputer at predict time
    class _Wrap:
        def predict_proba(self, X):
            return clf.predict_proba(imp.transform(X).astype(np.float32))
    return _Wrap()


FACTORY = {
    "logreg": _fit_sklearn_linear,
    "random_forest": _fit_random_forest,
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "catboost": _fit_catboost,
    "tabnet": _fit_tabnet,
}


def run_bakeoff(feature_dir: str | Path, feature_set: str, horizon: int,
                models: list[str], cfg: dict, mlflow_uri: str | None = None) -> list[dict]:
    feature_dir = Path(feature_dir)
    feats = _feature_cols(feature_dir / "train.parquet")
    cat_idx = feats.index(CAT)
    code_map = _model_code_map(feature_dir / "train.parquet")

    train = pl.read_parquet(feature_dir / "train.parquet")
    Xtr, ytr = _prep(train, feats, code_map)
    spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    del train

    try:
        import mlflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(f"driveguard_{feature_set}_N{horizon}")
        have_mlflow = True
    except Exception:
        have_mlflow = False

    leaderboard = []
    for name in models:
        t0 = time.time()
        try:
            est = FACTORY[name](Xtr, ytr, spw, cat_idx)
            res = {"model": name, "feature_set": feature_set, "horizon": horizon,
                   "train_rows": int(len(ytr)), "fit_sec": round(time.time() - t0, 1)}
            for split in ["val", "test"]:
                yt, ys = _batch_score(est, feature_dir / f"{split}.parquet", feats, code_map)
                res[split] = evaluate_classification(yt, ys, cfg)
            res["status"] = "ok"
        except Exception as e:  # keep the bake-off going if one model fails
            res = {"model": name, "feature_set": feature_set, "status": "error", "error": str(e)}
        leaderboard.append(res)

        if have_mlflow and res.get("status") == "ok":
            with mlflow.start_run(run_name=f"{name}_{feature_set}_N{horizon}"):
                mlflow.log_params({"model": name, "feature_set": feature_set,
                                   "horizon": horizon, "scale_pos_weight": round(spw, 2),
                                   "n_features": len(feats)})
                for split in ["val", "test"]:
                    mlflow.log_metric(f"{split}_pr_auc", res[split]["pr_auc"])
                    mlflow.log_metric(f"{split}_roc_auc", res[split]["roc_auc"])
                    mlflow.log_metric(f"{split}_recall_at_1pct_fpr",
                                      res[split]["recall_at_fpr_1pct"]["recall"] or 0.0)
        print(json.dumps({k: res.get(k) for k in ("model", "feature_set", "status")})
              + (f"  test_pr_auc={res['test']['pr_auc']:.4f}" if res.get("status") == "ok" else ""),
              flush=True)

    return leaderboard


if __name__ == "__main__":
    import sys

    from driveguard.config import PROJECT_ROOT, load_config

    cfg = load_config()
    fdir = sys.argv[1]
    fset = sys.argv[2] if len(sys.argv) > 2 else "rolling"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    mdls = sys.argv[4].split(",") if len(sys.argv) > 4 else list(FACTORY)
    board = run_bakeoff(fdir, fset, n, mdls, cfg, str(PROJECT_ROOT / "mlruns"))
    Path(PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / f"bakeoff_{fset}_N{n}.json").write_text(json.dumps(board, indent=2))
