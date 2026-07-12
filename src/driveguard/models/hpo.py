"""Milestone 3 - Optuna hyperparameter search for the top tree models.

Evidence-first: we tune ALL competitive contenders (lightgbm, random_forest, catboost) on
the rolling feature set rather than pre-picking one. logreg/tabnet are excluded (clearly
non-competitive in the bake-off - that pruning is evidence-based, not assumed).

To stay inside a Kaggle session:
  - HPO objective = PR-AUC on a VALIDATION SUBSAMPLE (all positives + capped negatives),
    so each trial fits on train and scores a few million rows, not 28M.
  - trial budgets are scaled to each model's fit cost.
  - the FINAL tuned config for each model is re-fit and scored on the FULL held-out test.
Test (2025-Q3) is never used for tuning - only for the final honest comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from sklearn.metrics import average_precision_score

from driveguard.models.train import (
    CAT, FACTORY, _batch_score, _feature_cols, _model_code_map, _prep,
)

# fit-cost-aware defaults; override per run
DEFAULT_TRIALS = {"lightgbm": 60, "random_forest": 20, "catboost": 15}


def _val_subsample(feature_dir: Path, feats, code_map, cap_neg=3_000_000, seed=42):
    lf = pl.scan_parquet(str(feature_dir / "val.parquet"))
    neg_total = lf.select((pl.col("y") == 0).sum()).collect(engine="streaming").item()
    frac = min(1.0, cap_neg / max(neg_total, 1))
    rnd = ((pl.col(CAT).cast(pl.String) + pl.col("date").cast(pl.String)).hash(seed)
           % 1_000_000) / 1_000_000
    sub = lf.filter((pl.col("y") == 1) | ((pl.col("y") == 0) & (rnd < frac)))
    return _prep(sub.collect(engine="streaming"), feats, code_map)


def _space(trial, model: str) -> dict:
    if model == "lightgbm":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        )
    if model == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 400),
            max_depth=trial.suggest_categorical("max_depth", [None, 12, 20, 30]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        )
    if model == "catboost":
        return dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            iterations=trial.suggest_int("iterations", 300, 900),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            task_type=trial.suggest_categorical("task_type", ["GPU"]),  # Kaggle GPU
        )
    raise ValueError(model)


def tune_model(feature_dir: Path, model: str, spw: float, cat_idx: int,
               Xtr, ytr, Xv, yv, feats, code_map, n_trials: int, cfg: dict) -> dict:
    import optuna

    def objective(trial):
        params = _space(trial, model)
        est = FACTORY[model](Xtr, ytr, spw, cat_idx, params)
        return average_precision_score(yv, est.predict_proba(Xv)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # refit best on train, evaluate on FULL test
    from driveguard.evaluation.metrics import evaluate_classification
    best = FACTORY[model](Xtr, ytr, spw, cat_idx, study.best_params)
    yt, ys = _batch_score(best, feature_dir / "test.parquet", feats, code_map)
    return {"model": model, "best_params": study.best_params,
            "best_val_pr_auc": float(study.best_value),
            "test": evaluate_classification(yt, ys, cfg), "n_trials": n_trials}


def run(feature_dir: str | Path, models: list[str], cfg: dict,
        trials: dict | None = None, mlflow_uri: str | None = None) -> list[dict]:
    feature_dir = Path(feature_dir)
    trials = trials or DEFAULT_TRIALS
    feats = _feature_cols(feature_dir / "train.parquet")
    cat_idx = feats.index(CAT)
    code_map = _model_code_map(feature_dir / "train.parquet")

    train = pl.read_parquet(feature_dir / "train.parquet")
    Xtr, ytr = _prep(train, feats, code_map)
    spw = float((ytr == 0).sum() / max(ytr.sum(), 1))
    del train
    Xv, yv = _val_subsample(feature_dir, feats, code_map)

    try:
        import mlflow
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("driveguard_hpo")
        have_mlflow = True
    except Exception:
        have_mlflow = False

    results = []
    for m in models:
        r = tune_model(feature_dir, m, spw, cat_idx, Xtr, ytr, Xv, yv,
                       feats, code_map, trials.get(m, 20), cfg)
        results.append(r)
        print(json.dumps({"model": m, "best_val_pr_auc": round(r["best_val_pr_auc"], 4),
                          "test_pr_auc": round(r["test"]["pr_auc"], 4),
                          "test_recall@1%fpr": round(
                              r["test"]["recall_at_fpr_1pct"]["recall"] or 0, 3)}),
              flush=True)
        if have_mlflow:
            with mlflow.start_run(run_name=f"hpo_{m}"):
                mlflow.log_params({f"best_{k}": v for k, v in r["best_params"].items()})
                mlflow.log_metric("val_pr_auc", r["best_val_pr_auc"])
                mlflow.log_metric("test_pr_auc", r["test"]["pr_auc"])
    return results
