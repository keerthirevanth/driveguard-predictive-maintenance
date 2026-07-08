"""Phase 2 - model bake-off. Empirical benchmark, no pre-picked winner.

Trains the full model lineup from config across imbalance strategies, tunes with
Optuna, logs every run to MLflow (params, metrics, artifacts, SHAP).

Lineup (see config.models):
  classification: logreg, lightgbm, xgboost, catboost, random_forest, tabnet
  sequence:       lstm, gru, cnn1d, temporal_transformer
  survival:       cox_ph, random_survival_forest, deepsurv, weibull_aft

TODO(milestone-3/4): implement per-model trainers + Optuna objective + MLflow logging.
"""
from __future__ import annotations


def run_bakeoff(cfg: dict) -> None:
    """Train + tune every configured model; log all runs to MLflow; return leaderboard."""
    raise NotImplementedError("Milestone 3: implement model bake-off.")
