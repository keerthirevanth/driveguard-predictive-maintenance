"""Phase 3 - imbalance-appropriate evaluation.

Accuracy is meaningless at 800-3400:1 imbalance, so the primary metric is PR-AUC
(average precision). We also report:
  - recall at a fixed low false-alarm rate (operational: how many failures caught if we
    tolerate only X% of healthy drives being flagged)
  - F-beta with beta>1 (missed failures cost more than false alarms)
  - Brier score + a calibration curve (are the probabilities trustworthy)
  - the precision-recall operating curve

All functions take numpy arrays so they work regardless of the model library.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> dict:
    """Highest recall achievable while keeping false-positive rate <= max_fpr."""
    order = np.argsort(-y_score)
    yt = y_true[order]
    P = yt.sum()
    N = len(yt) - P
    if P == 0 or N == 0:
        return {"recall": None, "threshold": None, "fpr": None}
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    fpr = fp / N
    ok = np.where(fpr <= max_fpr)[0]
    if len(ok) == 0:
        return {"recall": 0.0, "threshold": float(y_score[order][0]), "fpr": 0.0}
    i = ok[-1]
    return {"recall": float(tp[i] / P), "threshold": float(y_score[order][i]),
            "fpr": float(fpr[i])}


def best_fbeta(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> dict:
    """Best F-beta over all thresholds on the PR curve."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    denom = (beta**2 * prec) + rec
    with np.errstate(divide="ignore", invalid="ignore"):
        fb = np.where(denom > 0, (1 + beta**2) * prec * rec / denom, 0.0)
    j = int(np.argmax(fb))
    return {"fbeta": float(fb[j]), "threshold": float(thr[j]),
            "precision": float(prec[j]), "recall": float(rec[j])}


def evaluate_classification(y_true, y_score, cfg: dict | None = None) -> dict:
    """Full imbalance-aware metric bundle for a set of scores."""
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    beta = float((cfg or {}).get("evaluation", {}).get("fbeta_beta", 2.0)) if cfg else 2.0
    base_rate = float(y_true.mean())
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": base_rate,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "pr_auc_lift_over_base": float(average_precision_score(y_true, y_score) / base_rate)
        if base_rate > 0 else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        f"fbeta{beta:g}": best_fbeta(y_true, y_score, beta),
        "recall_at_fpr_0.1pct": recall_at_fpr(y_true, y_score, 0.001),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, y_score, 0.01),
    }
    return out


def evaluate_survival(event, time, risk_score, pred_time=None) -> dict:
    """Survival metrics.

    - concordance index (C-index): does a higher risk_score correspond to a shorter
      time-to-event, respecting right-censoring. 0.5 = random, 1.0 = perfect ranking.
    - RUL MAE: on drives that actually failed (event==1), mean abs error between the
      predicted remaining days (pred_time) and the true remaining days.
    """
    event = np.asarray(event).astype(bool)
    time = np.asarray(time, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    out = {"n": int(len(event)), "events": int(event.sum())}
    try:
        from sksurv.metrics import concordance_index_censored
        out["c_index"] = float(concordance_index_censored(event, time, risk_score)[0])
    except Exception as e:  # fall back to a lifelines c-index if sksurv is unavailable
        try:
            from lifelines.utils import concordance_index
            out["c_index"] = float(concordance_index(time, -risk_score, event))
        except Exception:
            out["c_index"], out["c_index_error"] = None, str(e)
    if pred_time is not None and event.sum() > 0:
        pred_time = np.asarray(pred_time, dtype=float)
        out["rul_mae_days"] = float(np.mean(np.abs(pred_time[event] - time[event])))
        out["rul_median_true_days"] = float(np.median(time[event]))
    return out
