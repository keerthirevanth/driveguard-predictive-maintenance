"""Phase 3 - imbalance-appropriate evaluation.

Primary: PR-AUC. Secondary: recall @ fixed low false-alarm rate, F-beta (recall-weighted),
Brier/calibration; survival: concordance index, RUL MAE. Time-based split only.

TODO(milestone-2+): implement. Scaffold only.
"""
from __future__ import annotations


def evaluate_classification(y_true, y_score, cfg: dict) -> dict:
    """Compute PR-AUC, recall@low-FPR, F-beta, Brier, and a cost-sensitive curve."""
    raise NotImplementedError("Milestone 2: implement classification metrics.")


def evaluate_survival(model, data, cfg: dict) -> dict:
    """Compute concordance index and RUL MAE."""
    raise NotImplementedError("Milestone 4: implement survival metrics.")
