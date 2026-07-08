"""Phase 1 - feature engineering. Benchmark multiple feature sets (config-driven).

Sets to compare: big5 SMART | full 90 cols | auto-selected (mutual info / SHAP).
Temporal features: rolling mean/std/slope over 7/14/30 days, deltas, drive age.

TODO(milestone-2/3): implement. Scaffold only.
"""
from __future__ import annotations


def make_features(df, feature_set: str, cfg: dict):
    """Return an engineered feature frame for the requested feature_set."""
    raise NotImplementedError("Milestone 2: implement feature engineering.")
