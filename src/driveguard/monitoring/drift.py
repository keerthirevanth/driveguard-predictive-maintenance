"""Phase 4 - drift monitoring + retrain trigger (Evidently).

Feature drift is organic here: SMART distributions shift as the fleet ages and new
drive models arrive. When feature drift or PR-AUC degradation crosses the configured
threshold, fire a retrain job.

TODO(milestone-6): implement Evidently reports + threshold-based retrain trigger.
"""
from __future__ import annotations


def check_drift(reference_df, current_df, cfg: dict) -> dict:
    """Return an Evidently drift summary and whether a retrain should trigger."""
    raise NotImplementedError("Milestone 6: implement drift detection.")
