"""Phase 0 - reconstruct each drive's full life history and build labels.

Because `failure=1` appears only on the last day, we join all snapshots per
`serial_number`, order by date, and derive:
  - remaining_useful_life_days (for survival/RUL framing)
  - will_fail_within_N_days    (for classification framing, per config horizons)

TODO(milestone-1): implement. Scaffold only.
"""
from __future__ import annotations


def build_labels(processed_dir, horizons_days: list[int]):
    """Attach RUL + will_fail_within_N labels to the per-drive-day table."""
    raise NotImplementedError("Milestone 1: implement label construction.")
