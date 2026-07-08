"""Phase 4 - orchestration (Prefect flows; Airflow DAG optional via docker-compose).

Flows: scheduled batch scoring, drift check, conditional retrain.

TODO(milestone-6): implement Prefect flows wiring ingest -> features -> score -> drift -> retrain.
"""
from __future__ import annotations


def scoring_flow():
    """Daily batch scoring flow."""
    raise NotImplementedError("Milestone 6: implement Prefect scoring flow.")
