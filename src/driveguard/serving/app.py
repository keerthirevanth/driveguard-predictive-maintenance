"""Phase 4 - FastAPI serving. Returns failure risk + RUL + top SHAP reasons.

Run (once implemented):  uvicorn driveguard.serving.app:app --reload

TODO(milestone-5): load model from MLflow registry, implement /predict with SHAP.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DriveGuard", version="0.1.0")


class DriveSnapshot(BaseModel):
    """One drive's current SMART snapshot (subset shown; extend per feature set)."""

    serial_number: str
    model: str
    capacity_bytes: int
    smart_5_raw: float | None = None
    smart_187_raw: float | None = None
    smart_197_raw: float | None = None
    smart_198_raw: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/predict")
def predict(snapshot: DriveSnapshot) -> dict:
    """Return failure risk, RUL estimate, and top contributing SMART features."""
    raise NotImplementedError("Milestone 5: load registry model + SHAP explanation.")
