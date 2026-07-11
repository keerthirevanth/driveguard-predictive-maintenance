"""Milestone 5 - FastAPI serving.

Loads the production artifacts (models_store/) and exposes /predict: given a drive's recent
daily SMART readings, it returns
  - failure_risk        P(fails within 30 days) from the tuned LightGBM
  - rul_days            estimated remaining useful life (Weibull AFT)
  - top_reasons         the SMART features pushing the risk up/down (SHAP)

Run:  uvicorn driveguard.serving.app:app --reload
Build artifacts first with:  python -m driveguard.models.finalize
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from driveguard.config import PROJECT_ROOT
from driveguard.features.build_features import SMART_BIG5
from driveguard.features.rolling import WINDOWS, feature_columns

STORE = PROJECT_ROOT / "models_store"
app = FastAPI(title="DriveGuard", version="1.0.0")
_state: dict = {}


def _load():
    import lightgbm as lgb
    meta = json.loads((STORE / "serving_meta.json").read_text())
    booster = lgb.Booster(model_file=str(STORE / "classifier.txt"))
    with open(STORE / "rul_weibull.pkl", "rb") as f:
        aft = pickle.load(f)
    try:
        import shap
        explainer = shap.TreeExplainer(booster)
    except Exception:
        explainer = None
    _state.update(meta=meta, booster=booster, aft=aft, explainer=explainer)


@app.on_event("startup")
def startup():
    if (STORE / "serving_meta.json").exists():
        _load()


class DayReading(BaseModel):
    smart_5_raw: float | None = None
    smart_187_raw: float | None = None
    smart_188_raw: float | None = None
    smart_197_raw: float | None = None
    smart_198_raw: float | None = None


class PredictRequest(BaseModel):
    model: str = Field(..., description="drive model string, e.g. 'ST12000NM0007'")
    capacity_bytes: int
    drive_age_days: int
    history: list[DayReading] = Field(..., description="daily readings, oldest first, up to 30 days")


def _feature_vector(req: PredictRequest, meta: dict) -> np.ndarray:
    # per-SMART arrays, forward-filled then 0 (same as training)
    arrs = {}
    for c in SMART_BIG5:
        vals = [getattr(d, c) for d in req.history]
        a = np.array([np.nan if v is None else v for v in vals], dtype=np.float64)
        # forward fill
        last = 0.0
        for i in range(len(a)):
            if np.isnan(a[i]):
                a[i] = last
            else:
                last = a[i]
        arrs[c] = a

    def roll(a, w, fn):
        seg = a[-w:] if len(a) >= 1 else a
        return float(fn(seg)) if len(seg) else 0.0

    cur = {c: (arrs[c][-1] if len(arrs[c]) else 0.0) for c in SMART_BIG5}
    rmean = {(c, w): roll(arrs[c], w, np.mean) for c in SMART_BIG5 for w in WINDOWS}
    rstd = {(c, w): roll(arrs[c], w, np.std) for c in SMART_BIG5 for w in WINDOWS}

    code = meta["model_code_map"].get(req.model, -1)
    feat = {
        "model": float(code),
        "capacity_gb": req.capacity_bytes / 1e9,
        "drive_age_days": float(req.drive_age_days),
    }
    for c in SMART_BIG5:
        feat[c] = cur[c]
    for c in SMART_BIG5:
        for w in WINDOWS:
            feat[f"{c}_rmean_{w}"] = rmean[(c, w)]
    for c in SMART_BIG5:
        for w in WINDOWS:
            feat[f"{c}_rstd_{w}"] = rstd[(c, w)]
    for c in SMART_BIG5:
        feat[f"{c}_dev30"] = cur[c] - rmean[(c, 30)]

    return np.array([feat[k] for k in feature_columns()], dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "booster" in _state, "version": app.version}


@app.post("/predict")
def predict(req: PredictRequest):
    if "booster" not in _state:
        raise HTTPException(503, "models not loaded - run `python -m driveguard.models.finalize`")
    meta = _state["meta"]
    x = _feature_vector(req, meta).reshape(1, -1)

    risk = float(_state["booster"].predict(x)[0])
    import pandas as pd
    row = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    rul = float(np.clip(_state["aft"].predict_median(row).to_numpy()[0], 1, 400))

    reasons = []
    if _state["explainer"] is not None:
        sv = _state["explainer"].shap_values(x)
        sv = sv[1] if isinstance(sv, list) else sv
        contrib = sv[0]
        order = np.argsort(-np.abs(contrib))[:5]
        cols = feature_columns()
        reasons = [{"feature": cols[i], "shap": round(float(contrib[i]), 4),
                    "value": round(float(x[0, i]), 3)} for i in order]

    return {
        "failure_risk": round(risk, 4),
        "will_fail_within_30d": bool(risk >= 0.5),
        "rul_days": round(rul, 1),
        "top_reasons": reasons,
    }
