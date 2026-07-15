"""DriveGuard dashboard - fleet health, live predictions, drift, and model results.

Run:  streamlit run dashboard/app.py
Requires the production models (python -m driveguard.models.finalize) and, for the fleet
view, the local Parquet data.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="DriveGuard", page_icon=":material/hard_drive:", layout="wide")

ALERT_COLOR = {"critical": "red", "warning": "orange", "watch": "blue", "ok": "green"}


@st.cache_resource
def load_models():
    from driveguard.serving import app as api
    api._load()
    return api


@st.cache_data(show_spinner="Scoring a sample of the current fleet...")
def load_fleet(n_drives: int = 2500) -> pd.DataFrame:
    from driveguard.config import load_config
    from driveguard.monitoring.fleet import score_fleet
    return score_fleet(load_config(), ROOT, n_drives=n_drives).to_pandas()


@st.cache_data
def load_drift() -> dict | None:
    p = ROOT / "reports" / "drift_report.json"
    return json.loads(p.read_text()) if p.exists() else None


def alert_badge(level: str) -> str:
    return f":{ALERT_COLOR.get(level, 'gray')}[**{level.upper()}**]"


st.title(":material/hard_drive: DriveGuard")
st.caption("Predictive maintenance for data-center drives - failure risk, remaining useful "
           "life, and drift monitoring on real Backblaze SMART data.")

tab_predict, tab_fleet, tab_drift, tab_results = st.tabs(
    ["Predict a drive", "Fleet health", "Drift monitor", "Model results"])


# ---------------------------------------------------------------- Predict a drive
with tab_predict:
    api = load_models()
    meta = api._state["meta"]
    models = sorted(meta["model_code_map"].keys())

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Drive under test**")
            model = st.selectbox("Drive model", models,
                                 index=models.index("ST12000NM0007") if "ST12000NM0007" in models else 0)
            capacity_tb = st.slider("Capacity (TB)", 1, 24, 12)
            age_years = st.slider("Drive age (years)", 0.0, 6.0, 3.5, 0.5)
            st.markdown("**Recent SMART readings** (current values; a 30-day rising trend is simulated)")
            s5 = st.slider("SMART 5 - reallocated sectors", 0, 300, 0)
            s197 = st.slider("SMART 197 - pending sectors", 0, 200, 0)
            s187 = st.slider("SMART 187 - reported uncorrectable", 0, 100, 0)
            s198 = st.slider("SMART 198 - offline uncorrectable", 0, 100, 0)

    from driveguard.serving.app import DayReading, PredictRequest, _feature_vector, score
    hist = []
    for i in range(30):
        f = i / 29.0
        hist.append(DayReading(smart_5_raw=s5 * f, smart_197_raw=s197 * f,
                               smart_187_raw=s187 * f, smart_198_raw=s198 * f))
    req = PredictRequest(model=model, capacity_bytes=int(capacity_tb * 1e12),
                         drive_age_days=int(age_years * 365), history=hist)
    res = score(_feature_vector(req, meta))

    with right:
        with st.container(horizontal=True):
            st.metric("Failure probability (30d)", f"{res['failure_probability_30d']*100:.1f}%", border=True)
            st.metric("Remaining useful life", f"{res['rul_days']:.0f} days", border=True)
        with st.container(border=True):
            st.markdown(f"Alert level: {alert_badge(res['alert_level'])} "
                        f"&nbsp;&nbsp; confidence: **{res['confidence']}**")
            if res["note"]:
                st.warning(res["note"])
        with st.container(border=True):
            st.markdown("**Why** - top contributing signals (SHAP)")
            rdf = pd.DataFrame(res["top_reasons"])
            if not rdf.empty:
                rdf["direction"] = rdf["shap"].apply(lambda v: "increases risk" if v > 0 else "lowers risk")
                st.dataframe(rdf[["feature", "value", "shap", "direction"]], hide_index=True,
                             width="stretch")
                st.bar_chart(rdf.set_index("feature")["shap"], horizontal=True)


# ---------------------------------------------------------------- Fleet health
def render_fleet(fleet):
    counts = fleet["alert_level"].value_counts().to_dict()
    with st.container(horizontal=True):
        st.metric("Drives scored", f"{len(fleet):,}", border=True)
        st.metric("Critical", counts.get("critical", 0), border=True)
        st.metric("Warning", counts.get("warning", 0), border=True)
        st.metric("Watch", counts.get("watch", 0), border=True)
        st.metric("OK", counts.get("ok", 0), border=True)
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True):
        st.markdown("**Failure-probability distribution**")
        h = pd.cut(fleet["failure_probability"], bins=[0, .05, .1, .25, .5, 1.0]).value_counts().sort_index()
        h.index = ["0-5%", "5-10%", "10-25%", "25-50%", "50-100%"]
        st.bar_chart(h)
    with c2, st.container(border=True):
        st.markdown("**Highest-risk drives**")
        top = fleet.head(12)[["model", "failure_probability", "rul_days", "alert_level"]]
        st.dataframe(top, hide_index=True, width="stretch",
                     column_config={"failure_probability": st.column_config.ProgressColumn(
                         "risk", min_value=0, max_value=1, format="%.2f")})


with tab_fleet:
    st.markdown("Scored sample of drives from the newest quarter (2025-Q3).")
    if st.button("Score a live fleet sample", type="primary") or st.session_state.get("fleet_loaded"):
        st.session_state["fleet_loaded"] = True
        render_fleet(load_fleet())
    else:
        st.info("Scores ~2,500 real drives with the production models. Takes ~30s the first "
                "time (builds rolling features), then it is cached.")


# ---------------------------------------------------------------- Drift monitor
with tab_drift:
    drift = load_drift()
    if not drift:
        st.info("No drift report yet. Run `python -m driveguard.monitoring.drift`.")
    else:
        d = drift["drift"]
        dec = drift["decision"]
        with st.container(horizontal=True):
            st.metric("Features drifted", f"{d['n_drifted']} / {d['n_features']}", border=True)
            st.metric("Drift share", f"{d['drifted_share']*100:.0f}%", border=True)
            st.metric("Retrain needed", "Yes" if dec["retrain"] else "No", border=True)
        st.caption(f"reference: {', '.join(drift['reference_quarters'])}  ->  "
                   f"current: {', '.join(drift['current_quarters'])}")
        with st.container(border=True):
            st.markdown("**Per-feature drift** (PSI: <0.1 stable, 0.1-0.2 moderate, >0.2 significant)")
            fdf = pd.DataFrame(d["features"])
            st.dataframe(fdf, hide_index=True, width="stretch")
        if dec["reasons"]:
            for r in dec["reasons"]:
                st.warning(r)
        else:
            st.success("Drift is within tolerance - no retrain triggered. "
                       "(The drive-model mix shifted, but failure-signal features are stable.)")


# ---------------------------------------------------------------- Model results
with tab_results:
    st.markdown("**Failure classification** - held-out test (2025-Q3), tuned LightGBM on rolling features")
    st.dataframe(pd.DataFrame({
        "horizon": ["7-day", "15-day", "30-day"],
        "PR-AUC": [0.077, 0.116, 0.164],
        "recall @1% false-alarm": [0.556, 0.522, 0.496],
        "lift over base rate": ["~265x", "~193x", "~135x"],
    }), hide_index=True, width="stretch")

    st.markdown("**Remaining-useful-life models** - concordance index (ranking) and RUL error")
    st.dataframe(pd.DataFrame({
        "model": ["Random Survival Forest", "LSTM", "1D-CNN", "DeepSurv", "Weibull AFT", "Cox PH"],
        "type": ["forest", "sequence", "sequence", "neural-cox", "classical", "classical"],
        "C-index": [0.712, 0.686, 0.685, 0.662, 0.630, 0.455],
        "RUL MAE (days)": [94, 135, 106, None, 58, 123],
    }), hide_index=True, width="stretch")
    st.caption("Served models: tuned LightGBM (failure risk) + Weibull AFT (RUL days). "
               "RSF is the best risk-ranker; Weibull the best days-estimator.")
