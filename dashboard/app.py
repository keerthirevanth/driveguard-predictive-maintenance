"""DriveGuard dashboard (Streamlit): fleet health, flagged drives, drift, model leaderboard.

Run (once implemented):  streamlit run dashboard/app.py

TODO(milestone-7): wire to MLflow leaderboard, serving API, and Evidently drift reports.
"""
import streamlit as st

st.set_page_config(page_title="DriveGuard", layout="wide")
st.title("DriveGuard - Predictive Maintenance")
st.caption("Real Backblaze Drive Stats: failure risk, RUL, and drift monitoring")

st.info(
    "Dashboard placeholder. Milestone 7 connects this to the model registry, "
    "serving API, and drift reports."
)
