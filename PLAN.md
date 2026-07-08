# DriveGuard - Predictive Maintenance & Remaining Useful Life for Data-Center Drives

**Goal:** A resume-grade, end-to-end ML project on **real production data** (Backblaze Drive Stats),
with a **full production MLOps stack**. Targets DS / AI / ML / GenAI roles.

**Working rule (non-negotiable):** No assumptions. Every choice - features, models, hyperparameters,
thresholds, imbalance strategy - is decided by **empirical benchmarking**, not intuition. Try many
options and compare.

---

## 1. Dataset (verified real - 2026-07-08)

**Backblaze Drive Stats** - https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
(Kaggle mirror: https://www.kaggle.com/datasets/backblaze/hard-drive-test-data)

- Real daily SMART snapshots from ~320,000+ live data-center drives, 2013-present (557M+ drive-days).
- One row = one drive-day. Columns: `date, serial_number, model, capacity_bytes, failure`
  + 90 columns (raw + normalized) for 45 SMART attributes.
- Label: `failure=1` only on a drive's **last operational day** -> extreme class imbalance (~0.1-1%).
- Size: ~1.2 GB compressed / ~12 GB uncompressed **per quarter**. We scope to **4-6 recent quarters**.
- License: free; must cite Backblaze; cannot resell raw data.
- Backblaze-flagged strong predictors: SMART 5, 187, 188, 197, 198 (treat as priors to TEST, not assume).

**Alternatives (confirmed SIMULATED - used only as optional secondary benchmark):**
NASA C-MAPSS, UCI AI4I 2020.

---

## 2. Problem framings (build both)

1. **Classification** - "Will this drive fail within the next N days?" Test N  in  {7, 15, 30}. Deployable alerting model.
2. **Survival / Remaining Useful Life (RUL)** - "How many days of life remain?" The portfolio differentiator.

---

## 3. Pipeline phases

### Phase 0 - Data engineering & scoping
- Ingest 4-6 recent quarters -> partitioned **Parquet** (columnar, ~10x smaller).
- Reconstruct each drive's full **life history** by joining on `serial_number` (failure appears once).
- Imbalance strategies to benchmark: class weights, focal loss, SMOTE/ADASYN, undersampling, threshold tuning.

### Phase 1 - Feature engineering (test multiple sets)
- Raw SMART, normalized SMART, **rolling stats** (7/14/30-day mean, slope, std), deltas.
- Drive age, power-on hours, model, capacity, per-model baselines.
- Feature sets to compare: Backblaze "big 5" vs full 90 vs automated selection (mutual info / SHAP).

### Phase 2 - Model lineup (broad benchmark; Optuna for HPO; MLflow tracking)
| Framing        | Models |
|----------------|--------|
| Classification | LogReg baseline; LightGBM / XGBoost / CatBoost; RandomForest; TabNet / FT-Transformer; autoencoder anomaly score |
| Sequence       | LSTM / GRU / 1D-CNN / Temporal Transformer on rolling SMART windows |
| Survival / RUL | Cox PH; Random Survival Forest; DeepSurv; Weibull AFT |

### Phase 3 - Evaluation (rigor is the standout)
- Metrics: **PR-AUC** (primary), recall @ fixed low false-alarm rate, F-beta (recall-weighted),
  Brier score / calibration; RUL: concordance index, MAE-on-RUL.
- **Time-based split** (train older quarters -> test newer). Never random split (leakage).
- Cost-sensitive analysis (missed-failure vs false-alarm curve). SHAP explainability.

### Phase 4 - MLOps (Full production stack)
- Orchestration: **Airflow or Prefect** (scheduled batch scoring + retrain jobs).
- Serving: **FastAPI + Docker** - returns failure risk + RUL + top SHAP reasons.
- Registry / versioning: **MLflow Model Registry** + **DVC** for data lineage.
- Monitoring / drift: **Evidently AI** (feature drift as fleet ages; performance drift).
- Auto-retrain: triggered on drift or PR-AUC degradation threshold.
- CI/CD: **GitHub Actions** (tests, lint, build, model-validation gate).
- Dashboard: **Streamlit** - fleet health, flagged drives, drift over time, model comparison.

---

## 4. Tech stack
Python, Polars/Pandas + Parquet, scikit-learn, LightGBM/XGBoost/CatBoost, PyTorch,
lifelines/scikit-survival, Optuna, MLflow, DVC, FastAPI, Docker, Airflow/Prefect, Evidently,
Streamlit, GitHub Actions.

## 5. Milestones
1. Data pipeline (ingest -> Parquet -> life-history -> EDA)
2. Baseline classifier + honest time-split eval
3. Model bake-off (full lineup + Optuna + MLflow)
4. Survival / RUL models
5. Serving (FastAPI + Docker + SHAP)
6. MLOps loop (monitoring, drift, auto-retrain, CI/CD)
7. Dashboard + README / writeup
