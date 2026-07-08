# DriveGuard

Predictive maintenance and Remaining Useful Life (RUL) prediction for data-center hard
drives, built on real production data
([Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data))
with a full production MLOps stack.

Predict which drives will fail before they do, and estimate how much life they have left,
then serve, monitor, and automatically retrain the models as a production system.

## Motivation

- Real telemetry, not simulated: 557M+ drive-days of SMART data from ~320k live drives.
- Two problem framings: failure classification (next N days) and survival/RUL regression.
- Extreme class imbalance (~0.1-1% failures) handled with PR-AUC and cost-sensitive thresholds.
- A complete MLOps loop: experiment tracking, model registry, drift monitoring, automated
  retraining, CI/CD, and a dashboard.

## Project status

Repository structure and configuration are in place. See [PLAN.md](PLAN.md) for the full
build plan and milestone breakdown.

## Repository layout

```
config/          YAML configs (data range, features, model params, thresholds)
data/            raw / interim / processed (git-ignored; tracked via DVC)
notebooks/       EDA and experiment notebooks
src/driveguard/
  data/          ingest, Parquet conversion, life-history reconstruction
  features/      feature engineering (SMART rolling stats, deltas, age)
  models/        classification, sequence, survival/RUL models and training
  evaluation/    time-split, metrics, calibration, SHAP
  serving/       FastAPI app
  monitoring/    Evidently drift and retrain triggers
  pipelines/     Prefect flows / Airflow DAGs
dashboard/       Streamlit app
docker/          Dockerfiles and docker-compose
tests/           pytest
.github/workflows/  CI/CD
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in Kaggle credentials if using the Kaggle mirror
```

## Data license

Data (c) Backblaze, used under their
[terms](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data): free to use
with attribution. Raw data is not redistributed in this repository.
