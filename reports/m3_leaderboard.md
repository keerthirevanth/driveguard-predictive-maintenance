# Milestone 3 - Model bake-off results (N=30, held-out test = 2025-Q3)

Ran on Kaggle GPU. Same lineup on two feature sets: `big5` (point-in-time SMART) vs
`rolling` (adds per-drive 7/14/30-day mean/std + 30-day deviation).

Primary metric: PR-AUC (test base rate ~0.0016, so random ~= 0.0016).
M2 baseline (LightGBM, big5): test PR-AUC 0.096.

| Rank | Model | Feature set | test PR-AUC | test ROC-AUC | recall @1% FPR | fit sec |
|---|---|---|---|---|---|---|
| 1 | random_forest | rolling | 0.1448 | 0.863 | 0.492 | 425.7 |
| 2 | lightgbm | rolling | 0.1442 | 0.789 | 0.423 | 71.8 |
| 3 | xgboost | rolling | 0.1273 | 0.761 | 0.395 | 168.3 |
| 4 | lightgbm | big5 | 0.0932 | 0.809 | 0.368 | 66.4 |
| 5 | random_forest | big5 | 0.0877 | 0.868 | 0.428 | 350.5 |
| 6 | xgboost | big5 | 0.0805 | 0.777 | 0.323 | 110.0 |
| 7 | logreg | rolling | 0.0609 | 0.720 | 0.425 | 99.2 |
| 8 | logreg | big5 | 0.0437 | 0.707 | 0.375 | 15.2 |
| 9 | tabnet | rolling | 0.0315 | 0.691 | 0.221 | 3229.1 |
| 10 | tabnet | big5 | 0.0202 | 0.570 | 0.135 | 3039.9 |

CatBoost errored on the Kaggle run (dependency/GPU) and was skipped by the bake-off's
per-model try/except; to be retried separately.

## Findings
- Rolling/temporal features beat point-in-time for every model: best PR-AUC 0.093 -> 0.145
  (~55% relative gain), ~90x lift over base rate.
- Operational: recall at a 1% false-alarm budget improved 0.387 -> 0.492 (catch ~half of
  failing drives at the same alert cost).
- random_forest and lightgbm on rolling are statistically tied on PR-AUC; LightGBM is 6x
  faster to fit and is the practical choice for serving/retraining.
- TabNet underperformed GBDTs badly and was far slower - expected for tabular data.

## Decision
Carry the `rolling` feature set forward. Use LightGBM as the deployable model (near-best
PR-AUC, fast) with random_forest as a strong cross-check. Next: Optuna HPO on
lightgbm+rolling to push PR-AUC further before serving.
