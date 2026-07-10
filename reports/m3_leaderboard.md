# Milestone 3 - Model bake-off results (N=30, held-out test = 2025-Q3)

Ran on Kaggle GPU. Same lineup on two feature sets: `big5` (point-in-time SMART) vs
`rolling` (adds per-drive 7/14/30-day mean/std + 30-day deviation).

Primary metric: PR-AUC (test base rate ~0.0016, so random ~= 0.0016).
M2 baseline (LightGBM, big5): test PR-AUC 0.096.

| Rank | Model | Feature set | test PR-AUC | test ROC-AUC | recall @1% FPR | fit sec |
|---|---|---|---|---|---|---|
| 1 | random_forest | rolling | 0.1448 | 0.863 | 0.492 | 425.7 |
| 2 | lightgbm | rolling | 0.1442 | 0.789 | 0.423 | 71.8 |
| 2 | catboost | rolling | 0.1442 | 0.855 | 0.461 | 1209.7 |
| 4 | xgboost | rolling | 0.1273 | 0.761 | 0.395 | 168.3 |
| 5 | lightgbm | big5 | 0.0932 | 0.809 | 0.368 | 66.4 |
| 6 | catboost | big5 | 0.0914 | 0.845 | 0.401 | 1034.1 |
| 7 | random_forest | big5 | 0.0877 | 0.868 | 0.428 | 350.5 |
| 8 | xgboost | big5 | 0.0805 | 0.777 | 0.323 | 110.0 |
| 9 | logreg | rolling | 0.0609 | 0.720 | 0.425 | 99.2 |
| 10 | logreg | big5 | 0.0437 | 0.707 | 0.375 | 15.2 |
| 11 | tabnet | rolling | 0.0315 | 0.691 | 0.221 | 3229.1 |
| 12 | tabnet | big5 | 0.0202 | 0.570 | 0.135 | 3039.9 |

## Findings
- Rolling/temporal features beat point-in-time for every model: best PR-AUC 0.093 -> 0.145
  (~55% relative gain), ~90x lift over base rate.
- Operational: recall at a 1% false-alarm budget improved 0.387 -> 0.49 (catch ~half of
  failing drives at the same alert cost).
- Top three (rolling) are a statistical tie on PR-AUC: random_forest 0.1448, lightgbm 0.1442,
  catboost 0.1442. random_forest leads on recall@1%FPR (0.492); catboost has the best
  ROC/recall balance; lightgbm is 6-17x faster to fit.
- TabNet underperformed GBDTs badly and was far slower - expected for tabular data.

## Decision
Carry the `rolling` feature set forward. No untuned winner among random_forest / lightgbm /
catboost, so run Optuna HPO on all three (evidence-first) and pick by tuned test PR-AUC +
recall@1%FPR, with fit/inference speed as a tiebreaker only. HPO objective on validation
(2025-Q2); test (2025-Q3) stays the held-out judge. logreg/tabnet excluded from HPO
(clearly non-competitive - that pruning is evidence-based).
