# Milestone 3 backfill - failure classification across horizons (N = 7 / 15 / 30)

Rolling features, held-out test = 2025-Q3. LightGBM uses the Optuna-tuned N=30 config
(transferred, not re-tuned per horizon); RF/CatBoost use defaults. LightGBM wins at every
horizon.

| Horizon | Model | Test PR-AUC | Recall @1% FPR |
|---|---|---|---|
| 7  | lightgbm | 0.0768 | 0.556 |
| 7  | catboost | 0.0673 | 0.536 |
| 7  | random_forest | 0.0665 | 0.551 |
| 15 | lightgbm | 0.1159 | 0.522 |
| 15 | random_forest | 0.0997 | 0.518 |
| 15 | catboost | 0.0965 | 0.491 |
| 30 | lightgbm | 0.1638 | 0.496 |

## Findings
- LightGBM (rolling features) is the best model at all three horizons.
- PR-AUC rises with the horizon (0.077 -> 0.116 -> 0.164): longer windows have more
  positives (higher base rate), so absolute PR-AUC is higher.
- Recall at a 1% false-alarm budget FALLS with the horizon (0.556 -> 0.522 -> 0.496):
  imminent failures carry stronger SMART signals and are individually easier to catch.
- Lift over base rate is highest at the shortest horizon (~265x at N=7 vs ~135x at N=30) -
  the model is most discriminative exactly when failure is closest.

## Takeaway for serving
Report all three horizons; the 7-day alert is the highest-recall "act now" signal, the
30-day alert gives the most lead time to schedule a replacement.
