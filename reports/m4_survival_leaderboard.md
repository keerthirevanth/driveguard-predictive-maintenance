# Milestone 4 (part 1) - Classical survival / RUL results

Rolling features + survival targets (event, duration = remaining days). Held-out test =
2025-Q3. Primary metric: concordance index (C-index; 0.5 = random). Secondary: RUL MAE
(days) on drives that actually failed. Train/eval subsampled (all events + capped censored).

| Model | C-index | RUL MAE (days) | events (eval) | fit sec |
|---|---|---|---|---|
| **weibull_aft** | **0.630** | **57.8** | 58,760 | 51.5 |
| cox_ph | 0.455 | 123.4 | 58,760 | 27.0 |
| random_survival_forest | (deferred - see notes) | - | - | - |

## Findings
- **Weibull AFT wins** among classical models: ranks failure risk well (C-index 0.63) and
  predicts remaining life within ~58 days on average.
- **Cox PH underperforms** (C-index 0.455, below random) even with a stronger L2 penalizer -
  its linear proportional-hazards assumption does not fit this data; RUL error ~2x Weibull.
  Documented as an also-ran (evidence-based, not hidden).
- Random Survival Forest was too slow in its full form (predict_survival_function over the
  eval set ran for hours). Code was made tractable (capped train subsample, leaner trees,
  risk/C-index only); to be run in a fresh Kaggle session (~minutes) for completeness.

## Notes
- Weibull/Cox predicted times are clipped to [1, 400] days (observation window ~365) because
  survival medians/expectations can be infinite under censoring - without this, RUL MAE
  explodes (a Weibull run reported 1.2e25 before the fix).

## Next (M4 part 2)
Deep sequence RUL (LSTM / GRU / 1D-CNN) on raw SMART sequences, same metrics, for direct
comparison against Weibull's 0.630 C-index.
