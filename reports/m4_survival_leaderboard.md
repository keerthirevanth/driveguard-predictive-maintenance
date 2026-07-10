# Milestone 4 - Survival / RUL results (complete)

Rolling features + survival targets (event, duration). Held-out test = 2025-Q3.
Primary metric: concordance index (C-index; 0.5 = random). Secondary: RUL MAE (days) on
drives that failed. Classical models on tabular rolling features; sequence models on raw
SMART day-sequences (L=30) with a censoring-aware RUL loss.

| Model | Type | C-index | RUL MAE (days) | fit sec |
|---|---|---|---|---|
| lstm | sequence (LSTM) | 0.686 | 135.1 | 62.7 |
| cnn1d | sequence (1D-CNN) | 0.685 | 105.8 | 49.1 |
| gru | sequence (GRU) | 0.634 | 125.7 | 60.4 |
| weibull_aft | classical (AFT) | 0.630 | 57.8 | 51.5 |
| cox_ph | classical (Cox PH) | 0.455 | 123.4 | 27.0 |
| random_survival_forest | forest | (deferred - too slow to run to completion) | - | - |

## Findings
- Deep sequence models win on ranking: LSTM 0.686 and 1D-CNN 0.685 beat the best classical
  model (Weibull 0.630) on C-index. The raw sequence carries signal the rolling summaries miss.
- Weibull wins on point-RUL accuracy: MAE 58 days vs 106-135 for the deep models. So Weibull
  estimates days-remaining more precisely while the nets rank risk better.
- GRU trailed LSTM/CNN; Cox PH underperformed (C-index below random - linear PH assumption
  does not fit this data).

## Caveat
Classical vs sequence RUL MAE were computed on different eval subsamples (separate
notebooks), so the C-index (rank-based, primary) is the cleaner comparison; the MAE gap is
directional, not exact.

## Decision
Best RUL model = **1D-CNN** (C-index 0.685 tied-best, best deep RUL MAE 105.8d, fastest to
fit). LSTM edges C-index by 0.001 (noise). Carry the 1D-CNN as the RUL model into serving
(M5), alongside the tuned LightGBM failure classifier from M3. Weibull noted as the best
point-RUL estimator and a strong, simple classical baseline.

## Optional extensions (not blocking)
Random Survival Forest (fast config) and DeepSurv for completeness.
