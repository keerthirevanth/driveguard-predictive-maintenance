# Milestone 4 - Survival / RUL results (complete)

Rolling features + survival targets (event, duration). Held-out test = 2025-Q3.
Primary metric: concordance index (C-index; 0.5 = random). Secondary: RUL MAE (days) on
drives that failed. Classical/forest/DeepSurv on tabular rolling features; sequence models
on raw SMART day-sequences (L=30) with a censoring-aware RUL loss.

## Tabular models (survival_rolling test set)
| Model | C-index | RUL MAE (days) | fit sec |
|---|---|---|---|
| random_survival_forest | 0.712 | n/a (risk score only) | 120.2 |
| deepsurv (neural Cox) | 0.663 | n/a (risk score only) | 17.6 |
| weibull_aft | 0.630 | 57.8 | 36.3 |
| cox_ph | 0.455 | 123.4 | 17.9 |

## Sequence models (sequence-window test set)
| Model | C-index | RUL MAE (days) | fit sec |
|---|---|---|---|
| lstm | 0.686 | 135.1 | 62.7 |
| cnn1d | 0.685 | 105.8 | 49.1 |
| gru | 0.634 | 125.7 | 60.4 |

## Findings
- Random Survival Forest is the best discriminator (C-index 0.712), ahead of DeepSurv,
  the sequence nets, and Weibull. Tree ensembles again dominate this tabular data.
- Weibull AFT gives the best point RUL estimate (58-day MAE) and is the only strong model
  that emits an absolute "days remaining" number (RSF/DeepSurv output relative risk only in
  these configs; sequence models predict RUL but with larger error, 106-135 days).
- Cox PH underperforms (C-index below random) - its linear proportional-hazards assumption
  does not fit this data.

## Caveats
- `RUL MAE = n/a` for RSF/DeepSurv: they produce a risk score, not a time; no MAE defined.
- Tabular vs sequence C-index used different eval subsamples (separate notebooks), so the
  cross-group comparison is directional; within each group it is clean.

## Decision (for serving, M5)
- **Risk ranking**: Random Survival Forest is the strongest (0.712), but for a lightweight,
  torch-free served RUL that emits an actual days-remaining number, use **Weibull AFT**
  (58-day MAE, C-index 0.630, fast CPU). RSF is cited as the best pure risk-ranker.
- Failure classifier stays the tuned LightGBM from M3 (PR-AUC 0.164).
- So serving returns: failure risk (LightGBM) + RUL days (Weibull) + SHAP reasons.
