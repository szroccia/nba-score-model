# Evaluation

Validation seasons: 2023-24, 2024-25. Each month is predicted by a model refit on every game before that month (rolling-origin cross-validation).
The test season (2025-26) is locked and not used here; see test_log.md.
Settings are ranked by margin_mae, then total_mae, then abs_total_bias (fixed before running).

## Current settings

Settings: alpha 10.0, playoff_games include, half_life 12, prior_season_weight 0.4, shrink_games 4, feature_version 4.

| Model | Games | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| Ridge regression | 2621 | 10.95 | 14.59 | 66.7% | -0.34 |

## Against the Vegas closing line

On the 2620 validation games with an ESPN closing line (100% of them):

| Month | Games | Margin MAE, model | Margin MAE, Vegas | Total MAE, model | Total MAE, Vegas | Winner %, model | Winner %, Vegas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10 | 54 | 10.19 | 9.57 | 13.35 | 13.52 | 63.0% | 68.5% |
| 2023-11 | 219 | 10.41 | 10.15 | 14.07 | 13.92 | 68.0% | 66.7% |
| 2023-12 | 207 | 10.28 | 10.05 | 13.67 | 13.06 | 68.6% | 70.5% |
| 2024-01 | 231 | 11.23 | 11.29 | 16.80 | 16.23 | 66.2% | 68.4% |
| 2024-02 | 174 | 12.00 | 11.03 | 14.33 | 13.81 | 63.2% | 70.7% |
| 2024-03 | 230 | 11.03 | 10.55 | 14.00 | 14.17 | 63.9% | 66.5% |
| 2024-04 | 151 | 11.11 | 10.61 | 15.00 | 14.18 | 66.9% | 68.9% |
| 2024-05 | 41 | 13.01 | 12.91 | 12.11 | 11.87 | 61.0% | 63.4% |
| 2024-06 | 5 | 13.94 | 14.10 | 10.55 | 12.50 | 80.0% | 80.0% |
| 2024-10 | 71 | 10.02 | 10.23 | 15.90 | 15.19 | 60.6% | 62.0% |
| 2024-11 | 220 | 9.74 | 9.85 | 15.29 | 14.79 | 71.4% | 68.6% |
| 2024-12 | 189 | 10.67 | 10.38 | 15.24 | 14.76 | 67.7% | 68.3% |
| 2025-01 | 225 | 11.80 | 11.30 | 13.70 | 13.06 | 64.0% | 69.3% |
| 2025-02 | 176 | 10.31 | 10.43 | 14.63 | 13.84 | 67.0% | 68.8% |
| 2025-03 | 238 | 10.96 | 10.67 | 13.80 | 13.60 | 71.4% | 71.0% |
| 2025-04 | 143 | 11.44 | 10.36 | 15.24 | 14.49 | 69.2% | 74.8% |
| 2025-05 | 39 | 14.68 | 13.68 | 15.09 | 15.53 | 51.3% | 53.8% |
| 2025-06 | 7 | 9.15 | 8.64 | 13.38 | 10.93 | 57.1% | 57.1% |
| All | 2620 | 10.94 | 10.62 | 14.59 | 14.15 | 66.7% | 68.7% |

## Complexity curve (ridge alpha)

Larger alpha shrinks the coefficients toward zero (a simpler model). Training error is in-sample on each fold's training games.

| Alpha | Margin MAE, validation | Margin MAE, training | Total MAE, validation | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 10.95 | 10.52 | 14.59 | 66.8% | -0.34 |
| 0.1 | 10.95 | 10.52 | 14.59 | 66.8% | -0.34 |
| 1 | 10.95 | 10.52 | 14.59 | 66.8% | -0.34 |
| 10 | 10.95 | 10.52 | 14.59 | 66.7% | -0.34 |
| 100 | 10.95 | 10.52 | 14.59 | 66.6% | -0.35 |
| 1000 | 10.98 | 10.56 | 14.60 | 66.0% | -0.36 |
| 10000 | 11.14 | 10.72 | 14.65 | 65.6% | -0.29 |
| 100000 | 11.63 | 11.04 | 15.09 | 65.3% | +0.30 |

Best on validation: alpha 10 (config.RIDGE_ALPHA is 10; changing it is a modeling decision, so it is not changed automatically).

## Learning curve

Each fit uses only the most recent N team-games of history; scored on 2024-25 months that have at least the largest N available.

| Training team-games | Margin MAE, validation | Margin MAE, training | Validation games |
| --- | --- | --- | --- |
| 100 | 11.31 | 11.46 | 1309 |
| 250 | 11.06 | 11.40 | 1309 |
| 500 | 11.00 | 11.11 | 1309 |
| 1000 | 10.92 | 10.94 | 1309 |
| 2000 | 10.87 | 10.98 | 1309 |
| 4000 | 10.88 | 10.86 | 1309 |

## Playoff games in ratings

Same folds for each setting. October rows are the first month of each season, when ratings lean most on last season's games.

| Setting | Games | Margin MAE | Total MAE | Winner % | Total bias | October total bias |
| --- | --- | --- | --- | --- | --- | --- |
| playoffs include | 2621 | 10.95 | 14.59 | 66.7% | -0.34 | -0.33 |
| playoffs exclude | 2621 | 10.93 | 14.72 | 66.3% | -0.33 | +1.63 |
| playoffs relative | 2621 | 10.95 | 14.76 | 66.8% | -0.07 | +2.16 |
