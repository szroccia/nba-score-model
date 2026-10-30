# Evaluation

Validation seasons: 2023-24, 2024-25. Each month is predicted by a model refit on every game before that month (rolling-origin cross-validation).
The test season (2025-26) is locked and not used here; see test_log.md.
Settings are ranked by margin_mae, then total_mae, then abs_total_bias (fixed before running).

## Current settings

Settings: alpha 10.0, four_factors False, feature_version 6, playoff_games include, half_life 24, prior_season_weight 0.4, shrink_games 4, opponent_adjust vs average team, replacement_obpm -2.25, replacement_dbpm -0.75, bpm_prior_minutes 500, minutes_half_life 12, rotation_minutes 20, max_rest_days 3, cusum [80, 4.0, 0.4], active_window_games 20, team_minutes 240.

| Model | Games | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| Ridge regression | 2621 | 10.86 | 14.64 | 67.1% | +0.06 |

## Against the Vegas closing line

On the 2620 validation games with an ESPN closing line (100% of them):

| Month | Games | Margin MAE, model | Margin MAE, Vegas | Total MAE, model | Total MAE, Vegas | Winner %, model | Winner %, Vegas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10 | 54 | 9.99 | 9.57 | 13.64 | 13.52 | 63.0% | 68.5% |
| 2023-11 | 219 | 10.25 | 10.15 | 14.00 | 13.92 | 66.7% | 66.7% |
| 2023-12 | 207 | 10.19 | 10.05 | 13.52 | 13.06 | 67.6% | 70.5% |
| 2024-01 | 231 | 11.22 | 11.29 | 16.63 | 16.23 | 66.7% | 68.4% |
| 2024-02 | 174 | 11.86 | 11.03 | 14.41 | 13.81 | 65.5% | 70.7% |
| 2024-03 | 230 | 10.89 | 10.55 | 14.40 | 14.17 | 67.0% | 66.5% |
| 2024-04 | 151 | 10.99 | 10.61 | 15.15 | 14.18 | 68.9% | 68.9% |
| 2024-05 | 41 | 13.22 | 12.91 | 12.24 | 11.87 | 53.7% | 63.4% |
| 2024-06 | 5 | 13.94 | 14.10 | 13.72 | 12.50 | 80.0% | 80.0% |
| 2024-10 | 71 | 9.87 | 10.23 | 15.90 | 15.19 | 62.0% | 62.0% |
| 2024-11 | 220 | 9.66 | 9.85 | 15.22 | 14.79 | 70.9% | 68.6% |
| 2024-12 | 189 | 10.63 | 10.38 | 15.23 | 14.76 | 66.7% | 68.3% |
| 2025-01 | 225 | 11.70 | 11.30 | 13.76 | 13.06 | 66.7% | 69.3% |
| 2025-02 | 176 | 10.41 | 10.43 | 14.66 | 13.84 | 67.0% | 68.8% |
| 2025-03 | 238 | 10.93 | 10.67 | 13.82 | 13.60 | 71.4% | 71.0% |
| 2025-04 | 143 | 11.17 | 10.36 | 15.32 | 14.49 | 71.3% | 74.8% |
| 2025-05 | 39 | 14.36 | 13.68 | 15.29 | 15.53 | 46.2% | 53.8% |
| 2025-06 | 7 | 8.96 | 8.64 | 14.02 | 10.93 | 57.1% | 57.1% |
| All | 2620 | 10.86 | 10.62 | 14.63 | 14.15 | 67.2% | 68.7% |

## Complexity curve (ridge alpha)

Larger alpha shrinks the coefficients toward zero (a simpler model). Training error is in-sample on each fold's training games.

| Alpha | Margin MAE, validation | Margin MAE, training | Total MAE, validation | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 10.86 | 10.44 | 14.64 | 67.2% | +0.06 |
| 0.1 | 10.86 | 10.44 | 14.64 | 67.2% | +0.06 |
| 1 | 10.86 | 10.44 | 14.64 | 67.2% | +0.06 |
| 10 | 10.86 | 10.44 | 14.64 | 67.1% | +0.06 |
| 100 | 10.86 | 10.44 | 14.63 | 67.0% | +0.06 |
| 1000 | 10.90 | 10.47 | 14.64 | 66.9% | +0.04 |
| 10000 | 11.08 | 10.64 | 14.71 | 65.8% | +0.06 |
| 100000 | 11.59 | 11.00 | 15.15 | 65.5% | +0.47 |

Best on validation: alpha 100 (config.RIDGE_ALPHA is 10; changing it is a modeling decision, so it is not changed automatically).

## Learning curve

Each fit uses only the most recent N team-games of history; scored on 2024-25 months that have at least the largest N available.

| Training team-games | Margin MAE, validation | Margin MAE, training | Validation games |
| --- | --- | --- | --- |
| 100 | 11.33 | 11.48 | 1309 |
| 250 | 11.02 | 11.38 | 1309 |
| 500 | 10.95 | 11.04 | 1309 |
| 1000 | 10.85 | 10.90 | 1309 |
| 2000 | 10.80 | 10.92 | 1309 |
| 4000 | 10.81 | 10.77 | 1309 |
