# Evaluation

Validation seasons: 2023-24, 2024-25. Each month is predicted by a model refit on every game before that month (rolling-origin cross-validation).
The test season (2025-26) is locked and not used here; see test_log.md.
Settings are ranked by margin_mae, then total_mae, then abs_total_bias (fixed before running).

## Current settings

Settings: alpha 10.0, feature_version 5, playoff_games include, half_life 12, prior_season_weight 0.4, shrink_games 4, opponent_adjust vs average team, replacement_obpm -1.5, replacement_dbpm -0.5, bpm_prior_minutes 500, minutes_half_life 5, rotation_minutes 20, max_rest_days 3, active_window_games 20, team_minutes 240.

| Model | Games | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| Ridge regression | 2621 | 10.92 | 14.56 | 67.1% | -0.41 |

## Against the Vegas closing line

On the 2620 validation games with an ESPN closing line (100% of them):

| Month | Games | Margin MAE, model | Margin MAE, Vegas | Total MAE, model | Total MAE, Vegas | Winner %, model | Winner %, Vegas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10 | 54 | 10.09 | 9.57 | 13.36 | 13.52 | 66.7% | 68.5% |
| 2023-11 | 219 | 10.34 | 10.15 | 13.99 | 13.92 | 66.2% | 66.7% |
| 2023-12 | 207 | 10.25 | 10.05 | 13.64 | 13.06 | 68.1% | 70.5% |
| 2024-01 | 231 | 11.21 | 11.29 | 16.59 | 16.23 | 66.2% | 68.4% |
| 2024-02 | 174 | 11.98 | 11.03 | 14.29 | 13.81 | 64.9% | 70.7% |
| 2024-03 | 230 | 11.01 | 10.55 | 14.02 | 14.17 | 65.7% | 66.5% |
| 2024-04 | 151 | 11.10 | 10.61 | 15.01 | 14.18 | 66.9% | 68.9% |
| 2024-05 | 41 | 12.86 | 12.91 | 11.55 | 11.87 | 61.0% | 63.4% |
| 2024-06 | 5 | 14.41 | 14.10 | 11.75 | 12.50 | 80.0% | 80.0% |
| 2024-10 | 71 | 9.92 | 10.23 | 15.91 | 15.19 | 62.0% | 62.0% |
| 2024-11 | 220 | 9.68 | 9.85 | 15.30 | 14.79 | 72.3% | 68.6% |
| 2024-12 | 189 | 10.72 | 10.38 | 15.39 | 14.76 | 67.7% | 68.3% |
| 2025-01 | 225 | 11.74 | 11.30 | 13.66 | 13.06 | 64.4% | 69.3% |
| 2025-02 | 176 | 10.36 | 10.43 | 14.62 | 13.84 | 68.8% | 68.8% |
| 2025-03 | 238 | 11.00 | 10.67 | 13.72 | 13.60 | 70.6% | 71.0% |
| 2025-04 | 143 | 11.34 | 10.36 | 15.29 | 14.49 | 70.6% | 74.8% |
| 2025-05 | 39 | 14.47 | 13.68 | 14.74 | 15.53 | 51.3% | 53.8% |
| 2025-06 | 7 | 9.36 | 8.64 | 13.17 | 10.93 | 57.1% | 57.1% |
| All | 2620 | 10.92 | 10.62 | 14.55 | 14.15 | 67.1% | 68.7% |

## Complexity curve (ridge alpha)

Larger alpha shrinks the coefficients toward zero (a simpler model). Training error is in-sample on each fold's training games.

| Alpha | Margin MAE, validation | Margin MAE, training | Total MAE, validation | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 10.92 | 10.51 | 14.56 | 67.0% | -0.41 |
| 0.1 | 10.92 | 10.51 | 14.56 | 67.0% | -0.41 |
| 1 | 10.92 | 10.51 | 14.56 | 67.0% | -0.41 |
| 10 | 10.92 | 10.51 | 14.56 | 67.1% | -0.41 |
| 100 | 10.92 | 10.52 | 14.56 | 67.0% | -0.42 |
| 1000 | 10.96 | 10.55 | 14.56 | 66.5% | -0.43 |
| 10000 | 11.13 | 10.71 | 14.62 | 65.4% | -0.35 |
| 100000 | 11.64 | 11.04 | 15.07 | 64.9% | +0.26 |

Best on validation: alpha 10 (config.RIDGE_ALPHA is 10; changing it is a modeling decision, so it is not changed automatically).

## Learning curve

Each fit uses only the most recent N team-games of history; scored on 2024-25 months that have at least the largest N available.

| Training team-games | Margin MAE, validation | Margin MAE, training | Validation games |
| --- | --- | --- | --- |
| 100 | 11.28 | 11.51 | 1309 |
| 250 | 11.04 | 11.38 | 1309 |
| 500 | 10.98 | 11.10 | 1309 |
| 1000 | 10.89 | 10.93 | 1309 |
| 2000 | 10.85 | 10.96 | 1309 |
| 4000 | 10.86 | 10.83 | 1309 |
