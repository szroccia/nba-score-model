# Evaluation

Validation seasons: 2023-24, 2024-25. Each month is predicted by a model refit on every game before that month (rolling-origin cross-validation).
The test season (2025-26) is locked and not used here; see test_log.md.
Settings are ranked by margin_mae, then total_mae, then abs_total_bias (fixed before running).

## Current settings

Settings: alpha 10.0, four_factors False, feature_version 6, playoff_games include, half_life 24, prior_season_weight 0.4, shrink_games 4, opponent_adjust vs average team, replacement_obpm -2.25, replacement_dbpm -0.75, bpm_prior_minutes 500, minutes_half_life 12, rotation_minutes 20, max_rest_days 3, active_window_games 20, team_minutes 240.

| Model | Games | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| Ridge regression | 2621 | 10.85 | 14.67 | 67.5% | +0.08 |

## Against the Vegas closing line

On the 2620 validation games with an ESPN closing line (100% of them):

| Month | Games | Margin MAE, model | Margin MAE, Vegas | Total MAE, model | Total MAE, Vegas | Winner %, model | Winner %, Vegas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-10 | 54 | 9.97 | 9.57 | 13.64 | 13.52 | 64.8% | 68.5% |
| 2023-11 | 219 | 10.24 | 10.15 | 13.98 | 13.92 | 67.1% | 66.7% |
| 2023-12 | 207 | 10.17 | 10.05 | 13.52 | 13.06 | 68.1% | 70.5% |
| 2024-01 | 231 | 11.22 | 11.29 | 16.62 | 16.23 | 68.8% | 68.4% |
| 2024-02 | 174 | 11.78 | 11.03 | 14.49 | 13.81 | 63.8% | 70.7% |
| 2024-03 | 230 | 10.96 | 10.55 | 14.52 | 14.17 | 66.5% | 66.5% |
| 2024-04 | 151 | 10.96 | 10.61 | 15.19 | 14.18 | 68.9% | 68.9% |
| 2024-05 | 41 | 12.92 | 12.91 | 12.63 | 11.87 | 58.5% | 63.4% |
| 2024-06 | 5 | 13.66 | 14.10 | 17.24 | 12.50 | 80.0% | 80.0% |
| 2024-10 | 71 | 9.88 | 10.23 | 15.94 | 15.19 | 62.0% | 62.0% |
| 2024-11 | 220 | 9.67 | 9.85 | 15.27 | 14.79 | 70.9% | 68.6% |
| 2024-12 | 189 | 10.66 | 10.38 | 15.28 | 14.76 | 66.7% | 68.3% |
| 2025-01 | 225 | 11.71 | 11.30 | 13.81 | 13.06 | 67.1% | 69.3% |
| 2025-02 | 176 | 10.40 | 10.43 | 14.64 | 13.84 | 66.5% | 68.8% |
| 2025-03 | 238 | 10.89 | 10.67 | 13.82 | 13.60 | 71.4% | 71.0% |
| 2025-04 | 143 | 11.15 | 10.36 | 15.24 | 14.49 | 72.7% | 74.8% |
| 2025-05 | 39 | 14.35 | 13.68 | 15.30 | 15.53 | 51.3% | 53.8% |
| 2025-06 | 7 | 8.90 | 8.64 | 14.17 | 10.93 | 57.1% | 57.1% |
| All | 2620 | 10.85 | 10.62 | 14.66 | 14.15 | 67.6% | 68.7% |

## Complexity curve (ridge alpha)

Larger alpha shrinks the coefficients toward zero (a simpler model). Training error is in-sample on each fold's training games.

| Alpha | Margin MAE, validation | Margin MAE, training | Total MAE, validation | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 10.85 | 10.43 | 14.67 | 67.6% | +0.08 |
| 0.1 | 10.85 | 10.43 | 14.67 | 67.6% | +0.08 |
| 1 | 10.85 | 10.43 | 14.67 | 67.6% | +0.08 |
| 10 | 10.85 | 10.43 | 14.67 | 67.5% | +0.08 |
| 100 | 10.85 | 10.43 | 14.67 | 67.4% | +0.07 |
| 1000 | 10.89 | 10.46 | 14.67 | 67.7% | +0.05 |
| 10000 | 11.07 | 10.63 | 14.74 | 65.7% | +0.07 |
| 100000 | 11.61 | 11.01 | 15.18 | 65.3% | +0.48 |

Best on validation: alpha 100 (config.RIDGE_ALPHA is 10; changing it is a modeling decision, so it is not changed automatically).

## Learning curve

Each fit uses only the most recent N team-games of history; scored on 2024-25 months that have at least the largest N available.

| Training team-games | Margin MAE, validation | Margin MAE, training | Validation games |
| --- | --- | --- | --- |
| 100 | 11.33 | 11.44 | 1309 |
| 250 | 11.00 | 11.33 | 1309 |
| 500 | 10.93 | 11.02 | 1309 |
| 1000 | 10.84 | 10.89 | 1309 |
| 2000 | 10.80 | 10.91 | 1309 |
| 4000 | 10.81 | 10.76 | 1309 |
