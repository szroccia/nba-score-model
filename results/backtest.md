# Backtest

Each season is predicted by a model trained only on earlier seasons.
Lineups use who actually played; ratings use only games before each date.
The locked test season (2025-26) is left out; only `evaluate --final` scores it.
Playoff games in ratings: include.

| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | Ridge (B) | 1312 | 9.24 | 10.92 | 14.67 | 66.6% | +1.65 |
| 2023-24 | Formula (A) | 1312 | 9.32 | 11.11 | 14.66 | 65.0% | +0.69 |
| 2023-24 | Ridge (B), first 15 games | 218 | 8.65 | 10.13 | 13.53 | 66.5% | +0.32 |
| 2024-25 | Ridge (B) | 1309 | 9.2 | 10.8 | 14.72 | 68.0% | -1.82 |
| 2024-25 | Formula (A) | 1309 | 9.3 | 11.07 | 14.81 | 65.7% | -2.1 |
| 2024-25 | Ridge (B), first 15 games | 214 | 9.21 | 9.48 | 15.61 | 67.8% | -2.26 |
