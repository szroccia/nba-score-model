# Backtest

Each season is predicted by a model trained only on earlier seasons.
Lineups use who actually played; ratings use only games before each date.
The locked test season (2025-26) is left out; only `evaluate --final` scores it.
Playoff games in ratings: include.

| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | Ridge (B) | 1312 | 9.19 | 11.0 | 14.45 | 66.1% | +0.3 |
| 2023-24 | Formula (A) | 1312 | 9.26 | 11.17 | 14.49 | 64.5% | +0.05 |
| 2023-24 | Ridge (B), first 15 games | 218 | 8.62 | 10.22 | 13.45 | 66.1% | -1.28 |
| 2024-25 | Ridge (B) | 1309 | 9.21 | 10.86 | 14.69 | 67.7% | -1.63 |
| 2024-25 | Formula (A) | 1309 | 9.3 | 11.08 | 14.8 | 65.7% | -1.81 |
| 2024-25 | Ridge (B), first 15 games | 214 | 9.28 | 9.56 | 15.7 | 68.7% | -3.33 |
