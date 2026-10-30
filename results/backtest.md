# Backtest

Each season is predicted by a model trained only on earlier seasons.
Lineups use who actually played; ratings use only games before each date.
The locked test season (2025-26) is left out; only `evaluate --final` scores it.
Playoff games in ratings: include.

| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | Ridge (B) | 1312 | 9.23 | 10.93 | 14.6 | 66.1% | +1.37 |
| 2023-24 | Formula (A) | 1312 | 9.3 | 11.13 | 14.6 | 64.6% | +0.41 |
| 2023-24 | Ridge (B), first 15 games | 218 | 8.66 | 10.13 | 13.55 | 66.1% | +0.28 |
| 2024-25 | Ridge (B) | 1309 | 9.19 | 10.8 | 14.7 | 67.8% | -1.53 |
| 2024-25 | Formula (A) | 1309 | 9.3 | 11.09 | 14.81 | 65.1% | -1.99 |
| 2024-25 | Ridge (B), first 15 games | 214 | 9.2 | 9.5 | 15.58 | 68.2% | -1.99 |
