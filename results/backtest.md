# Backtest

Each season is predicted by a model trained only on earlier seasons.
Lineups use who actually played; ratings use only games before each date.
The locked test season (2025-26) is left out; only `evaluate --final` scores it.

| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | Ridge (B) | 1312 | 9.23 | 11.03 | 14.51 | 65.7% | +0.34 |
| 2023-24 | Formula (A) | 1312 | 9.29 | 11.19 | 14.53 | 63.8% | -0.06 |
| 2024-25 | Ridge (B) | 1309 | 9.21 | 10.88 | 14.69 | 67.5% | -1.54 |
| 2024-25 | Formula (A) | 1309 | 9.3 | 11.12 | 14.83 | 66.5% | -1.87 |
