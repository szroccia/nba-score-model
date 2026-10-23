# Backtest

Each season is predicted by a model trained only on earlier seasons.
Lineups use who actually played; ratings use only games before each date.

| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | Ridge (B) | 1312 | 9.23 | 11.03 | 14.51 | 65.7% | +0.34 |
| 2023-24 | Formula (A) | 1312 | 9.29 | 11.19 | 14.53 | 63.8% | -0.06 |
| 2024-25 | Ridge (B) | 1309 | 9.21 | 10.88 | 14.69 | 67.5% | -1.54 |
| 2024-25 | Formula (A) | 1309 | 9.3 | 11.12 | 14.83 | 66.5% | -1.87 |
| 2025-26 | Ridge (B) | 1310 | 9.46 | 11.33 | 14.96 | 68.2% | -0.15 |
| 2025-26 | Formula (A) | 1310 | 9.55 | 11.46 | 15.07 | 69.3% | -0.67 |
