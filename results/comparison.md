# Comparing kinds of model

Decision 11 in docs/decisions.md. Every model predicts each team's points from the same inputs as ridge regression, on the same monthly folds of the validation seasons. In each fold it picks its own setting on the most recent 20% of the training games, then refits on all of them. Changes are in average error against ridge on the same games; noise is 2 standard errors, measured month to month. A model replaces ridge only if it beats the noise (margin error first, then total error) and doesn't move the average total more than 0.5 points further from the actual. If several do, the lowest margin error wins.

## Against ridge regression

| Model | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge regression | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| Elastic net | 2621 | 10.85 | 14.67 | 67.6% | +0.10 | +0.004 | 0.004 | +0.003 | 0.008 | tie |
| k-nearest neighbors | 2621 | 11.21 | 14.99 | 66.5% | +0.19 | +0.364 | 0.118 | +0.323 | 0.150 | worse |
| Support vector regression | 2621 | 11.06 | 14.77 | 66.9% | -0.03 | +0.206 | 0.104 | +0.099 | 0.111 | worse |
| Random forest | 2621 | 10.97 | 14.78 | 66.4% | +0.28 | +0.123 | 0.069 | +0.104 | 0.092 | worse |
| Gradient boosting | 2621 | 10.96 | 14.80 | 67.4% | +0.17 | +0.106 | 0.088 | +0.131 | 0.068 | worse |
| Neural network | 2621 | 10.95 | 14.79 | 67.5% | -0.05 | +0.104 | 0.091 | +0.114 | 0.103 | worse |
| Blend of all seven | 2621 | 10.86 | 14.87 | 67.6% | -0.65 | +0.015 | 0.034 | +0.197 | 0.233 | tie |

**Ridge regression stays.**

## Training and validation error

| Model | Margin MAE, training | Margin MAE, validation | Gap |
| --- | --- | --- | --- |
| Ridge regression | 10.34 | 10.85 | +0.51 |
| Elastic net | 10.34 | 10.85 | +0.51 |
| k-nearest neighbors | 10.46 | 11.21 | +0.75 |
| Support vector regression | 10.20 | 11.06 | +0.85 |
| Random forest | 9.70 | 10.97 | +1.28 |
| Gradient boosting | 10.06 | 10.96 | +0.90 |
| Neural network | 10.24 | 10.95 | +0.72 |

## Settings picked

| Model | What it is | Settings tried | Picked most often | Months |
| --- | --- | --- | --- | --- |
| Ridge regression | the current model: a weighted sum of the inputs | 1 | alpha 10 | 18 of 18 |
| Elastic net | a weighted sum that can also drop inputs | 6 | penalty 0.001, L1 share 0.9 | 8 of 18 |
| k-nearest neighbors | the average of the most similar past team-games | 4 | 100 neighbors | 15 of 18 |
| Support vector regression | a smooth curve that ignores small misses | 3 | C 0.3 | 17 of 18 |
| Random forest | the average of 300 decision trees, each on a resample of the games | 6 | leaves of 30+, 50% of inputs per split | 9 of 18 |
| Gradient boosting | small trees added one at a time, each fixing the last ones' misses | 6 | depth 2, 300 trees | 6 of 18 |
| Neural network | a small multilayer perceptron | 4 | 16 units, penalty 0.1 | 10 of 18 |

## Blend weights

Average weight of each model in the blend, across months.

| Model | Weight |
| --- | --- |
| Ridge regression | 0.39 |
| Elastic net | 0.18 |
| k-nearest neighbors | 0.01 |
| Support vector regression | 0.03 |
| Random forest | 0.19 |
| Gradient boosting | 0.03 |
| Neural network | 0.16 |
