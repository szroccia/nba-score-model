# Tuning the hand-picked settings

Decision 10 in docs/decisions.md. Each setting is checked in this order against the value in use, on the same validation folds. Changes are in average error on the same games; noise is 2 standard errors, measured month to month. A value replaces the one in use only if it beats the noise (margin error first, then total error) and doesn't move the average total more than 0.5 points further from the actual. Otherwise the value in use stays. The settings after it are checked with the winner in place.

## Rating half-life (games)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 2621 | 11.05 | 14.56 | 66.0% | -0.24 | +0.133 | 0.036 | +0.004 | 0.077 | worse |
| 9 | 2621 | 10.97 | 14.54 | 66.7% | -0.40 | +0.047 | 0.012 | -0.019 | 0.037 | worse |
| 12 | 2621 | 10.92 | 14.56 | 67.1% | -0.41 | - | - | - | - | current |
| 18 | 2621 | 10.89 | 14.63 | 67.1% | -0.22 | -0.033 | 0.013 | +0.069 | 0.055 | better |
| 24 | 2621 | 10.88 | 14.69 | 67.0% | +0.07 | -0.037 | 0.026 | +0.130 | 0.104 | better |

**Changed to 24.** The winner is at the end of the values tried.

## Last season's games (weight)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 2621 | 10.89 | 14.66 | 67.1% | +0.42 | +0.009 | 0.024 | -0.030 | 0.042 | tie |
| 0.4 | 2621 | 10.88 | 14.69 | 67.0% | +0.07 | - | - | - | - | current |
| 0.6 | 2621 | 10.90 | 14.72 | 67.1% | -0.14 | +0.011 | 0.023 | +0.036 | 0.034 | worse |
| 0.8 | 2621 | 10.91 | 14.76 | 66.8% | -0.26 | +0.029 | 0.041 | +0.072 | 0.061 | worse |

**Kept 0.4.**

## Rating shrinkage (games of league average)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2621 | 10.89 | 14.67 | 66.9% | -0.21 | +0.006 | 0.011 | -0.019 | 0.036 | tie |
| 2 | 2621 | 10.89 | 14.68 | 67.0% | -0.07 | +0.002 | 0.005 | -0.011 | 0.018 | tie |
| 4 | 2621 | 10.88 | 14.69 | 67.0% | +0.07 | - | - | - | - | current |
| 8 | 2621 | 10.88 | 14.71 | 67.2% | +0.35 | -0.001 | 0.007 | +0.025 | 0.034 | tie |

**Kept 4.**

## Minutes half-life (games)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2621 | 10.94 | 14.73 | 66.5% | +0.06 | +0.051 | 0.025 | +0.044 | 0.029 | worse |
| 3 | 2621 | 10.91 | 14.71 | 66.7% | +0.07 | +0.028 | 0.013 | +0.024 | 0.015 | worse |
| 5 | 2621 | 10.88 | 14.69 | 67.0% | +0.07 | - | - | - | - | current |
| 8 | 2621 | 10.87 | 14.67 | 67.3% | +0.07 | -0.017 | 0.009 | -0.013 | 0.011 | better |
| 12 | 2621 | 10.86 | 14.67 | 67.5% | +0.07 | -0.026 | 0.014 | -0.018 | 0.017 | better |

**Changed to 12.** The winner is at the end of the values tried.

## Rotation player cutoff (minutes)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 2621 | 10.85 | 14.67 | 67.1% | +0.02 | -0.007 | 0.015 | -0.000 | 0.018 | tie |
| 16 | 2621 | 10.85 | 14.67 | 67.5% | +0.05 | -0.011 | 0.012 | +0.001 | 0.012 | tie |
| 20 | 2621 | 10.86 | 14.67 | 67.5% | +0.07 | - | - | - | - | current |
| 24 | 2621 | 10.86 | 14.67 | 67.1% | +0.08 | +0.000 | 0.011 | -0.002 | 0.009 | tie |

**Kept 20.**

## Replacement level (BPM, offense + defense)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1 | 2621 | 10.87 | 14.67 | 67.3% | +0.07 | +0.014 | 0.010 | -0.003 | 0.002 | worse |
| -2 | 2621 | 10.86 | 14.67 | 67.5% | +0.07 | - | - | - | - | current |
| -3 | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | -0.009 | 0.009 | +0.003 | 0.003 | better |

**Changed to -3.** The winner is at the end of the values tried.

## BPM shrinkage (minutes)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 2621 | 10.85 | 14.67 | 67.5% | +0.09 | +0.003 | 0.005 | +0.000 | 0.003 | tie |
| 500 | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| 1000 | 2621 | 10.85 | 14.67 | 67.6% | +0.07 | -0.003 | 0.005 | +0.001 | 0.004 | tie |
| 2000 | 2621 | 10.84 | 14.67 | 67.6% | +0.06 | -0.005 | 0.010 | +0.002 | 0.008 | tie |

**Kept 500.**

## Rest days cap (days)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2621 | 10.85 | 14.67 | 67.5% | +0.07 | +0.002 | 0.004 | -0.002 | 0.006 | tie |
| 3 | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| 5 | 2621 | 10.85 | 14.66 | 67.6% | +0.08 | +0.001 | 0.004 | -0.009 | 0.011 | tie |

**Kept 3.**

## Summary

| Setting | In use before | Test picked |
| --- | --- | --- |
| Rating half-life | 12 | 24 |
| Last season's games | 0.4 | 0.4 |
| Rating shrinkage | 4 | 4 |
| Minutes half-life | 5 | 12 |
| Rotation player cutoff | 20 | 20 |
| Replacement level | -2 | -3 |
| BPM shrinkage | 500 | 500 |
| Rest days cap | 3 | 3 |
