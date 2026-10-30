# Better team ratings

Decision 12 in docs/decisions.md. Each change to how team ratings are built is checked in this order against the ratings in use, on the same validation folds. Changes are in average error on the same games; noise is 2 standard errors, measured month to month. A value replaces the one in use only if it beats the noise (margin error first, then total error) and doesn't move the average total more than 0.5 points further from the actual. Otherwise the value in use stays. The settings after it are checked with the winner in place.

Each step also shows the margin change on two slices, which are reported but don't decide: early season (neither team has played 15 games yet) and after trades (a team's first 15 games after gaining or losing, midseason, a player who averaged 20+ minutes).

## Season-start prior (share of last season's edge kept)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| 0.8 | 2621 | 10.85 | 14.67 | 67.4% | +0.15 | +0.001 | 0.015 | -0.001 | 0.022 | tie |
| 0.6 | 2621 | 10.86 | 14.68 | 67.4% | +0.24 | +0.013 | 0.029 | +0.011 | 0.040 | tie |
| 0.4 | 2621 | 10.88 | 14.71 | 67.3% | +0.34 | +0.035 | 0.049 | +0.035 | 0.059 | tie |

**Kept 1.**

### Season-start prior: by slice

| Setting | Early season games | Early season: margin change | Early season: noise | After trades games | After trades: margin change | After trades: noise |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 432 | - | - | 703 | - | - |
| 0.8 | 432 | +0.035 | 0.053 | 703 | -0.007 | 0.022 |
| 0.6 | 432 | +0.120 | 0.092 | 703 | -0.009 | 0.044 |
| 0.4 | 432 | +0.247 | 0.159 | 703 | -0.009 | 0.062 |

## Trend (share of the recent trend followed)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| 0.25 | 2621 | 10.86 | 14.61 | 67.1% | -0.44 | +0.011 | 0.017 | -0.063 | 0.071 | tie |
| 0.5 | 2621 | 10.89 | 14.60 | 67.0% | -0.85 | +0.043 | 0.027 | -0.073 | 0.122 | worse |
| 1 | 2621 | 10.99 | 14.66 | 66.5% | -1.39 | +0.145 | 0.046 | -0.009 | 0.190 | worse |

**Kept 0.**

### Trend: by slice

| Setting | Early season games | Early season: margin change | Early season: noise | After trades games | After trades: margin change | After trades: noise |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 432 | - | - | 703 | - | - |
| 0.25 | 432 | +0.014 | 0.012 | 703 | -0.009 | 0.035 |
| 0.5 | 432 | +0.047 | 0.024 | 703 | +0.012 | 0.057 |
| 1 | 432 | +0.138 | 0.058 | 703 | +0.107 | 0.081 |

## Change detection (CUSUM threshold, net rating points)

| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise | Total change | Total noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 2621 | 10.85 | 14.67 | 67.5% | +0.08 | - | - | - | - | current |
| 50 | 2621 | 10.90 | 14.60 | 67.1% | -0.07 | +0.054 | 0.044 | -0.074 | 0.071 | worse |
| 80 | 2621 | 10.86 | 14.64 | 67.1% | +0.06 | +0.009 | 0.023 | -0.036 | 0.033 | better |

**Changed to 80.** The winner is at the end of the values tried.

### Change detection: by slice

| Setting | Early season games | Early season: margin change | Early season: noise | After trades games | After trades: margin change | After trades: noise |
| --- | --- | --- | --- | --- | --- | --- |
| off | 432 | - | - | 703 | - | - |
| 50 | 432 | +0.020 | 0.063 | 703 | +0.034 | 0.067 |
| 80 | 432 | +0.014 | 0.019 | 703 | -0.016 | 0.065 |

## Summary

| Setting | In use before | Test picked |
| --- | --- | --- |
| Season-start prior | 1 | 1 |
| Trend | 0 | 0 |
| Change detection | off | 80 |
