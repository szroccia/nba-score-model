# Season and playoff odds

Decision 14 in docs/decisions.md. Each validation season (2023-24 and 2024-25) is played out 10,000 times from the first of November, December, January, February, March and April, with only what was known that day: the games played so far, team ratings as of the day, each team's roster as the players who had played for it that season (all assumed healthy), and a regression fit on earlier games. Every remaining game's prediction stays frozen for the rest of the simulated season. How widely simulated games land around their predictions comes from how far frozen predictions missed in earlier seasons only, so no season helps score itself.

Log loss scores the odds of making the playoffs (directly or through the play-in): minus the log of the chance given to what happened, averaged over teams and snapshot days, with every chance kept between 1% and 99%. Lower is better. Changes are against the method in use on the same forecasts; noise is 2 standard errors, with each team-season (its 6 snapshot days) as one block. A method replaces the one in use only if it beats the noise. Brier score, win totals, calibration and the champion's odds are reported but don't decide.

## Playoff odds

| Method | Forecasts | Log loss | Brier score | Log loss change | Noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Independent games | 360 | 0.3134 | 0.1003 | - | - | current |
| Lasting team errors | 360 | 0.3006 | 0.0972 | -0.0127 | 0.0281 | tie |

For scale: giving every team 53% (16 of 30 teams make the playoffs) scores a log loss of 0.6909.

**Independent games stay.**

### Simulation methods

| Method | What it is |
| --- | --- |
| Independent games | every simulated game misses by fresh luck, as wide as frozen predictions have missed |
| Lasting team errors | the same total miss, split into a part that stays with each team for the whole simulated season and a part that is fresh every game |

## How far frozen predictions miss

Each validation season's spread is fit on the frozen predictions of the seasons before it, from the same snapshot days. A game's miss is the home team's lasting part minus the away team's, plus that night's luck, so the typical miss is the square root of twice the lasting part squared plus the nightly part squared. Independent games simulate every game with the typical miss; lasting team errors draw each team's lasting part once per simulated season, plus nightly luck.

| Validation season | Fit on | Frozen predictions | Typical miss, points | Lasting team part | Nightly part |
| --- | --- | --- | --- | --- | --- |
| 2023-24 | 2022-23 | 3,549 | 13.41 | 2.46 | 12.95 |
| 2024-25 | 2022-23, 2023-24 | 7,411 | 14.12 | 2.82 | 13.54 |

## Win totals

Reported only. The average simulated win total against actual wins, and how often actual wins landed inside the middle 80% of simulated seasons (rounded outward to whole wins). Current pace scales each team's record so far to its full schedule.

| Forecast | Forecasts | Average miss, wins | Inside the range |
| --- | --- | --- | --- |
| Independent games | 360 | 4.10 | 61.9% |
| Lasting team errors | 360 | 4.10 | 78.9% |
| Current pace | 360 | 5.12 | - |

## Calibration

Reported only. The chance of making the playoffs, grouped in tenths, against how often those teams made it.

| Chance of playoffs | Independent forecasts | Independent said | Made it, independent | Lasting forecasts | Lasting said | Made it, lasting |
| --- | --- | --- | --- | --- | --- | --- |
| 0-10% | 103 | 2.0% | 2.9% | 84 | 1.9% | 0.0% |
| 10-20% | 19 | 14.7% | 21.1% | 21 | 14.6% | 9.5% |
| 20-30% | 12 | 23.9% | 33.3% | 20 | 24.1% | 25.0% |
| 30-40% | 18 | 35.9% | 50.0% | 15 | 34.7% | 46.7% |
| 40-50% | 16 | 45.5% | 43.8% | 25 | 44.3% | 44.0% |
| 50-60% | 12 | 55.7% | 58.3% | 18 | 55.6% | 61.1% |
| 60-70% | 17 | 65.1% | 58.8% | 23 | 65.1% | 60.9% |
| 70-80% | 15 | 74.7% | 66.7% | 22 | 75.8% | 68.2% |
| 80-90% | 24 | 85.7% | 75.0% | 31 | 84.6% | 90.3% |
| 90-100% | 124 | 97.6% | 96.8% | 101 | 97.4% | 98.0% |

## By snapshot day

Reported only. Both validation seasons together, by the day the simulation started.

| Simulated from | Forecasts | Log loss, independent | Log loss, lasting | Wins miss, independent | Wins miss, lasting |
| --- | --- | --- | --- | --- | --- |
| November 1 | 60 | 0.5201 | 0.4739 | 7.14 | 7.13 |
| December 1 | 60 | 0.4281 | 0.4005 | 5.83 | 5.83 |
| January 1 | 60 | 0.3692 | 0.3538 | 4.50 | 4.50 |
| February 1 | 60 | 0.2668 | 0.2679 | 3.67 | 3.67 |
| March 1 | 60 | 0.1598 | 0.1683 | 2.43 | 2.43 |
| April 1 | 60 | 0.1363 | 0.1395 | 1.02 | 1.02 |

## The champion

Reported only. The eventual champion's title odds under independent games (the method in use) from each snapshot day, with its rank among the 30 teams.

| Season | Champion | November 1 | December 1 | January 1 | February 1 | March 1 | April 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-24 | BOS | 30.5% (1st) | 32.7% (1st) | 41.5% (1st) | 30.7% (1st) | 53.7% (1st) | 71.0% (1st) |
| 2024-25 | OKC | 31.7% (2nd) | 32.1% (2nd) | 39.5% (1st) | 37.5% (1st) | 22.5% (2nd) | 45.7% (1st) |
