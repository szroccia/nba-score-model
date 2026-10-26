# Decision log

The major modeling choices, what was considered, the evidence and the reason for each choice. Every chart here is drawn by `python -m nbamodel report` from results files in this repo, so all of them can be reproduced.

**How decisions are made.** Each choice is a setting in `src/nbamodel/config.py`. The test that settles it is fixed before the results are seen. Choices are judged only on the validation seasons; the test season is kept for a final, logged check (see [Evaluation design](#5-evaluation-design)).

| # | Decision | Choice | Evidence |
| --- | --- | --- | --- |
| 1 | [Model approach](#1-model-approach) | Team ratings + ridge regression | Beats the plain formula in every validation season |
| 2 | [Data sources](#2-data-sources) | Official NBA stats and injury report, run from a home connection | Most reliable sources; cloud servers are blocked |
| 3 | [Player value](#3-player-value) | Box Plus/Minus, weighted by expected minutes | Lineup inputs earn real weight in the regression |
| 4 | [Offseason roster moves](#4-offseason-roster-moves) | Current rosters shift last season's ratings | Needed for October, when ratings are mostly last season |
| 5 | [Evaluation design](#5-evaluation-design) | Rolling monthly folds, locked test season, Vegas benchmark | Avoids look-ahead and gives an honest bar to beat |
| 6 | [Playoff games in ratings](#6-playoff-games-in-ratings) | Include | Best total error and early-season bias |
| 7 | [Ridge strength](#7-ridge-strength) | alpha = 10 | All reasonable values tie; ties go to the simpler model |
| 8 | [Strength of schedule](#8-strength-of-schedule) | Adjust for opponents | Beat the noise on margin error, after a fix for a flaw that biased totals |
| 9 | [Four factors](#9-four-factors) | Left out | Worse on margin error beyond the noise, though better on totals |

---

## 1. Model approach

**Options.**
- **(A) Possession formula only:** pace × (offense + opponent defense − league average).
- **(B) The same ratings fed into a ridge regression** with lineup, home and rest inputs.
- **(C) Gradient boosting** on a wide feature set.

**Choice: B.** The regression learns how much to trust each input instead of fixing the weights by hand, and its coefficients can be read directly ([What the model learned](../README.md#what-the-model-learned)). C can find interactions but is harder to explain and needs more features. It is planned as a later comparison, judged by the same evaluation.

**Evidence.** The regression's average margin error is 0.16 and 0.24 points lower than the formula's in the two validation seasons:

![Regression vs formula](figures/model_vs_formula.png)

## 2. Data sources

**Options:** official stats.nba.com data (through `nba_api`) run from a home connection; ESPN and Basketball-Reference run in the cloud; or a mix.

**Choice: official sources.**

| Data | Source |
| --- | --- |
| Box scores and schedule | stats.nba.com through `nba_api` |
| Injury and rest news | Official NBA injury report PDFs |
| Player value | Basketball-Reference |
| Benchmark | ESPN's archive of Vegas closing lines |

stats.nba.com blocks cloud servers, so the data steps run on a home computer. `python -m nbamodel validate` checks every download: game counts, box scores that add up, team totals matching player totals, and games still in progress, which are left out.

## 3. Player value

**Choice: Box Plus/Minus (BPM),** split into offense and defense. Each player's value is weighted by the minutes he is expected to play tonight compared with recent games. Last season's BPM is blended with this season's: they count equally once a player has 500 minutes this season, and this season's counts more after that.

**Why.** BPM exists for every player in every season, separates offense from defense, and is public, so anyone can check it.

**Evidence and limits.** The regression gives the lineup inputs real weight: about 1.25 for the opponent's lineup and 0.6 for the team's own ([coefficients](figures/coefficients.png)). So who plays carries signal, and the model trusts BPM-based shifts to its own offense at a bit over half their face value. BPM misses defense that doesn't show up in the box score.

## 4. Offseason roster moves

**Options:** only use who played recently, or pull current rosters and let them shift last season's ratings.

**Choice: current rosters.**
- Players who left a team are removed from its lineup.
- Newcomers are added with their minutes and BPM from their old team.

In October, team ratings are mostly last season's games, so this is what accounts for trades and signings. The shift fades on its own as the new roster plays games together.

## 5. Evaluation design

**Choice:** rolling-origin cross-validation by month, a locked test season, a selection rule fixed in advance, and Vegas closing lines as the benchmark.

![Evaluation scheme](figures/evaluation_scheme.png)

**Why.**
- **No look-ahead:** basketball data is a time series, so random k-fold splits would train on games played after the ones being predicted. Refitting before each month and predicting that month is also exactly how the live model runs.
- **Untouched test season:** 2025-26 is never used to choose anything. Every look at it is appended to `results/test_log.md`. One earlier look is on record here instead: the first backtest, run in October 2026 before this lock existed, also scored 2025-26 (margin error 11.33 for the regression, 11.46 for the formula). It showed the same thing as the other seasons and settled nothing they didn't. Since then `backtest` leaves the test season out.
- **A rule fixed in advance:** settings are ranked by margin error, then total error, then total bias nearest zero, with near-ties going to the simpler model. From decision 8 on, "near tie" has a measured meaning: a change within the noise (see [decision 8](#8-strength-of-schedule)).
- **A real bar to beat:** the closing line is the best public forecast.

**Evidence.** On 2,620 validation games the model (as of decision 8) is within a third of a point of Vegas:

| | Margin error | Total error | Winners picked |
| --- | --- | --- | --- |
| Model | 10.92 | 14.55 | 67.1% |
| Vegas closing line | 10.62 | 14.15 | 68.7% |

![Model vs Vegas by month](figures/vegas_by_month.png)

## 6. Playoff games in ratings

**Question.** Playoff games are slower and lower scoring, and early in a season they make up a big share of each team's recent games. Should ratings use them as they are, leave them out, or rescale them to the playoff league average?

**Test fixed in advance:** margin error, total error, and total bias in October (the month most affected), on the same validation folds.

**Choice: include.**
- Margin error is a near tie: exclude is 0.02 points better, well within game-to-game noise.
- Including playoffs gives the lowest total error (by 0.13).
- It is also the only setting that doesn't push October totals 1.6 to 2.2 points too high.

![Playoff settings](figures/playoff_settings.png)

## 7. Ridge strength

**Question.** How hard should ridge regression pull the coefficients toward zero (`RIDGE_ALPHA`)?

**Choice: alpha = 10.**
- Every alpha from 0.01 to 100 gives the same validation error (10.95).
- With about 10,000 team-games (5,000 games) and only 8 inputs, a small penalty barely moves the coefficients.
- Under the selection rule, ties go to the simpler (more regularized) model, which is 10.
- Error only rises once alpha reaches 1,000 and starts shrinking real effects.

![Ridge alpha](figures/ridge_alpha.png)

**What this says about the model.** Training and validation error stay close at every alpha, so the model is not overfitting. The learning curve agrees: accuracy stops improving after about half a season of history. The model is limited by its inputs (bias), not by data or variance, so better features are the next lever.

![Learning curve](figures/learning_curve.png)

---

## 8. Strength of schedule

**Question.** Should each game be corrected for how good the opponent was? Scoring 120 against a defense that usually allows 110 (an average defense allows 115) would count as 125, and 120 against a weak defense would count for less. Opponent strength comes only from games before each date, so nothing looks ahead.

**Test fixed in advance,** written down before any results existed (October 24, 2026):

1. Score the model with and without the adjustment on the same validation months.
2. For each game, take the change in the absolute margin error from turning it on, then average the changes.
3. The adjustment stays only if that average is a drop of more than 2 standard errors.
4. If the margin change is within that noise, apply the same test to the total error.
5. If both are within the noise, the simpler model (no adjustment) stays.

**Why pair the games, and why measure noise by month.** Pairing compares the two versions on the very same games, so the noise they share (a 30-point blowout misleads both) cancels out. The standard error is measured month to month: each month is one block, and the noise is how much the change varies between months. Games in the same month share conditions, such as injuries, schedule quirks and the point in the season. Counting every game as independent would make small differences look more certain than they are. Under the old rounding rule, the playoff question would have gone to "exclude" for being 0.02 points better, a gap that was judged to be noise; this test makes that judgment measurable.

**First run (October 24, 2026).** Strength of schedule passed, narrowly.

| Setting | Margin MAE | Total MAE | Winner % | Total bias | October total bias |
| --- | --- | --- | --- | --- | --- |
| Off | 10.95 | 14.59 | 66.7% | -0.34 | -0.33 |
| On | 10.92 | 14.55 | 67.0% | +1.22 | +2.19 |

The margin change was -0.03 points against noise of 0.03. The unrounded change was the larger of the two, so the verdict was "better". Rounded to two decimals, the table and chart looked like a tie, so they now show three. The total change, -0.04 against noise of 0.11, was within the noise.

**A flaw found before switching it on.** Totals went from 0.34 points too low to 1.22 too high, and the test did not look at bias. The first version measured each opponent against the league average, which is built from this season and, more lightly, last season. That average lags when league-wide scoring changes, so the change is mistaken for a tough or easy schedule and shifts every team's ratings. This is the most likely cause of the bias. A test with equal teams and a six-point jump in league scoring shows the first version moving the scoring level by 2 points (`test_opponent_adjustment_ignores_league_wide_scoring_changes`). The fix compares each opponent with the average team's rating on the same day, built the same way, so only differences between teams count.

**Rerun rule, fixed before the rerun (October 24, 2026).** Steve chose to fix the flaw and run the same test once more. The test above is unchanged, and the rerun settles this decision, pass or fail. There is no third version. Both runs stay in this log.

**Rerun (October 24, 2026).**

| Setting | Margin MAE | Total MAE | Winner % | Total bias | October total bias |
| --- | --- | --- | --- | --- | --- |
| Off | 10.95 | 14.59 | 66.7% | -0.34 | -0.33 |
| On | 10.92 | 14.56 | 67.1% | -0.41 | -0.16 |

**Choice: adjust for opponents.**
- Margin error dropped by 0.027 points against noise of 0.026, so it beats the noise, by a hair. The gain is real but small: about 0.25% of the average miss.
- Total error dropped by 0.037, within its noise of 0.050.
- The fix worked. Total bias is back to -0.41, close to the -0.34 without the adjustment, and October bias is the smallest of any setting tried (-0.16).
- The total noise band halved from the first run (0.11 to 0.05). That is most likely because the first version's bias swung from month to month.

![Strength of schedule](figures/strength_of_schedule.png)

## 9. Four factors

**Question.** Two teams with the same efficiency can get there in different ways. Does knowing how a team scores, and not just how much, help predict its next game? Dean Oliver's four factors split efficiency into:

| Factor | Measure |
| --- | --- |
| Shooting | Effective field goal %: (FGM + 0.5 x 3PM) / FGA |
| Turnovers | Turnovers per possession: TOV / (FGA + 0.44 x FTA + TOV) |
| Offensive rebounding | Share of its own misses the offense rebounds: OREB / (OREB + opponent DREB) |
| Free throws | Free throws made per field goal attempt: FTM / FGA |

Each is tracked for a team's offense and for its defense (what opponents did against it), built like the ratings: recent games weighted more, last season counted less, and shrunk toward the league average. They are not adjusted for opponents. For each game, four matchup edges go into the regression, in percentage points: this team's offense plus the opponent's defense, minus twice the league average.

**Choice of design (Steve, October 25, 2026): next to the formula.** The pace x efficiency formula stays, and the four edges are added as inputs (12 instead of 8). Their weights say how much of each kind of edge carries into the next game beyond what the formula already counts. The other option was to rebuild efficiency from the factors alone.

**Test fixed in advance,** written down before any results existed (October 25, 2026). It is decision 8's test with one guard added:

1. Score the model with and without the four edges on the same validation months. Everything else, the features included, is identical.
2. For each game, take the change in the absolute margin error, then average the changes.
3. The factors stay only if that average is a drop of more than 2 standard errors, measured month to month.
4. If the margin change is within that noise, apply the same test to the total error.
5. If both are within the noise, the simpler model (without the factors) stays.
6. **Bias guard.** Even if they pass, the factors stay off if the average total moves more than 0.5 points further from the actual. Decision 8's first run passed the noise test while pushing totals 1.2 points too high; this rule catches that case (`config.BIAS_GUARD_POINTS`, applied to every comparison from here on).

**Result (October 25, 2026).**

| Setting | Margin MAE | Total MAE | Winner % | Total bias | October total bias |
| --- | --- | --- | --- | --- | --- |
| Off | 10.92 | 14.56 | 67.1% | -0.41 | -0.16 |
| On | 10.95 | 14.51 | 67.0% | -0.15 | +0.37 |

**Choice: leave them out.**
- Margin error rose by 0.029 points against noise of 0.028, so it is worse beyond the noise, by a hair. Margin decides first, so the factors stay off.
- Total error fell by 0.049 against noise of 0.036, a real gain on totals, and the average total moved closer to the actual (-0.15). Because margin had already decided, the test never reaches total.
- Read together, the two results suggest the factors say something about how many points a game will have, but blur which team is better. Using them for the total only would be a different design, so it would need its own decision and its own test written in advance.
- The code stays, switched off (`FOUR_FACTORS = False`), so the factors can be tried again later without rebuilding anything.

![Four factors](figures/four_factors.png)

