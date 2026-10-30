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
| 10 | [Hand-picked settings](#10-hand-picked-settings) | Rating half-life 24, minutes half-life 12, replacement level -3; five others kept | Beat the noise on margin error; total error rose 0.11 |
| 11 | [Kinds of model](#11-kinds-of-model) | Ridge regression stays | Five flexible models were worse on margin error beyond the noise; elastic net and a blend tied |

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

**Evidence.** On 2,620 validation games the model (as of decision 10) trails Vegas by about a quarter of a point on margin error and half a point on total error:

| | Margin error | Total error | Winners picked |
| --- | --- | --- | --- |
| Model | 10.85 | 14.66 | 67.6% |
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

---

## 10. Hand-picked settings

**Question.** Eight numbers in `config.py` were set by judgment when the model was built, not by evidence. Would other values predict better?

| # | Setting | What it controls | In use | Values tried |
| --- | --- | --- | --- | --- |
| 1 | Rating half-life (`RATING_HALF_LIFE_GAMES`) | How many games until an old game counts half as much in a team's ratings | 12 | 6, 9, 12, 18, 24 |
| 2 | Last season's games (`PRIOR_SEASON_GAME_WEIGHT`) | How much last season's games count once a new season starts | 0.4 | 0.2, 0.4, 0.6, 0.8 |
| 3 | Rating shrinkage (`SHRINK_GAMES`) | Games of league-average play added to every team rating, so a few games can't swing it | 4 | 0, 2, 4, 8 |
| 4 | Minutes half-life (`MINUTES_HALF_LIFE_GAMES`) | How many games until a player's old minutes count half as much | 5 | 2, 3, 5, 8, 12 |
| 5 | Rotation player cutoff (`ROTATION_MINUTES`) | Minutes a player usually plays, when he plays, to be expected at those minutes when back from a rest or injury | 20 | 12, 16, 20, 24 |
| 6 | Replacement level (`REPLACEMENT_OBPM`, `REPLACEMENT_DBPM`) | BPM for players with no record, offense + defense, split 3 to 1 | -2 | -1, -2, -3 |
| 7 | BPM shrinkage (`BPM_PRIOR_MINUTES`) | How hard players with few minutes last season are pulled toward replacement level. At 500, a player with 250 minutes gets half his own BPM and half replacement level | 500 | 250, 500, 1000, 2000 |
| 8 | Rest days cap (`MAX_REST_DAYS`) | Rest beyond this many days counts the same | 3 | 2, 3, 5 |

**What this test can't check.**
- `PLAY_PROBABILITY`, the chance a player plays given his injury report status (for example 50% for Questionable). Past injury reports aren't kept, so validation uses who actually played and never needs these numbers. They stay a judgment call.
- Part of `BPM_PRIOR_MINUTES`: during a season it also sets how fast this season's BPM takes over from last season's. To avoid looking ahead, validation uses only last season's BPM, so only the pull toward replacement level is tested.

**Test fixed in advance,** written down before any results existed (October 26, 2026). `python -m nbamodel tune` runs it and writes [results/tuning.md](../results/tuning.md):

1. Check the settings one at a time, in the order of the table. The team rating settings go first because they shape the biggest input, the pace x efficiency formula.
2. Score every value on the same validation months as the value in use, with the noise test from decisions 8 and 9. Margin error decides first, then total error, and a change must exceed 2 standard errors, measured month to month. The bias guard applies: no value that moves the average total more than 0.5 points further from the actual.
3. A value replaces the one in use only if it is better beyond the noise. If several are, the lowest margin error wins (then total error, then bias). Otherwise the value in use stays. None of these values is simpler than another, so without evidence there's no reason to move.
4. Later settings are checked with the earlier winners in place. There is one pass only, so the number of comparisons is fixed at 24.
5. **Luck.** Each comparison has about a 1 in 20 chance of beating the noise by luck alone, so across 24 of them about one could, even if no setting truly mattered. That's why the locked test season, scored once at the end, has the last word on whether tuning helped.
6. If a winner sits at the end of its range, the write-up says so. Trying values beyond it would be a new test.

*Fix during the first run (October 27, 2026).* The run stopped at a shrinkage of 0. Before a team's first game there was nothing to average, so the rating came out empty and the regression refused it. Now a team with no games is rated at the league average, which is what any shrinkage above 0 already does. Nothing changes for other values, and no results had been seen. The test is otherwise unchanged.

**Result (October 28, 2026).** Full tables are in [results/tuning.md](../results/tuning.md). Three settings beat the noise:

| Setting | Was | Test picked | Margin change | Noise | Total change | Noise |
| --- | --- | --- | --- | --- | --- | --- |
| Rating half-life | 12 | 24 | -0.037 | 0.026 | +0.130 | 0.104 |
| Minutes half-life | 5 | 12 | -0.026 | 0.014 | -0.018 | 0.017 |
| Replacement level | -2 | -3 | -0.009 | 0.009 | +0.003 | 0.003 |

The other five (last season's games, rating shrinkage, rotation cutoff, BPM shrinkage and the rest cap) tied, so they keep their values. Before and after, on the same 2,621 validation games:

| | Margin MAE | Total MAE | Winner % | Total bias |
| --- | --- | --- | --- | --- |
| Before | 10.92 | 14.56 | 67.1% | -0.41 |
| After | 10.85 | 14.67 | 67.5% | +0.08 |

**Choice (Steve, October 28, 2026): apply all three,** as the rule fixed in advance says.
- **Rating half-life 24.** Team ratings now remember twice as far back. Longer memory makes them steadier and better at telling which team is stronger (margin), but probably slower to follow changes in how much teams score, which is where the cost shows: total error rose 0.13, beyond its noise. A half-life of 18 also beat the noise, at a smaller cost to totals (+0.069), but the rule takes the lowest margin error.
- **Minutes half-life 12.** A player's expected minutes now draw on about his last dozen games instead of five, so one short or long night moves them less. This helped both margin and total.
- **Replacement level -3.** A player with no BPM record is assumed to be a little worse. This won by a hair.
- **Margin decides first.** It's the same rule that left the four factors out in decision 9, where totals improved and margin got worse. Twice now, margin and total have pulled in different directions. That suggests a separate model for totals might be worth its own test.
- **Ends of the ranges.** All three winners sit at the end of the values tried, so values further out might do better still. Trying them would be a new test, with its own luck.
- **Luck.** With 24 comparisons, about one could win by luck (point 5). The smallest win, replacement level, is the likeliest. The locked test season has the last word.

![Tuning the hand-picked settings](figures/tuning.png)


---

## 11. Kinds of model

**Question.** Ridge regression adds up the inputs with fixed weights. Would a more flexible kind of model, one that can follow curves and combine inputs, predict better from the same inputs?

**Why ask.** Decision 7 found ridge limited by bias, not by data: its error on the games it learned from (10.43) is not far below its error on new months (10.85), and more history stopped helping after about 2,000 team-games. A model that can bend is the textbook remedy for bias. It can also memorize noise in past games, which is what this test checks.

| Model | How it predicts | Settings tried, simplest first |
| --- | --- | --- |
| Ridge regression (in use) | A weighted sum of the inputs | alpha 10, from decision 7 |
| Elastic net | A weighted sum that can also drop inputs | 6: penalty 0.1, 0.01, 0.001, each with an L1 share of 0.9 and 0.1 |
| k-nearest neighbors | The average of the most similar past team-games | 4: 400, 200, 100, 50 neighbors |
| Support vector regression | A smooth curve (RBF kernel) that ignores small misses | 3: C 0.3, 1, 3 |
| Random forest | The average of 300 decision trees, each grown on a resample of the games | 6: smallest leaf 100, 30, 10 team-games, each with half or all of the inputs per split |
| Gradient boosting | Small trees added one at a time, each correcting the misses of those before | 6: depth 2, 3, 4, each with 100 and 300 trees (learning rate 0.05) |
| Neural network | A small multilayer perceptron | 4: one layer of 16 or two of 32 and 16, each with penalty 0.1 and 0.001 |
| Blend | A weighted average of all seven | weights learned every month |

Elastic net, k-nearest neighbors, support vector regression and the neural network see the inputs and points rescaled to a common scale, which they need; the others don't.

**Test fixed in advance,** written down before any results existed (October 30, 2026). `python -m nbamodel compare` runs it and writes [results/comparison.md](../results/comparison.md):

1. **Same everything but the model.** The 8 inputs in use, the same target (each team's points; margin and total come from the pair), and the same 2,620 validation games and monthly folds as every earlier decision.
2. **Settings chosen without peeking.** In each month, every model tries its settings on that month's training games: it learns from the older 80% and is scored on the most recent 20%, in the usual order (margin error, then total error, then bias; within 0.01 points, the simpler setting). Its pick is refit on all the training games and predicts the month. The month being predicted never helps choose a setting.
3. **The blend** learns its weights on the same recent 20%, from each model's predictions with its picked setting. Weights can't go below zero, so they read as shares, such as 60% ridge and 40% boosting.
4. **Each challenger against ridge** with the noise test from decisions 8 to 10. Margin error decides first, then total error, and a change must exceed 2 standard errors, measured month to month. The bias guard applies: no model that moves the average total more than 0.5 points further from the actual.
5. **A challenger replaces ridge if it is better beyond the noise.** If several are, the lowest margin error wins (then total error, then bias). Otherwise ridge stays, since it is the simplest and the easiest to explain.
6. **A winner goes live** (Steve, October 30, 2026). Daily predictions switch to it, and charts are added that explain it: which inputs matter most, and how each one moves the score. The per-game breakdown (`explain`) adds up exactly only for a weighted sum, so for any other winner it becomes an approximation.
7. **Luck.** Each comparison has about a 1 in 20 chance of passing by luck alone. Across seven, there's up to about a 1 in 4 chance that one passes even if none is truly better. The locked test season, scored once at the end, has the last word.
8. **Also reported, not part of the rule:** each model's error on its own training games next to its error on the months it predicted. A wide gap means a model memorizes past games (overfits); two high numbers mean it misses the pattern (underfits).

**What this test can't check.**
- Other inputs. Flexible models might do better with the raw pieces of the formula (pace, offensive and defensive ratings) than with the formula's result. That would be a new question with its own test.
- Settings outside the ranges above.

**Result (October 30, 2026): ridge regression stays.** No other kind of model beat it beyond the noise. Full tables are in [results/comparison.md](../results/comparison.md).

![Kinds of model against ridge regression](figures/model_comparison.png)

- **The flexible models were worse, not better.** The neural network, gradient boosting, random forest, support vector regression and k-nearest neighbors all raised margin error, by 0.10 to 0.36 points, beyond the noise. Their total error rose too.
- **Elastic net is ridge again.** Its most common pick was the smallest penalty tried, so it ended up nearly identical to ridge and tied (+0.004 on margin).
- **The blend tied.** It leaned on ridge most (39%), then the random forest (19%), elastic net (18%) and the neural network (16%). Its margin error matched ridge within the noise, but its totals ran 0.65 points low on average. A tie keeps the simpler model.
- **Why: the flexible models memorized past games.** Each fit its own training games better than ridge (the random forest's error there was 9.70 against ridge's 10.34) and new months worse (10.97 against 10.85). Ridge's own gap of 0.51 is the baseline any model pays because each new month differs a little from the past; a weighted sum of 8 inputs has almost nothing to memorize with. The wider gaps of the flexible models, 0.72 to 1.28, are noise they learned as if it were pattern.
- **What it says about the model.** The same as decision 7's learning curve: the limit is the inputs, not the shape of the model. The pace x efficiency formula already holds the main interaction in these inputs, so trees and networks find little left to bend. Gains have to come from better inputs.
- **Luck.** Nothing passed, so there is no lucky winner to worry about (point 7). The locked test season stays untouched.

![Training and validation error by kind of model](figures/model_fit.png)
