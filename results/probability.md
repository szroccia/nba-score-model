# How sure is each prediction

Decision 13 in docs/decisions.md. Every month from November 2022 on is predicted by a model fit on the games before it, the same way as the evaluation; how far those predictions missed is the record. Each validation month's probabilities are fit only on the record before that month, so no month helps score itself.

Log loss scores a probability: minus the log of the chance given to what happened, averaged over games. Lower is better, and a coin flip scores 0.693. Changes are in average log loss against the method in use on the same games; noise is 2 standard errors, measured month to month. A method replaces the one in use only if it beats the noise; if several do, the lowest log loss wins. Brier score, the average squared miss of the probability, is reported but doesn't decide.

## Win probability

| Method | Games | Log loss | Brier score | Log loss change | Noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Normal spread | 2621 | 0.5988 | 0.2064 | - | - | current |
| Early-season spread | 2621 | 0.5990 | 0.2065 | +0.0001 | 0.0002 | tie |
| Logistic calibration | 2621 | 0.5981 | 0.2062 | -0.0008 | 0.0027 | tie |
| Isotonic calibration | 2621 | 0.6041 | 0.2071 | +0.0052 | 0.0107 | tie |

**The normal spread stays.**

### Win probability methods

| Method | What it is |
| --- | --- |
| Normal spread | misses spread like a bell curve, as wide as the typical past miss |
| Early-season spread | the same, with its own width while neither team has played 15 games |
| Logistic calibration | Platt scaling: a logistic regression of past wins on the predicted margin |
| Isotonic calibration | a step curve that only rises with the predicted margin, fit to past wins |

## Against simple forecasts and Vegas

Reported only. Vegas's spread becomes a win probability the same way as the normal spread, with the width of its own misses in earlier months.

| Forecast | Games | Log loss | Brier score |
| --- | --- | --- | --- |
| Coin flip | 2621 | 0.6931 | 0.2500 |
| Home team, at the home win rate of earlier games | 2621 | 0.6898 | 0.2483 |
| Model, normal spread, games with a line | 2620 | 0.5986 | 0.2063 |
| Vegas spread, normal spread from its own earlier misses | 2620 | 0.5840 | 0.2000 |

## Calibration

Home win chance from the normal spread (the method in use) and from Vegas, grouped in tenths.

| Home win chance | Model games | Model said | Home won, model | Vegas games | Vegas said | Home won, Vegas |
| --- | --- | --- | --- | --- | --- | --- |
| 0-10% | 8 | 7.4% | 0.0% | 3 | 8.7% | 0.0% |
| 10-20% | 57 | 16.6% | 14.0% | 124 | 15.9% | 10.5% |
| 20-30% | 193 | 25.4% | 16.1% | 214 | 24.8% | 20.6% |
| 30-40% | 303 | 35.4% | 33.7% | 415 | 35.3% | 34.5% |
| 40-50% | 406 | 45.5% | 43.8% | 220 | 43.9% | 47.3% |
| 50-60% | 492 | 55.0% | 52.8% | 326 | 55.3% | 51.2% |
| 60-70% | 498 | 65.0% | 64.7% | 585 | 64.9% | 62.1% |
| 70-80% | 401 | 75.0% | 75.3% | 414 | 75.2% | 77.1% |
| 80-90% | 221 | 84.2% | 84.6% | 286 | 84.3% | 86.0% |
| 90-100% | 42 | 92.3% | 97.6% | 33 | 91.9% | 97.0% |

## Score ranges

The middle 80% of outcomes under a normal spread: the prediction plus or minus 1.28 typical misses. If the spread is right, 80% of games land inside. The typical miss is the one fit on the whole record, which daily predictions use.

| Prediction | Typical miss, points | Range, points | Games inside the range |
| --- | --- | --- | --- |
| Margin | 13.6 | ± 17.5 | 78.7% |
| Total | 18.5 | ± 23.8 | 81.0% |

## Cover and over/under odds

On validation games with an ESPN closing line, leaving out games that landed exactly on the line (pushes). Each method's log loss is checked against a coin flip; noise is 2 standard errors, measured month to month. Daily predictions show these odds only from a method that beats a coin flip; if both do, the lower log loss.

| Bet | Games | Log loss | Brier score | Log loss change | Noise | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Home team covers: Coin flip | 2620 | 0.6931 | 0.2500 | - | - | baseline |
| Home team covers: Normal spread around the model | 2620 | 0.7047 | 0.2554 | +0.0115 | 0.0106 | worse |
| Home team covers: Calibrated to the gap | 2620 | 0.6933 | 0.2501 | +0.0001 | 0.0011 | tie |
| Over: Coin flip | 2620 | 0.6931 | 0.2500 | - | - | baseline |
| Over: Normal spread around the model | 2620 | 0.7116 | 0.2588 | +0.0185 | 0.0091 | worse |
| Over: Calibrated to the gap | 2620 | 0.6934 | 0.2501 | +0.0003 | 0.0003 | tie |

**Home team covers: no method beats a coin flip.**

**Over: no method beats a coin flip.**

The model's side of the line won 51.8% of 2,620 games on the spread and 48.7% of 2,620 games on the total. At standard odds (risk 110 to win 100), breaking even takes 52.4%.

### Odds methods

| Method | What it is |
| --- | --- |
| Coin flip | 50% every game |
| Normal spread around the model | the model's number as the center, its typical miss as the width |
| Calibrated to the gap | a logistic regression of past results on how far the model was from the line |
