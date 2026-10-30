"""Decision 11: other kinds of model against ridge regression, on the same inputs and folds.

Every candidate predicts a team's points from the same inputs as ridge. In each monthly fold it
first picks its own setting (how many neighbors, how deep the trees and so on) on the most recent
fifth of that fold's training games, then refits on all of them and predicts the month. No choice
ever sees the month being predicted, so the validation games stay a fair test for every model, the
same way they are for ridge. The blend weighs every candidate's predictions, with weights learned
on that same recent fifth.
"""

import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config, evaluate, features

# Share of each fold's training games, the most recent ones, that candidates pick settings on.
INNER_SHARE = 0.2
BLEND = "Blend of all seven"


def _scaled(estimator):
    """Inputs and points put on a common scale, which distance- and gradient-based models need."""
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), estimator), transformer=StandardScaler())


# Fixed before any results (decision 11). Each: (name, what it is, [(setting, make a model)]).
# Settings run from simplest to most complex, and a near tie (within 0.01 points) goes to the
# earlier one. Ridge keeps the alpha decision 7 chose.
CANDIDATES = [
    ("Ridge regression", "the current model: a weighted sum of the inputs",
     [(f"alpha {config.RIDGE_ALPHA:g}", lambda: Ridge(alpha=config.RIDGE_ALPHA))]),
    ("Elastic net", "a weighted sum that can also drop inputs",
     [(f"penalty {a:g}, L1 share {r:g}", lambda a=a, r=r: _scaled(ElasticNet(alpha=a, l1_ratio=r)))
      for a in (0.1, 0.01, 0.001) for r in (0.9, 0.1)]),
    ("k-nearest neighbors", "the average of the most similar past team-games",
     [(f"{k} neighbors", lambda k=k: _scaled(KNeighborsRegressor(n_neighbors=k))) for k in (400, 200, 100, 50)]),
    ("Support vector regression", "a smooth curve that ignores small misses",
     [(f"C {c:g}", lambda c=c: _scaled(SVR(C=c, epsilon=0.1))) for c in (0.3, 1, 3)]),
    ("Random forest", "the average of 300 decision trees, each on a resample of the games",
     [(f"leaves of {leaf}+, {share:.0%} of inputs per split",
       lambda leaf=leaf, share=share: RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf,
                                                            max_features=share, n_jobs=-1, random_state=0))
      for leaf in (100, 30, 10) for share in (0.5, 1.0)]),
    ("Gradient boosting", "small trees added one at a time, each fixing the last ones' misses",
     [(f"depth {d}, {n} trees",
       lambda d=d, n=n: HistGradientBoostingRegressor(learning_rate=0.05, max_depth=d, max_iter=n,
                                                      early_stopping=False, random_state=0))
      for d in (2, 3, 4) for n in (100, 300)]),
    ("Neural network", "a small multilayer perceptron",
     [(f"{'x'.join(map(str, h))} units, penalty {a:g}",
       lambda h=h, a=a: _scaled(MLPRegressor(hidden_layer_sizes=h, alpha=a, max_iter=500, random_state=0)))
      for h in ((16,), (32, 16)) for a in (0.1, 0.001)]),
]


def _score(rows: pd.DataFrame, pred: np.ndarray) -> dict:
    return evaluate.metrics(evaluate.game_frame(rows, pred))


def _key(m: dict) -> list:
    return [round(m[c], 2) for c in config.SELECTION_METRICS]


def run_folds(feats: pd.DataFrame, seasons: list[str], candidates=None, log=print):
    """Every candidate and the blend on every monthly fold of the given seasons.

    Returns (predictions by model, one row per game as cross_validate gives them; one row per
    fold and model with the setting picked, its blend weight and its training error)."""
    candidates = candidates or CANDIDATES
    names = features.inputs()
    preds = {name: [] for name, _, _ in candidates} | {BLEND: []}
    folds = []
    start = time.time()
    for season, month, train, test in evaluate.monthly_folds(feats, seasons):
        train = train.sort_values("GAME_DATE")
        cut = train["GAME_DATE"].iloc[int(len(train) * (1 - INNER_SHARE))]
        early, recent = train[train["GAME_DATE"] < cut], train[train["GAME_DATE"] >= cut]
        inner, outer, rows = [], [], []
        for name, _, settings in candidates:
            tried = {label: make().fit(early[names], early["PTS"]).predict(recent[names]) for label, make in settings}
            picked = min(tried, key=lambda label: _key(_score(recent, tried[label])))  # first wins a tie
            fitted = dict(settings)[picked]().fit(train[names], train["PTS"])
            inner.append(tried[picked])
            outer.append(fitted.predict(test[names]))
            preds[name].append(evaluate.game_frame(test, outer[-1]).assign(FOLD=month))
            rows.append({"fold": month, "model": name, "setting": picked,
                         "train_margin_mae": _score(train, fitted.predict(train[names]))["margin_mae"]})
        # Non-negative weights, learned on the recent fifth, so no model can count against the others.
        blend = LinearRegression(positive=True, fit_intercept=False).fit(np.column_stack(inner), recent["PTS"])
        preds[BLEND].append(evaluate.game_frame(test, np.column_stack(outer) @ blend.coef_).assign(FOLD=month))
        for row, w in zip(rows, blend.coef_):
            row["blend_weight"] = float(w)
        folds += rows
        log(f"  {month}: done ({(time.time() - start) / 60:.0f} min so far)")
    return {name: pd.concat(p, ignore_index=True) for name, p in preds.items()}, pd.DataFrame(folds)


def report(preds: dict, folds: pd.DataFrame, candidates=None) -> tuple[list[str], str]:
    """Markdown for results/comparison.md, and the model the test picks."""
    candidates = candidates or CANDIDATES
    k = config.NOISE_STANDARD_ERRORS
    base = "Ridge regression"
    lines = ["## Against ridge regression", "",
             "| Model | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change | Margin noise "
             "| Total change | Total noise | Verdict |", "|" + " --- |" * 11]
    scores, winners = {}, []
    for name in preds:
        m = scores[name] = evaluate.metrics(preds[name])
        if name == base:
            test = ["-", "-", "-", "-", "current"]
        else:
            t = evaluate.noise_test(preds[base], preds[name])
            test = [f"{t['margin'][0]:+.3f}", f"{k * t['margin'][1]:.3f}",
                    f"{t['total'][0]:+.3f}", f"{k * t['total'][1]:.3f}", t["verdict"]]
            if t["verdict"] == "better":
                winners.append(name)
        lines.append(evaluate._row(name, m, *test))
    picked = min(winners, key=lambda w: [scores[w][c] for c in config.SELECTION_METRICS]) if winners else base
    lines += ["", f"**{'Ridge regression stays' if picked == base else picked + ' beats ridge regression'}.**", ""]

    # Bias and variance: a big gap between training and validation error means a model memorizes
    # past games (overfits); two high numbers mean it can't capture the pattern (underfits).
    games = folds.merge(preds[base].groupby("FOLD").size().rename("games"), left_on="fold", right_index=True)
    by_model = games.assign(weighted=games["train_margin_mae"] * games["games"]).groupby("model")
    train_err = by_model["weighted"].sum() / by_model["games"].sum()
    lines += ["## Training and validation error", "",
              "| Model | Margin MAE, training | Margin MAE, validation | Gap |", "| --- | --- | --- | --- |"]
    for name, _, _ in candidates:
        val = scores[name]["margin_mae"]
        lines.append(f"| {name} | {train_err[name]:.2f} | {val:.2f} | {val - train_err[name]:+.2f} |")

    lines += ["", "## Settings picked", "",
              "| Model | What it is | Settings tried | Picked most often | Months |", "| --- | --- | --- | --- | --- |"]
    for name, about, settings in candidates:
        label, count = Counter(folds.loc[folds["model"] == name, "setting"]).most_common(1)[0]
        lines.append(f"| {name} | {about} | {len(settings)} | {label} | {count} of {folds['fold'].nunique()} |")

    weights = folds.groupby("model", sort=False)["blend_weight"].mean()
    lines += ["", "## Blend weights", "",
              "Average weight of each model in the blend, across months.", "",
              "| Model | Weight |", "| --- | --- |", *[f"| {name} | {w:.2f} |" for name, w in weights.items()]]
    return lines, picked


def run(feats: pd.DataFrame, seasons: list[str], candidates=None, log=print) -> tuple[list[str], str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # the network's iteration cap is part of its setting
        preds, folds = run_folds(feats, seasons, candidates, log)
    return report(preds, folds, candidates)
