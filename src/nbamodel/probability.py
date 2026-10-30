"""Decision 13: how sure is each prediction?

The regression gives one number per team. The model's own misses show how far real games land
from those numbers, which turns a predicted margin into a win probability, a range of likely
scores, and odds against a betting line. Every month is scored with probabilities fit only on
misses from the months before it, so the games being scored stay unseen, the same way they do for
the regression. The record of misses starts with the season before the validation seasons,
predicted month by month the same way.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config, evaluate, vegas

# Fixed before any results (decision 13), simplest first: config name -> (name, what it is).
WIN_METHODS = {
    "normal": ("Normal spread", "misses spread like a bell curve, as wide as the typical past miss"),
    "early": ("Early-season spread", "the same, with its own width while neither team has played "
              f"{config.EARLY_SEASON_GAMES} games"),
    "logistic": ("Logistic calibration", "Platt scaling: a logistic regression of past wins on the predicted margin"),
    "isotonic": ("Isotonic calibration", "a step curve that only rises with the predicted margin, fit to past wins"),
}
# Odds against a line: (name, the line's name, actual result, model's number, line). The home team
# covers when its margin beats the line's expected home margin; a game goes over when its total
# beats the line.
BETS = {
    "cover": ("Home team covers", "the spread", "MARGIN", "PRED_MARGIN", "VEGAS_HOME_MARGIN"),
    "over": ("Over", "the total", "TOTAL", "PRED_TOTAL", "VEGAS_TOTAL"),
}
LINE_METHODS = {
    "coin": ("Coin flip", "50% every game"),
    "normal": ("Normal spread around the model", "the model's number as the center, its typical miss as the width"),
    "calibrated": ("Calibrated to the gap", "a logistic regression of past results on how far the model was from "
                   "the line"),
}
RANGE = 0.8                    # score ranges hold the middle 80% of outcomes
BREAK_EVEN = 110 / 210         # share of bets a side must win at standard odds (risk 110 to win 100)


def record_seasons() -> list[str]:
    """Seasons whose months make up the record of misses: every training season before the test season."""
    return [s for s in config.TRAIN_SEASONS if s < config.TEST_SEASON]


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, float)))))


def _logistic(x, y, intercept: bool = True) -> tuple[float, float]:
    """Logistic regression of y on one input, effectively unpenalized: (intercept, slope)."""
    fit = LogisticRegression(C=1e6, fit_intercept=intercept).fit(np.asarray(x, float).reshape(-1, 1), y)
    return float(np.ravel(fit.intercept_)[0]), float(fit.coef_[0, 0])


def _bounded(p) -> np.ndarray:
    return np.clip(p, config.PROBABILITY_FLOOR, 1 - config.PROBABILITY_FLOOR)


def fit_win(method: str, history: pd.DataFrame) -> dict:
    """What one way of turning a predicted margin into a win probability learns from past games."""
    miss = history["MARGIN"] - history["PRED_MARGIN"]
    won = (history["MARGIN"] > 0).astype(int)
    if method == "normal":
        return {"sd": _rms(miss)}
    if method == "early":
        early = history["EARLY"].astype(bool)
        return {"sd": _rms(miss[~early] if (~early).any() else miss),
                "sd_early": _rms(miss[early] if early.any() else miss)}
    if method == "logistic":
        a, b = _logistic(history["PRED_MARGIN"], won)
        return {"intercept": a, "slope": b}
    if method == "isotonic":
        fit = IsotonicRegression(y_min=config.PROBABILITY_FLOOR, y_max=1 - config.PROBABILITY_FLOOR,
                                 out_of_bounds="clip").fit(history["PRED_MARGIN"], won)
        return {"margins": fit.X_thresholds_.tolist(), "probs": fit.y_thresholds_.tolist()}
    raise ValueError(f"Unknown win probability method {method!r}")


def win_probability(method: str, params: dict, games: pd.DataFrame) -> np.ndarray:
    """Chance the home team wins each game, from PRED_MARGIN (and EARLY, for the early-season spread)."""
    margin = games["PRED_MARGIN"].to_numpy(float)
    if method == "normal":
        p = norm.cdf(margin / params["sd"])
    elif method == "early":
        p = norm.cdf(margin / np.where(games["EARLY"].to_numpy(bool), params["sd_early"], params["sd"]))
    elif method == "logistic":
        p = 1 / (1 + np.exp(-(params["intercept"] + params["slope"] * margin)))
    elif method == "isotonic":
        p = np.interp(margin, params["margins"], params["probs"])
    else:
        raise ValueError(f"Unknown win probability method {method!r}")
    return _bounded(p)


def fit_line(method: str, bet: str, history: pd.DataFrame) -> dict:
    """What one way of giving odds against a line learns from past games. Games landing exactly on
    the line (pushes) are left out, as a sportsbook refunds them."""
    _, _, actual, predicted, line = BETS[bet]
    if method == "coin":
        return {}
    if method == "normal":
        return {"sd": _rms(history[actual] - history[predicted])}
    if method == "calibrated":
        h = history.dropna(subset=[line])
        h = h[h[actual] != h[line]]
        # No intercept: a model that agrees with the line says 50%.
        _, slope = _logistic(h[predicted] - h[line], (h[actual] > h[line]).astype(int), intercept=False)
        return {"slope": slope}
    raise ValueError(f"Unknown line method {method!r}")


def line_probability(method: str, params: dict, bet: str, games: pd.DataFrame) -> np.ndarray:
    """Chance the home team covers (bet "cover") or the game goes over ("over") the line."""
    *_, predicted, line = BETS[bet]
    gap = (games[predicted] - games[line]).to_numpy(float)
    if method == "coin":
        p = np.full(len(gap), 0.5)
    elif method == "normal":
        p = norm.cdf(gap / params["sd"])
    elif method == "calibrated":
        p = 1 / (1 + np.exp(-params["slope"] * gap))
    else:
        raise ValueError(f"Unknown line method {method!r}")
    return _bounded(p)


def record(feats: pd.DataFrame, seasons: list[str], abbr: dict | None = None,
           lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every month of the given seasons predicted by a model fit on the games before it, as the
    evaluation does, with whether each game is early season and, given lines, its closing line."""
    preds, _ = evaluate.cross_validate(feats, seasons, abbr=abbr)
    early = feats.groupby("GAME_ID")["SEASON_GAMES"].max() < config.EARLY_SEASON_GAMES
    preds["EARLY"] = preds["GAME_ID"].map(early).fillna(False).astype(bool)
    if lines is not None:
        preds = vegas.attach(preds, lines)
    return preds


def walk_forward(rec: pd.DataFrame, scored: pd.DataFrame, fit, predict, needs=None) -> pd.Series:
    """Probabilities for the scored games, each month's fit only on record games before it.
    With needs (a column), only earlier games that have it count, and a month with none is left out."""
    out = []
    for fold, games in scored.groupby("FOLD"):
        history = rec[rec["GAME_DATE"] < pd.Period(fold, "M").start_time]
        if needs is not None:
            history = history.dropna(subset=[needs])
        if len(history):
            out.append(pd.Series(predict(fit(history), games), index=games.index))
    return pd.concat(out).reindex(scored.index) if out else pd.Series(np.nan, index=scored.index)


def log_loss(p, happened) -> np.ndarray:
    """Per game: minus the log of the chance given to what happened. Lower is better; a coin flip
    scores 0.693 on every game."""
    p, y = np.asarray(p, float), np.asarray(happened, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, happened) -> np.ndarray:
    """Per game: the squared miss of the probability (0 is perfect; a coin flip scores 0.25)."""
    return (np.asarray(p, float) - np.asarray(happened, float)) ** 2


def _verdict(change: float, se: float) -> str:
    if abs(change) <= config.NOISE_STANDARD_ERRORS * se:
        return "tie"
    return "better" if change < 0 else "worse"


def noise_table(first: str, probs: dict, happened, folds, base: str, base_verdict: str,
                names: dict) -> tuple[list[str], dict]:
    """Rows of log loss and Brier score, each method's log loss checked against base on the same
    games with the noise test. Returns the rows and each method's verdict."""
    k = config.NOISE_STANDARD_ERRORS
    base_loss = log_loss(probs[base], happened)
    rows, verdicts = [], {}
    for key, p in probs.items():
        loss = log_loss(p, happened)
        if key == base:
            test, verdicts[key] = ["-", "-", base_verdict], base_verdict
        else:
            change, se = evaluate.clustered_mean(loss - base_loss, folds)
            verdicts[key] = _verdict(change, se)
            test = [f"{change:+.4f}", f"{k * se:.4f}", verdicts[key]]
        rows.append(f"| {first}{names[key][0]} | {len(p)} | {loss.mean():.4f} | {brier(p, happened).mean():.4f} | "
                    + " | ".join(test) + " |")
    return rows, verdicts


def best(verdicts: dict, losses: dict, fallback):
    winners = [m for m, v in verdicts.items() if v == "better"]
    return min(winners, key=lambda m: losses[m]) if winners else fallback


def _pct(v) -> str:
    return f"{v:.1%}" if pd.notna(v) else "-"


def calibration_rows(p, happened) -> list[dict]:
    """Games in each tenth of predicted probability: how many, the average said, and how often it happened."""
    p, y = np.asarray(p, float), np.asarray(happened, float)
    bins = np.minimum((p * 10).astype(int), 9)
    return [{"games": int((bins == b).sum()),
             "said": p[bins == b].mean() if (bins == b).any() else np.nan,
             "happened": y[bins == b].mean() if (bins == b).any() else np.nan} for b in range(10)]


def run(feats: pd.DataFrame, abbr: dict | None, lines: pd.DataFrame | None) -> tuple[list[str], dict]:
    """Markdown for results/probability.md, and what the test picked: the win probability method,
    and the method for cover and over/under odds (None when nothing beats a coin flip)."""
    rec = record(feats, record_seasons(), abbr, lines)
    scored = rec[rec["SEASON"].isin(config.VALIDATION_SEASONS)]
    won = (scored["MARGIN"] > 0).astype(int)
    k = config.NOISE_STANDARD_ERRORS
    current = config.WIN_PROBABILITY
    first = pd.Period(rec["FOLD"].min(), "M").start_time
    intro = ["# How sure is each prediction", "",
             f"Decision 13 in docs/decisions.md. Every month from {first:%B %Y} on is predicted by a model fit on the "
             "games before it, the same way as the evaluation; how far those predictions missed is the record. Each "
             "validation month's probabilities are fit only on the record before that month, so no month helps "
             "score itself.", "",
             "Log loss scores a probability: minus the log of the chance given to what happened, averaged over "
             "games. Lower is better, and a coin flip scores 0.693. Changes are in average log loss against the "
             f"method in use on the same games; noise is {k} standard errors, measured month to month. A method "
             "replaces the one in use only if it beats the noise; if several do, the lowest log loss wins. Brier "
             "score, the average squared miss of the probability, is reported but doesn't decide.", ""]

    # Win probability: every method, with the method in use as the base.
    probs = {m: walk_forward(rec, scored, lambda h, m=m: fit_win(m, h), lambda p, g, m=m: win_probability(m, p, g))
             for m in WIN_METHODS}
    probs = {current: probs.pop(current), **probs}
    rows, verdicts = noise_table("", probs, won, scored["FOLD"], current, "current", WIN_METHODS)
    losses = {m: log_loss(p, won).mean() for m, p in probs.items()}
    picked = best(verdicts, losses, current)
    in_use = WIN_METHODS[current][0].lower()
    result = f"The {in_use} stays" if picked == current else f"{WIN_METHODS[picked][0]} beats the {in_use}"
    lines_out = [*intro, "## Win probability", "",
                 "| Method | Games | Log loss | Brier score | Log loss change | Noise | Verdict |",
                 "|" + " --- |" * 7, *rows, "", f"**{result}.**",
                 "", "### Win probability methods", "", "| Method | What it is |", "| --- | --- |",
                 *[f"| {name} | {about} |" for name, about in WIN_METHODS.values()], ""]

    # Simple forecasts and Vegas, reported only.
    home_rate = walk_forward(rec, scored, lambda h: float((h["MARGIN"] > 0).mean()),
                             lambda rate, g: np.full(len(g), rate))
    bench = [("Coin flip", pd.Series(0.5, index=scored.index), scored.index),
             ("Home team, at the home win rate of earlier games", home_rate, scored.index)]
    vegas_p = None
    if lines is not None:
        with_line = scored[scored["VEGAS_HOME_MARGIN"].notna()]
        vegas_p = walk_forward(rec, with_line, lambda h: _rms(h["MARGIN"] - h["VEGAS_HOME_MARGIN"]),
                               lambda sd, g: _bounded(norm.cdf(g["VEGAS_HOME_MARGIN"] / sd)),
                               needs="VEGAS_HOME_MARGIN").dropna()
        bench += [(f"Model, {in_use}, games with a line", probs[current], vegas_p.index),
                  ("Vegas spread, normal spread from its own earlier misses", vegas_p, vegas_p.index)]
    lines_out += ["## Against simple forecasts and Vegas", "",
                  "Reported only. Vegas's spread becomes a win probability the same way as the normal spread, with "
                  "the width of its own misses in earlier months.", "",
                  "| Forecast | Games | Log loss | Brier score |", "| --- | --- | --- | --- |"]
    for name, p, idx in bench:
        lines_out.append(f"| {name} | {len(idx)} | {log_loss(p[idx], won[idx]).mean():.4f} | "
                         f"{brier(p[idx], won[idx]).mean():.4f} |")

    # Calibration: when the forecast said 70%, did the home team win about 70% of the time?
    model_bins = calibration_rows(probs[current], won)
    vegas_bins = calibration_rows(vegas_p, won[vegas_p.index]) if vegas_p is not None else [{}] * 10
    lines_out += ["", "## Calibration", "",
                  f"Home win chance from the {in_use} (the method in use) and from Vegas, "
                  "grouped in tenths.", "",
                  "| Home win chance | Model games | Model said | Home won, model | Vegas games | Vegas said "
                  "| Home won, Vegas |", "|" + " --- |" * 7]
    for b, (m, v) in enumerate(zip(model_bins, vegas_bins)):
        lines_out.append(f"| {10 * b}-{10 * b + 10}% | {m['games']} | {_pct(m['said'])} | {_pct(m['happened'])} | "
                         f"{v.get('games', '-')} | {_pct(v.get('said'))} | {_pct(v.get('happened'))} |")

    # Score ranges from the same normal spread, on margin and total.
    z = norm.ppf(0.5 + RANGE / 2)
    lines_out += ["", "## Score ranges", "",
                  f"The middle {RANGE:.0%} of outcomes under a normal spread: the prediction plus or minus "
                  f"{z:.2f} typical misses. If the spread is right, {RANGE:.0%} of games land inside. The typical "
                  "miss is the one fit on the whole record, which daily predictions use.", "",
                  "| Prediction | Typical miss, points | Range, points | Games inside the range |",
                  "| --- | --- | --- | --- |"]
    for name, actual, pred in (("Margin", "MARGIN", "PRED_MARGIN"), ("Total", "TOTAL", "PRED_TOTAL")):
        sd = walk_forward(rec, scored, lambda h: _rms(h[actual] - h[pred]), lambda s, g: np.full(len(g), s))
        inside = ((scored[actual] - scored[pred]).abs() <= z * sd).mean()
        whole = _rms(rec[actual] - rec[pred])
        lines_out.append(f"| {name} | {whole:.1f} | ± {z * whole:.1f} | {inside:.1%} |")

    picks = {"win": picked, "cover": None, "over": None}
    if lines is not None:
        lines_out += ["", "## Cover and over/under odds", "",
                      "On validation games with an ESPN closing line, leaving out games that landed exactly on the "
                      f"line (pushes). Each method's log loss is checked against a coin flip; noise is {k} standard "
                      "errors, measured month to month. Daily predictions show these odds only from a method that "
                      "beats a coin flip; if both do, the lower log loss.", "",
                      "| Bet | Games | Log loss | Brier score | Log loss change | Noise | Verdict |",
                      "|" + " --- |" * 7]
        sides = []
        for bet, (bet_name, line_name, actual, predicted, line) in BETS.items():
            games = scored[scored[line].notna() & (scored[actual] != scored[line])]
            happened = (games[actual] > games[line]).astype(int)
            bet_probs = {m: walk_forward(rec, games, lambda h, m=m: fit_line(m, bet, h),
                                         lambda p, g, m=m: line_probability(m, p, bet, g), needs=line)
                         for m in LINE_METHODS}
            keep = pd.concat(bet_probs, axis=1).dropna().index
            bet_probs = {m: p[keep] for m, p in bet_probs.items()}
            games, happened = games.loc[keep], happened[keep]
            rows, verdicts = noise_table(f"{bet_name}: ", bet_probs, happened, games["FOLD"], "coin", "baseline",
                                         LINE_METHODS)
            lines_out += rows
            losses = {m: log_loss(p, happened).mean() for m, p in bet_probs.items()}
            picks[bet] = best({m: v for m, v in verdicts.items() if m != "coin"}, losses, None)
            # The side the model was on: did it win? Ties with the line pick no side.
            gap = games[predicted] - games[line]
            sided = gap != 0
            sides.append(f"{((gap > 0) == (happened == 1))[sided].mean():.1%} of {sided.sum():,} games on {line_name}")
        for bet, (bet_name, *_) in BETS.items():
            verdict = (f"{LINE_METHODS[picks[bet]][0].lower()} beats a coin flip" if picks[bet]
                       else "no method beats a coin flip")
            lines_out += ["", f"**{bet_name}: {verdict}.**"]
        lines_out += ["", f"The model's side of the line won {' and '.join(sides)}. At standard odds (risk 110 to "
                      f"win 100), breaking even takes {BREAK_EVEN:.1%}.", "", "### Odds methods", "",
                      "| Method | What it is |", "| --- | --- |",
                      *[f"| {name} | {about} |" for name, about in LINE_METHODS.values()]]
    return lines_out, picks


def fit_live(feats: pd.DataFrame, abbr: dict | None = None, lines: pd.DataFrame | None = None) -> dict:
    """What daily predictions need, fit on the whole record of misses (never the test season)."""
    seasons = record_seasons()
    rec = record(evaluate.validation_rows(feats), seasons, abbr, lines)
    out = {"win_method": config.WIN_PROBABILITY, "win": fit_win(config.WIN_PROBABILITY, rec),
           "margin_sd": _rms(rec["MARGIN"] - rec["PRED_MARGIN"]), "total_sd": _rms(rec["TOTAL"] - rec["PRED_TOTAL"]),
           "games": int(len(rec)), "seasons": seasons}
    for bet, method in (("cover", config.COVER_ODDS), ("over", config.OVER_UNDER_ODDS)):
        if method:
            out[bet] = {"method": method, **fit_line(method, bet, rec)}
    return out
