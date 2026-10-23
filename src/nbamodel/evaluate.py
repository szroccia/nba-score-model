"""Model evaluation: rolling-origin cross-validation, a locked test season, tuning curves
and the Vegas benchmark.

Every fold refits the regression on all games before the first day of a month and
predicts that month, the same way the live model is used. Settings are compared on the
validation seasons only. The test season is scored by `evaluate --final` and nothing
else, so its result is an honest estimate of accuracy on games the model has never seen.
"""

import subprocess

import numpy as np
import pandas as pd

from . import config, model

def monthly_folds(feats: pd.DataFrame, seasons: list[str]):
    """(season, month, training rows, test rows) for every month of the given seasons.

    Training rows are every game before the month starts; test rows are that month's games
    in which both teams have enough history to be rated (the same rule training uses)."""
    month = feats["GAME_DATE"].dt.to_period("M")
    for season in seasons:
        for m in sorted(month[feats["SEASON"] == season].unique()):
            train = feats[feats["GAME_DATE"] < m.start_time]
            test = feats[(feats["SEASON"] == season) & (month == m)]
            rated = test.groupby("GAME_ID")["GAMES_PLAYED"].min() >= model.MIN_GAMES_FOR_TRAINING
            test = test[test["GAME_ID"].isin(rated[rated].index)]
            if not train.empty and not test.empty:
                yield season, str(m), train, test


def game_frame(rows: pd.DataFrame, pred: np.ndarray, abbr: dict | None = None) -> pd.DataFrame:
    """One row per game with actual and predicted margin and total (home perspective)."""
    rows = rows.assign(PRED=pred)
    home = rows[rows["IS_HOME"] == 1].set_index("GAME_ID")
    away = rows[rows["IS_HOME"] == 0].set_index("GAME_ID")
    both = home.index.intersection(away.index)
    home, away = home.loc[both], away.loc[both]
    games = pd.DataFrame({
        "GAME_DATE": home["GAME_DATE"], "SEASON": home["SEASON"],
        "HOME_ID": home["TEAM_ID"], "AWAY_ID": away["TEAM_ID"],
        "HOME_PTS": home["PTS"], "AWAY_PTS": away["PTS"],
        "PRED_HOME": home["PRED"], "PRED_AWAY": away["PRED"],
    }).reset_index()
    if abbr is not None:
        games["HOME"] = games["HOME_ID"].map(abbr)
        games["AWAY"] = games["AWAY_ID"].map(abbr)
    games["MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games["TOTAL"] = games["HOME_PTS"] + games["AWAY_PTS"]
    games["PRED_MARGIN"] = games["PRED_HOME"] - games["PRED_AWAY"]
    games["PRED_TOTAL"] = games["PRED_HOME"] + games["PRED_AWAY"]
    return games


def metrics(games: pd.DataFrame, margin_col: str = "PRED_MARGIN", total_col: str = "PRED_TOTAL") -> dict:
    """Average misses in points on the margin and total, winner %, and total bias."""
    bias = float((games[total_col] - games["TOTAL"]).mean())
    return {
        "games": int(len(games)),
        "margin_mae": float((games["MARGIN"] - games[margin_col]).abs().mean()),
        "total_mae": float((games["TOTAL"] - games[total_col]).abs().mean()),
        "winner_pct": float((games["MARGIN"] * games[margin_col] > 0).mean()),
        "total_bias": bias,
        "abs_total_bias": abs(bias),
    }


def cross_validate(feats: pd.DataFrame, seasons: list[str], alpha: float | None = None,
                   abbr: dict | None = None, max_train_rows: int | None = None):
    """Predictions for every fold, plus the in-sample (training) metrics of each fit.

    With max_train_rows, each fit uses only the most recent rows of its training data."""
    preds, train_scores = [], []
    for season, month, train, test in monthly_folds(feats, seasons):
        train = train[train["GAMES_PLAYED"] >= model.MIN_GAMES_FOR_TRAINING].sort_values("GAME_DATE")
        if max_train_rows is not None:
            if len(train) < max_train_rows:
                continue
            train = train.tail(max_train_rows)
        fitted = model.fit(train, alpha)
        preds.append(game_frame(test, model.predict(fitted, test), abbr).assign(FOLD=month))
        train_scores.append(metrics(game_frame(train, model.predict(fitted, train))))
    if not preds:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(preds, ignore_index=True), pd.DataFrame(train_scores)


def validation_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Everything except the locked test season (and anything after it)."""
    start = feats.loc[feats["SEASON"] == config.TEST_SEASON, "GAME_DATE"].min()
    keep = feats["SEASON"] != config.TEST_SEASON
    if pd.notna(start):
        keep &= feats["GAME_DATE"] < start
    return feats[keep]


def rank(table: pd.DataFrame) -> pd.DataFrame:
    """Sort by the selection metrics fixed in config (rounded to 0.01 points, so noise-level
    differences tie) and, on a tie, prefer the simpler model (larger alpha)."""
    keys = [table[m].round(2) for m in config.SELECTION_METRICS]
    order = pd.DataFrame({f"k{i}": k for i, k in enumerate(keys)})
    if "alpha" in table:
        order["k_alpha"] = -table["alpha"]
    return table.loc[order.sort_values(list(order.columns)).index]


def complexity_curve(feats: pd.DataFrame, seasons: list[str], alphas: list[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        preds, train = cross_validate(feats, seasons, alpha)
        rows.append({"alpha": alpha, **metrics(preds),
                     "train_margin_mae": float(np.average(train["margin_mae"], weights=train["games"]))})
    return pd.DataFrame(rows)


def learning_curve(feats: pd.DataFrame, season: str, sizes: list[int]) -> pd.DataFrame:
    """Accuracy against the amount of recent training history, on one validation season.

    Uses only folds that have at least the largest size available, so every size is
    scored on the same months."""
    available = [len(train[train["GAMES_PLAYED"] >= model.MIN_GAMES_FOR_TRAINING])
                 for _, _, train, _ in monthly_folds(feats, [season])]
    sizes = [n for n in sizes if available and n <= min(available)]
    rows = []
    for n in sizes:
        preds, train = cross_validate(feats, [season], max_train_rows=n)
        rows.append({"train_rows": n, "val_margin_mae": metrics(preds)["margin_mae"],
                     "train_margin_mae": float(np.average(train["margin_mae"], weights=train["games"])),
                     "games": len(preds)})
    return pd.DataFrame(rows)


def vegas_comparison(preds: pd.DataFrame, vegas_lines: pd.DataFrame) -> pd.DataFrame:
    """Per fold and overall: the model against the closing line, on games that have one."""
    from . import vegas

    matched = vegas.attach(preds, vegas_lines).dropna(subset=["VEGAS_HOME_MARGIN"])
    rows = []
    for fold, games in [*matched.groupby("FOLD"), ("All", matched)]:
        ours, theirs = metrics(games), metrics(games, "VEGAS_HOME_MARGIN", "VEGAS_TOTAL")
        rows.append({"fold": fold, "games": ours["games"],
                     "coverage": ours["games"] / max(1, len(preds) if fold == "All" else (preds["FOLD"] == fold).sum()),
                     "model_margin_mae": ours["margin_mae"], "vegas_margin_mae": theirs["margin_mae"],
                     "model_total_mae": ours["total_mae"], "vegas_total_mae": theirs["total_mae"],
                     "model_winner_pct": ours["winner_pct"], "vegas_winner_pct": theirs["winner_pct"]})
    return pd.DataFrame(rows)


def settings() -> dict:
    return {"alpha": config.RIDGE_ALPHA, "playoff_games": config.PLAYOFF_GAMES,
            "half_life": config.RATING_HALF_LIFE_GAMES, "prior_season_weight": config.PRIOR_SEASON_GAME_WEIGHT,
            "shrink_games": config.SHRINK_GAMES, "feature_version": config.FEATURE_VERSION}


def _md_table(df: pd.DataFrame, columns: dict) -> list[str]:
    """Markdown table; columns maps a frame column to (header, formatter)."""
    lines = ["| " + " | ".join(h for h, _ in columns.values()) + " |", "|" + " --- |" * len(columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c, (_, fmt) in columns.items()) + " |")
    return lines


def _f2(v):
    return f"{v:.2f}"


def _int(v):
    return str(int(v))


def _pct(v):
    return f"{v:.1%}"


def _row(name: str, m: dict, *extra: str) -> str:
    cells = [name, str(m["games"]), f"{m['margin_mae']:.2f}", f"{m['total_mae']:.2f}",
             f"{m['winner_pct']:.1%}", f"{m['total_bias']:+.2f}", *extra]
    return "| " + " | ".join(cells) + " |"


METRIC_HEADER = ["| Model | Games | Margin MAE | Total MAE | Winner % | Total bias |",
                 "| --- | --- | --- | --- | --- | --- |"]


def write_report(out_dir, seasons, current: dict, vegas_table, curve, learn, extra=None) -> list[str]:
    best = rank(curve).iloc[0]
    lines = [
        "# Evaluation", "",
        f"Validation seasons: {', '.join(seasons)}. Each month is predicted by a model refit on every "
        "game before that month (rolling-origin cross-validation).",
        f"The test season ({config.TEST_SEASON}) is locked and not used here; see test_log.md.",
        "Settings are ranked by " + ", then ".join(config.SELECTION_METRICS) + " (fixed before running).", "",
        "## Current settings", "",
        "Settings: " + ", ".join(f"{k} {v}" for k, v in settings().items()) + ".", "",
        *METRIC_HEADER, _row("Ridge regression", current),
    ]
    if vegas_table is not None and not vegas_table.empty:
        total = vegas_table[vegas_table["fold"] == "All"].iloc[0]
        lines += ["", "## Against the Vegas closing line", "",
                  f"On the {int(total['games'])} validation games with an ESPN closing line "
                  f"({total['coverage']:.0%} of them):", ""]
        lines += _md_table(vegas_table, {
            "fold": ("Month", str), "games": ("Games", _int),
            "model_margin_mae": ("Margin MAE, model", _f2), "vegas_margin_mae": ("Margin MAE, Vegas", _f2),
            "model_total_mae": ("Total MAE, model", _f2), "vegas_total_mae": ("Total MAE, Vegas", _f2),
            "model_winner_pct": ("Winner %, model", _pct), "vegas_winner_pct": ("Winner %, Vegas", _pct)})
    lines += ["", "## Complexity curve (ridge alpha)", "",
              "Larger alpha shrinks the coefficients toward zero (a simpler model). Training error is "
              "in-sample on each fold's training games.", ""]
    lines += _md_table(curve, {
        "alpha": ("Alpha", lambda v: f"{v:g}"), "margin_mae": ("Margin MAE, validation", _f2),
        "train_margin_mae": ("Margin MAE, training", _f2), "total_mae": ("Total MAE, validation", _f2),
        "winner_pct": ("Winner %", _pct), "total_bias": ("Total bias", lambda v: f"{v:+.2f}")})
    lines += ["", f"Best on validation: alpha {best['alpha']:g} (config.RIDGE_ALPHA is {config.RIDGE_ALPHA:g}; "
              "changing it is a modeling decision, so it is not changed automatically)."]
    if learn is not None and not learn.empty:
        lines += ["", "## Learning curve", "",
                  f"Each fit uses only the most recent N team-games of history; scored on {seasons[-1]} "
                  "months that have at least the largest N available.", ""]
        lines += _md_table(learn, {
            "train_rows": ("Training team-games", _int), "val_margin_mae": ("Margin MAE, validation", _f2),
            "train_margin_mae": ("Margin MAE, training", _f2), "games": ("Validation games", _int)})
    if extra:
        lines += ["", *extra]
    (out_dir / "evaluation.md").write_text("\n".join(lines) + "\n")
    return lines


def run(feats: pd.DataFrame, vegas_lines: pd.DataFrame | None, abbr: dict, extra=None) -> list[str]:
    """Validation-only evaluation: writes results/evaluation.md (charts come from `report`)."""
    feats = validation_rows(feats)
    seasons = [s for s in config.VALIDATION_SEASONS if s in set(feats["SEASON"])]
    preds, _ = cross_validate(feats, seasons, abbr=abbr)
    vegas_table = vegas_comparison(preds, vegas_lines) if vegas_lines is not None else None
    curve = complexity_curve(feats, seasons, config.ALPHA_GRID)
    learn = learning_curve(feats, seasons[-1], config.LEARNING_CURVE_SIZES)
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    return write_report(out, seasons, metrics(preds), vegas_table, curve, learn, extra)


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=config.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return (sha or "unknown") + (" (uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def final(feats: pd.DataFrame, vegas_lines: pd.DataFrame | None, abbr: dict) -> list[str]:
    """Score the locked test season once with the current settings and log the run."""
    preds, _ = cross_validate(feats, [config.TEST_SEASON], abbr=abbr)
    if preds.empty:
        raise SystemExit(f"No games for test season {config.TEST_SEASON}.")
    ours = metrics(preds)
    rows = [_row("Ridge regression", ours)]
    if vegas_lines is not None:
        table = vegas_comparison(preds, vegas_lines)
        total = table[table["fold"] == "All"]
        if not total.empty:
            t = total.iloc[0]
            rows.append(f"| Vegas closing line ({int(t['games'])} games with a line) | {int(t['games'])} | "
                        f"{t['vegas_margin_mae']:.2f} | {t['vegas_total_mae']:.2f} | {t['vegas_winner_pct']:.1%} | |")
            rows.append(f"| Ridge regression, same games | {int(t['games'])} | {t['model_margin_mae']:.2f} | "
                        f"{t['model_total_mae']:.2f} | {t['model_winner_pct']:.1%} | |")
    path = config.ROOT / "results" / "test_log.md"
    path.parent.mkdir(exist_ok=True)
    entry = [f"## {pd.Timestamp.now():%Y-%m-%d %H:%M}, commit {_commit()}", "",
             "Settings: " + ", ".join(f"{k} {v}" for k, v in settings().items()) + ".", "",
             *METRIC_HEADER, *rows, ""]
    if not path.exists():
        entry = ["# Test season log", "",
                 f"Every run of `evaluate --final` on the locked test season ({config.TEST_SEASON}) is "
                 "recorded here, so the number of looks at the test set is visible.", "", *entry]
    with open(path, "a") as f:
        f.write("\n".join(entry) + "\n")
    return entry
