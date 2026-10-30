"""Command line: fetch data, train, backtest and predict.

    python -m nbamodel fetch            # download data (run from a home connection)
    python -m nbamodel validate         # sanity checks on the downloaded data
    python -m nbamodel train            # fit the regression on past seasons
    python -m nbamodel backtest         # season-by-season accuracy check
    python -m nbamodel report           # redraw the charts in docs/figures
    python -m nbamodel explain --game BOS   # how one game's prediction is built
    python -m nbamodel evaluate         # cross-validation, tuning curves and Vegas benchmark
    python -m nbamodel evaluate --compare all   # plus with/without tables for each setting decision
    python -m nbamodel tune             # test the hand-picked settings one by one (decision 10)
    python -m nbamodel compare          # other kinds of model against ridge regression (decision 11)
    python -m nbamodel injuries         # check what the injury report reader finds
    python -m nbamodel predict          # today's games with the latest injury report
    python -m nbamodel daily            # refresh this season's data, then predict
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import date as date_cls

import pandas as pd

from . import compare, config, data, evaluate, features, injuries, lineup, model, ratings, report, validate, vegas
from .names import normalize_name


def _all_seasons(current: str | None = None) -> list[str]:
    """Training seasons, the season before them (to warm up ratings), and the current one."""
    seasons = [data.previous_season(config.TRAIN_SEASONS[0])] + config.TRAIN_SEASONS
    if current and current not in seasons:
        seasons.append(current)
    return seasons


def _load(seasons: list[str], refresh_season: str | None = None):
    games = pd.concat(
        [ratings.team_game_table(data.team_games(s, refresh=(s == refresh_season))) for s in seasons],
        ignore_index=True,
    )
    if config.OPPONENT_ADJUST:  # across all seasons at once, so October opponents have a history
        games = ratings.opponent_adjust(games)
    players = pd.concat(
        [data.player_games(s, refresh=(s == refresh_season)) for s in seasons], ignore_index=True
    )
    return games, players


def cmd_fetch(args):
    current = data.season_for_date(pd.Timestamp.today())
    for season in _all_seasons(current):
        print(f"Fetching {season}...")
        data.team_games(season, refresh=True)
        data.player_games(season, refresh=True)
        data.bpm(season, refresh=True)
    data.schedule(current, refresh=True)
    data.rosters(current, refresh=True)
    for season in _evaluation_seasons():
        vegas.lines(season)  # past seasons don't change, so these download once
    print(f"Saved to {config.DATA_DIR}")


def _historical(seasons: list[str]) -> pd.DataFrame:
    # One cache per combination of feature settings, so stale features are never reused.
    fingerprint = hashlib.sha1(json.dumps(features.settings(), sort_keys=True).encode()).hexdigest()[:8]
    path = config.DATA_DIR / f"features_v{config.FEATURE_VERSION}_playoffs-{config.PLAYOFF_GAMES}_{fingerprint}.csv"
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["GAME_DATE"])
        # Gaps mean a build from before a fix (zero rating shrinkage once left them); build it again.
        complete = not cached.filter(items=features.FEATURES).isna().any().any()
        if set(seasons) <= set(cached["SEASON"]) and complete:
            return cached[cached["SEASON"].isin(seasons)]
    games, players = _load(_all_seasons())
    impact = {s: lineup.player_impact(data.bpm(data.previous_season(s))) for s in seasons}
    print(f"Building features for {', '.join(seasons)}, playoff games: {config.PLAYOFF_GAMES} (a few minutes)...")
    feats = features.historical_features(games, players, impact, seasons)
    feats.to_csv(path, index=False)
    return feats


def cmd_validate(args):
    current = data.season_for_date(pd.Timestamp.today())
    ok = validate.run(_all_seasons(current), current)
    print("\nAll checks passed." if ok else "\nSome checks need a look (WARN above).")


def cmd_train(args):
    feats = _historical(config.TRAIN_SEASONS)
    fitted = model.fit(feats)
    fitted["playoff_games"] = config.PLAYOFF_GAMES
    model.save(fitted)
    print(json.dumps(fitted, indent=2))


EARLY_SEASON_GAMES = 15
TABLE_HEADER = ["| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % | Total bias |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]


def _backtest_rows(label: str = "") -> list[str]:
    """Walk-forward backtest rows for the current settings."""
    feats = _historical(config.TRAIN_SEASONS)
    lines = []
    for i, season in enumerate(config.TRAIN_SEASONS[1:], start=1):
        if season >= config.TEST_SEASON:  # the locked test season is scored only by `evaluate --final`
            continue
        train = feats[feats["SEASON"].isin(config.TRAIN_SEASONS[:i])]
        test = feats[feats["SEASON"] == season].copy()
        fitted = model.fit(train)
        test["PRED_PTS"] = model.predict(fitted, test)
        # Option A from the plan for comparison: the ratings formula plus a fixed home edge.
        test["FORMULA_PTS"] = test["BASE_PTS"] + 1.2 * (2 * test["IS_HOME"] - 1)
        # Early season: games where neither team had played EARLY_SEASON_GAMES games yet,
        # so ratings still lean on last season.
        early_ids = test.groupby("GAME_ID")["SEASON_GAMES"].max().loc[lambda g: g < EARLY_SEASON_GAMES].index
        runs = [("Ridge (B)", "PRED_PTS", test), ("Formula (A)", "FORMULA_PTS", test),
                (f"Ridge (B), first {EARLY_SEASON_GAMES} games", "PRED_PTS", test[test["GAME_ID"].isin(early_ids)])]
        for name, col, rows in runs:
            if rows.empty:
                continue
            s = model.score(rows, col)
            lines.append(f"| {season} | {name}{label} | {s['games']} | {s['team_points_mae']} | "
                         f"{s['margin_mae']} | {s['total_mae']} | {s['winner_pct']:.1%} | {s['total_bias']:+} |")
    return lines


def cmd_backtest(args):
    intro = ["Each season is predicted by a model trained only on earlier seasons.",
             "Lineups use who actually played; ratings use only games before each date.",
             f"The locked test season ({config.TEST_SEASON}) is left out; only `evaluate --final` scores it."]
    if args.playoffs == "all":
        lines = ["# Playoff games in ratings: comparison", ""] + intro + [""] + TABLE_HEADER
        for mode in ("include", "exclude", "relative"):
            config.PLAYOFF_GAMES = mode
            lines += _backtest_rows(f", playoffs {mode}")
        config.PLAYOFF_GAMES = args.playoffs_default
        out = config.ROOT / "results" / "playoff_comparison.md"
    else:
        lines = ["# Backtest", "", *intro, f"Playoff games in ratings: {config.PLAYOFF_GAMES}.", ""] + TABLE_HEADER
        lines += _backtest_rows()
        out = config.ROOT / "results" / "backtest.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _evaluation_seasons() -> list[str]:
    return sorted(set(config.VALIDATION_SEASONS) | {config.TEST_SEASON})


def _vegas_lines() -> pd.DataFrame | None:
    try:
        return pd.concat([vegas.lines(s) for s in _evaluation_seasons()], ignore_index=True)
    except Exception as e:  # the benchmark is optional; the rest of the evaluation still runs
        print(f"Could not load Vegas lines ({e}); skipping the benchmark.")
        return None


# Settings `evaluate --compare` puts side by side: config name, (value, label) pairs, heading.
# The first value listed is the simpler one.
COMPARISONS = {
    "playoffs": ("PLAYOFF_GAMES", [("include", "playoffs include"), ("exclude", "playoffs exclude"),
                                   ("relative", "playoffs relative")], "Playoff games in ratings"),
    "sos": ("OPPONENT_ADJUST", [(False, "off"), (True, "on")], "Strength of schedule"),
    "factors": ("FOUR_FACTORS", [(False, "off"), (True, "on")], "Four factors"),
}


def _comparison(name: str, seasons: list[str]) -> list[str]:
    """A table of one setting's values on the same validation folds, each checked against the
    current value with the noise test fixed in advance (evaluate.noise_test)."""
    attr, options, heading = COMPARISONS[name]
    current = getattr(config, attr)
    preds = {}
    try:
        for value, _ in options:
            setattr(config, attr, value)
            preds[value], _ = evaluate.cross_validate(evaluate.validation_rows(_historical(seasons)),
                                                      config.VALIDATION_SEASONS)
    finally:
        setattr(config, attr, current)
    k = config.NOISE_STANDARD_ERRORS
    label = dict(options)[current]
    lines = [f"## {heading}", "",
             f"Same folds for each setting. Changes are in average error against the current setting "
             f"({label}) on the same games; noise is {k} standard errors, measured month to month. "
             "A change within the noise is a tie, and a tie keeps the simpler setting. October is the "
             "first month of each season, when ratings lean most on last season's games.", "",
             "| Setting | Games | Margin MAE | Total MAE | Winner % | Total bias | October total bias "
             "| Margin change | Margin noise | Total change | Total noise | Verdict |",
             "|" + " --- |" * 12]
    for value, name_ in options:
        m = evaluate.metrics(preds[value])
        october = evaluate.metrics(preds[value][preds[value]["FOLD"].str.endswith("-10")])
        if value == current:
            test = ["-", "-", "-", "-", "current"]
        else:
            t = evaluate.noise_test(preds[current], preds[value])
            # Three decimals: changes this small round to the same two-decimal number as the noise.
            test = [f"{t['margin'][0]:+.3f}", f"{k * t['margin'][1]:.3f}",
                    f"{t['total'][0]:+.3f}", f"{k * t['total'][1]:.3f}", t["verdict"]]
        lines.append(evaluate._row(name_, m, f"{october['total_bias']:+.2f}", *test))
    return lines


def cmd_evaluate(args):
    seasons = sorted(set(config.TRAIN_SEASONS) | set(_evaluation_seasons()))
    lines_ = _vegas_lines()
    abbr = vegas.team_abbreviations()
    if args.final:
        print(f"Scoring the locked test season {config.TEST_SEASON}. Every run is logged in results/test_log.md;")
        print("run this once per finished change, never to choose between settings.")
        entry = evaluate.final(_historical(seasons), lines_, abbr)
        print("\n".join(entry))
        return
    names = list(COMPARISONS) if "all" in (args.compare or []) else list(args.compare or [])
    if args.playoffs == "all" and "playoffs" not in names:
        names.append("playoffs")
    extra = []
    for name in names:
        extra += ([""] if extra else []) + _comparison(name, seasons)
    print("\n".join(evaluate.run(_historical(seasons), lines_, abbr, extra)))
    print("Saved results/evaluation.md")
    cmd_report(args)


# The hand-picked settings `tune` checks (decision 10, fixed before any results), in this order.
# Each option sets one or more config values; the value in use must be one of them.
TUNING = [
    ("Rating half-life", "games", [(str(v), {"RATING_HALF_LIFE_GAMES": v}) for v in (6, 9, 12, 18, 24)]),
    ("Last season's games", "weight", [(str(v), {"PRIOR_SEASON_GAME_WEIGHT": v}) for v in (0.2, 0.4, 0.6, 0.8)]),
    ("Rating shrinkage", "games of league average", [(str(v), {"SHRINK_GAMES": v}) for v in (0, 2, 4, 8)]),
    ("Minutes half-life", "games", [(str(v), {"MINUTES_HALF_LIFE_GAMES": v}) for v in (2, 3, 5, 8, 12)]),
    ("Rotation player cutoff", "minutes", [(str(v), {"ROTATION_MINUTES": v}) for v in (12, 16, 20, 24)]),
    # Offense and defense keep the 3:1 split of the current -1.5 / -0.5; the label is the total.
    ("Replacement level", "BPM, offense + defense",
     [(f"{o + d:g}", {"REPLACEMENT_OBPM": o, "REPLACEMENT_DBPM": d}) for o, d in ((-0.75, -0.25), (-1.5, -0.5),
                                                                                   (-2.25, -0.75))]),
    ("BPM shrinkage", "minutes", [(str(v), {"BPM_PRIOR_MINUTES": v}) for v in (250, 500, 1000, 2000)]),
    ("Rest days cap", "days", [(str(v), {"MAX_REST_DAYS": v}) for v in (2, 3, 5)]),
]


def _current_option(options: list) -> str:
    return next(label for label, values in options
                if all(getattr(config, attr) == v for attr, v in values.items()))


def cmd_tune(args):
    """Decision 10: check each setting in TUNING, in order, against the value in use, with the
    noise test. A value replaces it only if it is "better" (several: the lowest margin error);
    the settings after it are then checked with the new value in place. config.py is not
    changed; results/tuning.md shows what the test picked."""
    seasons = [s for s in config.TRAIN_SEASONS if s < config.TEST_SEASON]
    original = {attr: getattr(config, attr) for _, _, options in TUNING for _, values in options for attr in values}
    folds = {}

    def cross_validate():
        key = json.dumps(features.settings(), sort_keys=True)
        if key not in folds:
            folds[key] = evaluate.cross_validate(evaluate.validation_rows(_historical(seasons)),
                                                 config.VALIDATION_SEASONS)[0]
        return folds[key]

    k = config.NOISE_STANDARD_ERRORS
    lines = ["# Tuning the hand-picked settings", "",
             "Decision 10 in docs/decisions.md. Each setting is checked in this order against the value in use, "
             f"on the same validation folds. Changes are in average error on the same games; noise is {k} standard "
             "errors, measured month to month. A value replaces the one in use only if it beats the noise "
             f"(margin error first, then total error) and doesn't move the average total more than "
             f"{config.BIAS_GUARD_POINTS} points further from the actual. Otherwise the value in use stays. "
             "The settings after it are checked with the winner in place.", ""]
    summary = []
    try:
        for heading, unit, options in TUNING:
            current = _current_option(options)
            in_use = dict(options)[current]
            print(f"Tuning {heading.lower()} (in use: {current})...")
            preds = {}
            for label, values in options:
                for attr, v in values.items():
                    setattr(config, attr, v)
                preds[label] = cross_validate()
                for attr, v in in_use.items():
                    setattr(config, attr, v)
            lines += [f"## {heading} ({unit})", "",
                      "| Value | Games | Margin MAE | Total MAE | Winner % | Total bias | Margin change "
                      "| Margin noise | Total change | Total noise | Verdict |",
                      "|" + " --- |" * 11]
            scores, winners = {}, []
            for label, _ in options:
                m = scores[label] = evaluate.metrics(preds[label])
                if label == current:
                    test = ["-", "-", "-", "-", "current"]
                else:
                    t = evaluate.noise_test(preds[current], preds[label])
                    test = [f"{t['margin'][0]:+.3f}", f"{k * t['margin'][1]:.3f}",
                            f"{t['total'][0]:+.3f}", f"{k * t['total'][1]:.3f}", t["verdict"]]
                    if t["verdict"] == "better":
                        winners.append(label)
                lines.append(evaluate._row(label, m, *test))
            picked = min(winners, key=lambda w: [scores[w][c] for c in config.SELECTION_METRICS]) if winners else current
            for attr, v in dict(options)[picked].items():
                setattr(config, attr, v)
            edge = picked != current and picked in (options[0][0], options[-1][0])
            lines += ["", f"**{'Changed to ' + picked if picked != current else 'Kept ' + current}.**"
                      + (" The winner is at the end of the values tried." if edge else ""), ""]
            summary.append(f"| {heading} | {current} | {picked} |")
    finally:
        for attr, v in original.items():
            setattr(config, attr, v)
    lines += ["## Summary", "", "| Setting | In use before | Test picked |", "| --- | --- | --- |", *summary]
    out = config.ROOT / "results" / "tuning.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(summary) - 2:]))
    print(f"Saved {out}")
    cmd_report(args)


def cmd_compare(args):
    """Decision 11: other kinds of model against ridge regression, on the same inputs and validation
    folds, with the noise test. Nothing in the model changes; results/comparison.md shows what the
    test picked."""
    seasons = [s for s in config.TRAIN_SEASONS if s < config.TEST_SEASON]
    feats = evaluate.validation_rows(_historical(seasons))
    print(f"Comparing {len(compare.CANDIDATES)} kinds of model and a blend, month by month (5 to 15 minutes)...")
    lines, _ = compare.run(feats, config.VALIDATION_SEASONS)
    k = config.NOISE_STANDARD_ERRORS
    head = ["# Comparing kinds of model", "",
            "Decision 11 in docs/decisions.md. Every model predicts each team's points from the same inputs as "
            "ridge regression, on the same monthly folds of the validation seasons. In each fold it picks its own "
            f"setting on the most recent {compare.INNER_SHARE:.0%} of the training games, then refits on all of "
            "them. Changes are in average error against ridge on the same games; noise is "
            f"{k} standard errors, measured month to month. A model replaces ridge only if it beats the noise "
            "(margin error first, then total error) and doesn't move the average total more than "
            f"{config.BIAS_GUARD_POINTS} points further from the actual. If several do, the lowest margin error "
            "wins.", ""]
    out = config.ROOT / "results" / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + lines) + "\n")
    print("\n".join(lines[:len(compare.CANDIDATES) + 7]))  # the table and the result
    print(f"Saved {out}")
    cmd_report(args)


def cmd_report(args):
    for path in report.run():
        print(f"Saved {path.relative_to(config.ROOT)}")


def cmd_explain(args):
    """Chart of how one game's prediction is built, from the details `predict` saved."""
    day = args.date or date_cls.today().isoformat()
    path = config.PREDICTIONS_DIR / f"{day}_details.csv"
    if not path.exists():
        raise SystemExit(f"No {path.name}. Run `python -m nbamodel predict --date {day}` first.")
    details = pd.read_csv(path)
    team = args.game.upper()
    game_ids = details.loc[details["team"] == team, "game_id"]
    if game_ids.empty:
        raise SystemExit(f"{team} has no game on {day}. Teams playing: {', '.join(sorted(details['team']))}")
    game = details[details["game_id"] == game_ids.iloc[0]]
    away, home = game.sort_values("IS_HOME")["team"]
    out = config.ROOT / args.out if args.out else config.PREDICTIONS_DIR / f"{day}_{away}-at-{home}.png"
    print(f"Saved {report.prediction_breakdown(game, out)}")


def _warn_unmatched(statuses: pd.DataFrame, players: pd.DataFrame, teams: set) -> None:
    """Point out listed players whose names don't match anyone in the box scores."""
    known = set(players["PLAYER_NAME"].map(normalize_name))
    listed = statuses[statuses["TEAM"].isin(teams) & (statuses["STATUS"] != injuries.NOT_SUBMITTED)]
    unmatched = [p for p in listed["PLAYER"] if normalize_name(p) not in known]
    if unmatched:
        print(f"Note: {len(unmatched)} listed players not found in box scores (rookies or name mismatches): "
              + "; ".join(unmatched[:10]))


def _warn_unreported(parsed: pd.DataFrame, on_day: pd.DataFrame, playing: set, day: pd.Timestamp) -> None:
    """Point out teams playing today that the injury report doesn't cover yet. Their players are
    all assumed available, including anyone out long term."""
    if on_day.empty and not parsed.empty:
        dates = ", ".join(sorted(str(d.date()) for d in pd.to_datetime(parsed["GAME_DATE"]).dropna().unique()))
        print(f"Warning: the latest injury report covers {dates}, not {day.date()}. "
              "Every player is assumed available.")
        return
    pending = sorted(set(on_day.loc[on_day["STATUS"] == injuries.NOT_SUBMITTED, "TEAM"]) & playing)
    if pending:
        print(f"Warning: not yet submitted to the injury report: {', '.join(pending)}. "
              "Their players are assumed available; run again later.")
    absent = sorted(playing - set(on_day["TEAM"]))
    if absent and not parsed.empty:
        print(f"Note: nobody listed on the injury report for {', '.join(absent)}.")


def cmd_injuries(args):
    """Show what the injury report reader finds, to check it against the PDF."""
    if args.file:
        source, pdf = args.file, open(args.file, "rb").read()
    else:
        found = injuries.latest_report()
        if not found:
            print("No injury report found in the last 24 hours at", injuries.REPORT_URL.format(stamp="<date_time>"))
            print("Download today's PDF from https://official.nba.com and run: python -m nbamodel injuries --file <path>")
            return
        source, pdf = found
    print(f"Report: {source}")
    text = injuries.report_text(pdf)
    out = config.DATA_DIR / "injury_report_text.txt"
    config.DATA_DIR.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    parsed = injuries.parse_report_text(text, injuries._team_names())
    pending = parsed[parsed["STATUS"] == injuries.NOT_SUBMITTED]
    parsed = parsed[parsed["STATUS"] != injuries.NOT_SUBMITTED]
    print(f"{len(parsed)} players read from the report")
    if parsed.empty:
        print("First lines of the PDF text (full text saved to", out, "):")
        print("\n".join(text.splitlines()[:40]))
        return
    for game_date, rows in parsed.groupby("GAME_DATE", dropna=False):
        label = "no date found" if pd.isna(game_date) else f"games on {game_date.date()}"
        print(f"\n{label}:")
        print(rows.groupby(["TEAM", "STATUS"]).size().unstack(fill_value=0).to_string())
    if not pending.empty:
        print("\nNot yet submitted:", ", ".join(sorted(set(pending["TEAM"]))))
    print()
    print(parsed.head(25).to_string(index=False))


def predict_date(day: pd.Timestamp, use_injury_report: bool = True, refresh: bool = False,
                 details: list | None = None) -> pd.DataFrame:
    """One row per game. If a list is passed as details, one row per team is appended to it
    with the inputs and the points each step of the prediction adds."""
    season = data.season_for_date(day)
    prev = data.previous_season(season)
    games, players = _load([prev, season], refresh_season=season if refresh else None)
    sched = data.schedule(season, refresh=refresh)
    today = sched[sched["GAME_DATE"] == day]
    if today.empty:
        print(f"No games on {day.date()}.")
        return pd.DataFrame()

    statuses = []
    source = "none"
    playing = set(today["HOME_TEAM"]) | set(today["AWAY_TEAM"])
    if use_injury_report:
        found = injuries.latest_report()
        if found:
            source, pdf = found
            parsed = injuries.parse_report_pdf(pdf)
            if parsed.empty:
                print("Warning: the injury report had no readable players. Run `python -m nbamodel injuries`.")
            # A report covers more than one day of games; keep only this day's rows.
            on_day = injuries.for_date(parsed, day)
            print(f"Injury report: {source} ({(on_day['STATUS'] != injuries.NOT_SUBMITTED).sum()} "
                  f"players listed for {day.date()})")
            _warn_unreported(parsed, on_day, playing, day)
            statuses.append(on_day)
        else:
            print("Warning: no injury report found in the last 24 hours. Run `python -m nbamodel injuries`.")
    statuses.append(injuries.manual_overrides())
    all_statuses = pd.concat(statuses, ignore_index=True)
    probs = injuries.play_probabilities(all_statuses)
    _warn_unmatched(all_statuses, players, playing)

    impact = lineup.player_impact(data.bpm(prev), data.bpm(season, refresh=refresh))
    roster = data.rosters(season, refresh=refresh).groupby("TEAM_ID")["PLAYER_ID"].apply(set)
    league = ratings.league_average(games, day, season)
    fitted = model.load()
    rows = []
    for g in today.itertuples():
        pair = features.matchup_rows(games, players, impact, day, season, g.HOME_TEAM_ID, g.AWAY_TEAM_ID,
                                     probs.get(g.HOME_TEAM, {}), probs.get(g.AWAY_TEAM, {}), league,
                                     roster.get(g.HOME_TEAM_ID) or None, roster.get(g.AWAY_TEAM_ID) or None)
        X = pd.DataFrame(pair)
        home_pts, away_pts = model.predict(fitted, X)
        if details is not None:
            for row, team, opp, pts in ((pair[0], g.HOME_TEAM, g.AWAY_TEAM, home_pts),
                                        (pair[1], g.AWAY_TEAM, g.HOME_TEAM, away_pts)):
                details.append({"date": day.date().isoformat(), "game_id": _game_id(g.GAME_ID), "team": team,
                                "opponent": opp, **{f: row[f] for f in fitted["coefficients"]},
                                **model.contributions(fitted, row), "predicted_pts": pts,
                                "missing": row["MISSING"]})
        rows.append({
            "date": day.date().isoformat(),
            "game_id": _game_id(g.GAME_ID),
            "away": g.AWAY_TEAM,
            "home": g.HOME_TEAM,
            "away_pts": round(away_pts, 1),
            "home_pts": round(home_pts, 1),
            "home_margin": round(home_pts - away_pts, 1),
            "total": round(home_pts + away_pts, 1),
            "home_lineup_pts": round(pair[0]["LINEUP_OFF_PTS"] - pair[1]["LINEUP_DEF_PTS"], 1),
            "away_lineup_pts": round(pair[1]["LINEUP_OFF_PTS"] - pair[0]["LINEUP_DEF_PTS"], 1),
            "home_missing": pair[0]["MISSING"],
            "away_missing": pair[1]["MISSING"],
            "injury_report": source,
            "tip_off": g.GAME_TIME_UTC,
        })
    return pd.DataFrame(rows)


def _game_id(game_id) -> str:
    """The official 10-digit id, e.g. '0022600119'. Read back from a CSV it loses its leading zeros."""
    return str(game_id).zfill(10)


def _read_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"game_id": str})
    df["game_id"] = df["game_id"].map(_game_id)
    return df


def _started(preds: pd.DataFrame, now: pd.Timestamp) -> list:
    """Ids of the games whose scheduled tip-off is at or before now."""
    tip_off = pd.to_datetime(preds["tip_off"], utc=True, errors="coerce")
    return preds.loc[tip_off <= now, "game_id"].tolist()


def _matchups(preds: pd.DataFrame, game_ids: list) -> str:
    rows = preds[preds["game_id"].isin(game_ids)]
    return ", ".join(f"{r.away} @ {r.home}" for r in rows.itertuples())


def keep_started(old: pd.DataFrame, new: pd.DataFrame, now: pd.Timestamp,
                 frozen: list | None = None) -> tuple[pd.DataFrame, list]:
    """Put back the earlier rows for games that had tipped off by now, so a later run never
    rewrites a prediction once its game is under way. Also works on the per-team details
    rows when given the frozen game ids. Returns the rows and the frozen ids."""
    if frozen is None:
        frozen = [g for g in _started(new, now) if g in set(old["game_id"])]
    if not frozen:
        return new, []
    out = pd.concat([new[~new["game_id"].isin(frozen)], old[old["game_id"].isin(frozen)]], ignore_index=True)
    # Back in the new run's order. Files saved before tip_off was added get it from the new run.
    order = {g: i for i, g in enumerate(dict.fromkeys(new["game_id"]))}
    out = out.sort_values("game_id", key=lambda ids: ids.map(order), kind="stable").reset_index(drop=True)
    out = out.reindex(columns=new.columns)
    if "tip_off" in new:
        out["tip_off"] = out["game_id"].map(new.set_index("game_id")["tip_off"])
    return out, frozen


def _log_changes(old: pd.DataFrame, new: pd.DataFrame, day: str) -> None:
    """Append a line for every game whose prediction moved by a point or more."""
    merged = new.merge(old, on="game_id", suffixes=("", "_old"))
    lines = []
    for r in merged.itertuples():
        if abs(r.home_margin - r.home_margin_old) >= 1 or abs(r.total - r.total_old) >= 1:
            lines.append(
                f"- {pd.Timestamp.now():%H:%M} {r.away} @ {r.home}: margin {r.home_margin_old:+} -> "
                f"{r.home_margin:+}, total {r.total_old} -> {r.total}. Out/doubtful now: "
                f"{r.home}: {r.home_missing or 'none'}; {r.away}: {r.away_missing or 'none'}"
            )
    if lines:
        with open(config.PREDICTIONS_DIR / f"{day}_changes.md", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


def cmd_predict(args, refresh: bool = False, now: pd.Timestamp | None = None):
    day = pd.Timestamp(args.date or date_cls.today().isoformat())
    details = []
    preds = predict_date(day, use_injury_report=not args.no_injury_report, refresh=refresh, details=details)
    if preds.empty:
        return
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    preds["predicted_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    details = pd.DataFrame(details)
    config.PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = config.PREDICTIONS_DIR / f"{day.date()}.csv"
    details_path = config.PREDICTIONS_DIR / f"{day.date()}_details.csv"
    started = _started(preds, now)
    if path.exists():
        old = _read_predictions(path)
        preds, frozen = keep_started(old, preds, now)
        if frozen and details_path.exists():
            details = keep_started(_read_predictions(details_path), details, now, frozen)[0]
        if frozen:
            print("Kept the earlier prediction for games already under way: " + _matchups(preds, frozen))
        started = [g for g in started if g not in frozen]
        _log_changes(old, preds, str(day.date()))
    if started:
        print("Note: these games had already tipped off before their first prediction: " + _matchups(preds, started))
    try:
        preds.to_csv(path, index=False)
        details.round(3).to_csv(details_path, index=False)
    except PermissionError:
        print(f"Could not save {path}: it is probably open in Excel. Close it and run again.")
        return
    print(preds[["away", "home", "away_pts", "home_pts", "home_margin", "total"]].to_string(index=False))
    print(f"Saved {path}. For a game's breakdown: python -m nbamodel explain --date {day.date()} --game <team>")
    if getattr(args, "push", False):
        push_predictions(str(day.date()))


def push_predictions(day: str) -> None:
    """Commit the predictions folder to its own branch on config.PREDICTIONS_REMOTE and push it.

    Uses a separate git index, so the checked-out branch and anything staged for it are
    untouched, and the predictions never mix with the code's history."""
    remote, branch = config.PREDICTIONS_REMOTE, config.PREDICTIONS_BRANCH

    def git(*cmd, env=None, check=True):
        return subprocess.run(["git", *cmd], cwd=config.ROOT, env=env, check=check,
                              capture_output=True, text=True).stdout.strip()

    # Exit code 2 means the remote is reachable but the branch doesn't exist yet (first push).
    listed = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", remote, branch],
                            cwd=config.ROOT, capture_output=True, text=True)
    if listed.returncode not in (0, 2):
        print(f"Could not reach {remote} to push the predictions: {listed.stderr.strip()}")
        return
    parent = None
    if listed.returncode == 0:
        git("fetch", "-q", remote, branch)
        parent = git("rev-parse", "FETCH_HEAD")
    index = config.ROOT / git("rev-parse", "--git-path", "predictions-index")
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    try:
        git("read-tree", "--empty", env=env)
        git("add", "-f", "--", config.PREDICTIONS_DIR.name, env=env)
        tree = git("write-tree", env=env)
    finally:
        index.unlink(missing_ok=True)
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree:
        print(f"Predictions already on {remote}/{branch}.")
        return
    commit = git("commit-tree", tree, "-m", f"Predictions for {day}", *(["-p", parent] if parent else []))
    pushed = subprocess.run(["git", "push", "-q", remote, f"{commit}:refs/heads/{branch}"],
                            cwd=config.ROOT, capture_output=True, text=True)
    if pushed.returncode:
        print(f"Could not push the predictions to {remote}/{branch}: {pushed.stderr.strip()}")
        return
    print(f"Pushed predictions to {remote}/{branch}.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nbamodel", description="Predict NBA final scores.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Download all seasons of data")
    sub.add_parser("validate", help="Check the downloaded data and list games to spot-check")
    playoff_help = "How playoff games feed ratings (default from config.py)"
    p_train = sub.add_parser("train", help="Fit the regression and save data/model.json")
    p_train.add_argument("--playoffs", choices=["include", "exclude", "relative"], help=playoff_help)
    p_back = sub.add_parser("backtest", help="Season-by-season accuracy, written to results/backtest.md")
    p_back.add_argument("--playoffs", choices=["include", "exclude", "relative", "all"],
                        help=playoff_help + "; 'all' compares the three in results/playoff_comparison.md")
    p_eval = sub.add_parser("evaluate", help="Cross-validation, tuning curves and Vegas benchmark, "
                            "written to results/evaluation.md")
    p_eval.add_argument("--final", action="store_true",
                        help="Score the locked test season once and log it in results/test_log.md")
    p_eval.add_argument("--playoffs", choices=["include", "exclude", "relative", "all"],
                        help=playoff_help + "; 'all' adds a comparison of the three")
    p_eval.add_argument("--compare", action="append", choices=[*COMPARISONS, "all"],
                        help="Add a side-by-side table of a setting's values, with the noise test; "
                             "'all' does every one (repeatable)")
    inj = sub.add_parser("injuries", help="Show what the injury report reader finds")
    inj.add_argument("--file", help="Read a downloaded injury report PDF instead of fetching the latest")
    sub.add_parser("tune", help="Test the hand-picked settings one by one (decision 10), "
                   "written to results/tuning.md")
    sub.add_parser("compare", help="Other kinds of model against ridge regression (decision 11), "
                   "written to results/comparison.md")
    sub.add_parser("report", help="Redraw the charts in docs/figures from results/ and data/model.json")
    p_explain = sub.add_parser("explain", help="Chart of how one game's prediction is built")
    p_explain.add_argument("--game", required=True, help="Either team's abbreviation, e.g. BOS")
    p_explain.add_argument("--date", help="YYYY-MM-DD, default today")
    p_explain.add_argument("--out", help="Where to save the chart, default predictions/<date>_<away>-at-<home>.png")
    for name in ("predict", "daily"):
        p = sub.add_parser(name, help="Predict a day's games" if name == "predict"
                           else "Refresh this season's data, then predict today")
        p.add_argument("--date", help="YYYY-MM-DD, default today")
        p.add_argument("--no-injury-report", action="store_true", help="Use only manual_status.csv")
        p.add_argument("--push", action="store_true",
                       help="Push the predictions folder to its own branch (config.PREDICTIONS_REMOTE)")
        p.add_argument("--playoffs", choices=["include", "exclude", "relative"], help=playoff_help)
    args = parser.parse_args(argv)
    args.playoffs_default = config.PLAYOFF_GAMES
    if getattr(args, "playoffs", None) and args.playoffs != "all":
        config.PLAYOFF_GAMES = args.playoffs
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "injuries":
        cmd_injuries(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "daily":
        cmd_predict(args, refresh=True)
