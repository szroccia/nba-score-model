"""Decision 14: season and playoff odds, from playing out the rest of the season many times.

Every remaining game gets the model's predicted margin from each team's ratings and roster as they
stand on the day of the simulation, with home court and rest from the schedule. The predictions stay
frozen: nothing is re-rated as simulated games are played. The season is then played out many
times: each game's result is its prediction plus a miss drawn at random, as wide as frozen
predictions missed in earlier seasons. Standings, the play-in and four best-of-seven rounds follow
the NBA's format, with a simpler tiebreaker (head-to-head among the tied teams, then a coin flip).
A team's odds of something are the share of simulated seasons in which it happens.
"""

import numpy as np
import pandas as pd

from . import config, evaluate, features, model, probability, ratings

CONFERENCES = {
    "East": "ATL BOS BKN CHA CHI CLE DET IND MIA MIL NYK ORL PHI TOR WAS".split(),
    "West": "DAL DEN GSW HOU LAC LAL MEM MIN NOP OKC PHX POR SAC SAS UTA".split(),
}
# Fixed before any results (decision 14), simplest first: config name -> (name, what it is).
METHODS = {
    "independent": ("Independent games", "every simulated game misses by fresh luck, as wide as frozen predictions "
                    "have missed"),
    "lasting": ("Lasting team errors", "the same total miss, split into a part that stays with each team for the "
                "whole simulated season and a part that is fresh every game"),
}
# Each validation season is simulated from the first of these months, with only what was known that day.
SNAPSHOT_MONTHS = [11, 12, 1, 2, 3, 4]
PLAYOFF_REST = 1        # days of rest both teams are assumed to have in play-in and playoff games
RANGE = 0.8             # win ranges hold the middle 80% of simulated seasons
# Odds reported for each team: column -> what it means.
ODDS = {
    "TOP6": "finishes in the top six of its conference",
    "PLAY_IN": "finishes 7th to 10th (the play-in)",
    "PLAYOFFS": "reaches the playoffs, directly or through the play-in",
    "ROUND2": "reaches the second round",
    "CONF_FINALS": "reaches the conference finals",
    "FINALS": "reaches the Finals",
    "TITLE": "wins the title",
}


def snapshot_dates(season: str) -> list[pd.Timestamp]:
    start = int(season[:4])
    return [pd.Timestamp(year=start if m >= 10 else start + 1, month=m, day=1) for m in SNAPSHOT_MONTHS]


def conferences(abbr: dict) -> dict:
    """{team id: conference}, from {team id: abbreviation}."""
    by_abbr = {a: c for c, members in CONFERENCES.items() for a in members}
    unknown = sorted(set(abbr.values()) - set(by_abbr))
    if unknown:
        raise ValueError(f"No conference listed for {', '.join(unknown)} (simulation.CONFERENCES)")
    return {t: by_abbr[a] for t, a in abbr.items()}


def regular_games(games: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per finished regular-season game of a season: home and away team and the home margin."""
    g = games[(games["SEASON"] == season) & (games["IS_HOME"] == 1) & ~games["PLAYOFF"].astype(bool)]
    return pd.DataFrame({"GAME_ID": g["GAME_ID"].to_numpy(), "GAME_DATE": g["GAME_DATE"].to_numpy(),
                         "HOME_ID": g["TEAM_ID"].to_numpy(), "AWAY_ID": g["OPP_TEAM_ID"].to_numpy(),
                         "MARGIN": (g["PTS"] - g["OPP_PTS"]).to_numpy()})


def with_rest(sched: pd.DataFrame) -> pd.DataFrame:
    """Each team's days of rest before each game, from the schedule itself, capped as in
    ratings.rest_days (a team's first game counts as fully rested)."""
    long = pd.concat([sched[["GAME_ID", "GAME_DATE", side]].rename(columns={side: "TEAM"}).assign(SIDE=side)
                      for side in ("HOME_ID", "AWAY_ID")]).sort_values(["TEAM", "GAME_DATE"])
    days = (long["GAME_DATE"] - long.groupby("TEAM")["GAME_DATE"].shift()).dt.days - 1
    long["REST"] = days.clip(upper=config.MAX_REST_DAYS).fillna(config.MAX_REST_DAYS).astype(int)
    rest = long.pivot_table(index="GAME_ID", columns="SIDE", values="REST", aggfunc="first")
    return sched.assign(HOME_REST=sched["GAME_ID"].map(rest["HOME_ID"]).to_numpy(),
                        AWAY_REST=sched["GAME_ID"].map(rest["AWAY_ID"]).to_numpy())


def rosters_on(players: pd.DataFrame, season: str, day: pd.Timestamp) -> dict:
    """{team id: player ids}: each player counts for the last team he played for this season before the day."""
    seen = players[(players["SEASON"] == season) & (players["GAME_DATE"] < day)].sort_values("GAME_DATE")
    last = seen.groupby("PLAYER_ID").tail(1)
    return last.groupby("TEAM_ID")["PLAYER_ID"].apply(set).to_dict()


def _predict(states: dict, league: dict, fitted: dict, home, away, home_rest, away_rest) -> np.ndarray:
    rows = [r for h, a, hr, ar in zip(home, away, home_rest, away_rest)
            for r in features.pair_rows(states[h], states[a], int(hr), int(ar), league)]
    if not rows:
        return np.array([])
    pts = model.predict(fitted, pd.DataFrame(rows))
    return pts[0::2] - pts[1::2]


def snapshot(games: pd.DataFrame, players: pd.DataFrame, impact: pd.DataFrame, season: str, day: pd.Timestamp,
             fitted: dict, teams: list, remaining: pd.DataFrame, probs: dict | None = None,
             rosters: dict | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """What the model expected on the day: every remaining game's predicted home margin (PRED), and
    a table of margins for any pairing (row team at home), used for the play-in and playoffs.

    Ratings and rosters are as of the day and stay frozen. probs (by team id: chance of playing, by
    name) holds players at that chance in every game; everyone else on a roster plays. Without
    rosters, a team's roster is everyone who has played for it this season."""
    league = ratings.league_average(games, day, season)
    probs = probs or {}
    rosters = rosters if rosters is not None else rosters_on(players, season, day)
    states = {t: features.team_state(games, players, impact, day, season, t, probs.get(t, {}), league,
                                     rosters.get(t) or None) for t in teams}
    remaining = remaining.assign(PRED=_predict(states, league, fitted, remaining["HOME_ID"], remaining["AWAY_ID"],
                                               remaining["HOME_REST"], remaining["AWAY_REST"]))
    n = len(teams)
    home, away = np.divmod(np.arange(n * n), n)
    keep = home != away
    rest = np.full(keep.sum(), PLAYOFF_REST)
    pairs = np.zeros((n, n))
    pairs[home[keep], away[keep]] = _predict(states, league, fitted, np.asarray(teams)[home[keep]],
                                             np.asarray(teams)[away[keep]], rest, rest)
    return remaining, pairs


def fit_spread(frozen: pd.DataFrame) -> dict:
    """How far predictions made from a frozen day missed, split into a part that stays with a team
    and a part that is fresh each game (a one-way random-effects split, by method of moments).

    Each game's miss is the home team's lasting part minus the away team's, plus the game's own
    luck, so the total spread is 2 x lasting + nightly (as variances). A team's average miss over
    its remaining games varies more from team to team than its game-to-game spread alone would
    make it; the excess is the lasting part."""
    miss = (frozen["MARGIN"] - frozen["PRED"]).to_numpy(float)
    total = float(np.mean(miss ** 2))
    key = frozen[["SEASON", "SNAPSHOT"]]
    long = pd.concat([key.assign(TEAM=frozen["HOME_ID"].to_numpy(), MISS=miss),
                      key.assign(TEAM=frozen["AWAY_ID"].to_numpy(), MISS=-miss)])
    teams = long.groupby(["SEASON", "SNAPSHOT", "TEAM"])["MISS"].agg(["mean", "var", "size"])
    teams = teams[teams["size"] >= 2]
    lasting = max(0.0, float(teams["mean"].var() - (teams["var"] / teams["size"]).mean()))
    lasting = min(lasting, total / 2)
    return {"sd_total": total ** 0.5, "sd_team": lasting ** 0.5, "sd_game": (total - 2 * lasting) ** 0.5,
            "games": int(len(frozen))}


def spreads_for(method: str, spread: dict) -> tuple[float, float]:
    """(nightly, lasting) spread a method simulates with: independent games put all of the miss
    in each night; lasting team errors split it."""
    if method == "independent":
        return spread["sd_total"], 0.0
    if method == "lasting":
        return spread["sd_game"], spread["sd_team"]
    raise ValueError(f"Unknown season simulation method {method!r}")


def simulate(teams: list, conference: dict, played: pd.DataFrame, remaining: pd.DataFrame, pairs: np.ndarray,
             sd_game: float, sd_team: float, n_sims: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Play out the rest of the season n_sims times. played has HOME_ID, AWAY_ID and MARGIN for
    games already played; remaining has HOME_ID, AWAY_ID and PRED; pairs is the home margin for
    any pairing; conference is {team id: "East" or "West"}. Returns one row per team: games played
    and in all, wins so far, average and middle-80% wins, and the share of seasons for each odds column."""
    rng = np.random.default_rng(seed)
    n_sims = n_sims or config.SEASON_SIMULATIONS
    n = len(teams)
    index = {t: i for i, t in enumerate(teams)}
    sims = np.arange(n_sims)
    lasting = rng.normal(0, sd_team, (n_sims, n)).astype(np.float32) if sd_team > 0 else None

    def luck(shape):
        return rng.standard_normal(shape, dtype=np.float32) * np.float32(sd_game)

    # The regular season: wins so far plus wins in the simulated games (one column per game).
    ph, pa = played["HOME_ID"].map(index).to_numpy(int), played["AWAY_ID"].map(index).to_numpy(int)
    won = played["MARGIN"].to_numpy() > 0
    wins_now = np.bincount(ph[won], minlength=n) + np.bincount(pa[~won], minlength=n)
    h, a = remaining["HOME_ID"].map(index).to_numpy(int), remaining["AWAY_ID"].map(index).to_numpy(int)
    margin = remaining["PRED"].to_numpy(np.float32)[None, :] + luck((n_sims, len(h)))
    if lasting is not None:
        margin += lasting[:, h] - lasting[:, a]
    result = margin > 0
    del margin
    onehot = np.eye(n, dtype=np.float32)
    r = result.astype(np.float32)
    wins = wins_now + np.rint(r @ onehot[h] + (1 - r) @ onehot[a]).astype(int)
    del r
    played_n = np.bincount(ph, minlength=n) + np.bincount(pa, minlength=n)
    games_n = played_n + np.bincount(h, minlength=n) + np.bincount(a, minlength=n)

    # Head-to-head wins between every two teams (row beat column), for the tiebreaker.
    head = np.zeros((n, n), np.int16)
    np.add.at(head, (ph[won], pa[won]), 1)
    np.add.at(head, (pa[~won], ph[~won]), 1)
    head = np.repeat(head[None], n_sims, axis=0)
    if len(h):
        pair = h * n + a
        order = np.argsort(pair, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(pair[order]) != 0])
        home_won = np.add.reduceat(result[:, order].astype(np.int16), starts, axis=1)
        meetings = np.diff(np.r_[starts, len(order)]).astype(np.int16)
        hi, ai = np.divmod(pair[order][starts], n)
        head[:, hi, ai] += home_won
        head[:, ai, hi] += meetings - home_won

    # Seeding by win percentage, then head-to-head among teams tied with it in its conference, then
    # a coin flip. The tiebreaks are scaled so they never outweigh a difference in record.
    conf = np.array([conference[t] for t in teams])
    pct = wins / np.maximum(games_n, 1)
    tied = (pct[:, :, None] == pct[:, None, :]) & (conf[:, None] == conf[None, :]) & ~np.eye(n, dtype=bool)
    h2h_won = (tied * head).sum(2)
    h2h_games = (tied * (head + head.transpose(0, 2, 1))).sum(2)
    share = np.where(h2h_games > 0, h2h_won / np.maximum(h2h_games, 1), 0.5)
    key = pct + 1e-6 * share + 1e-9 * rng.random((n_sims, n))
    del tied, head

    def game(home, away):
        """Did the home team win one game, in each simulation? home and away hold one team per simulation."""
        margin = pairs[home, away] + luck(n_sims)
        if lasting is not None:
            margin += lasting[sims, home] - lasting[sims, away]
        return margin > 0

    def series(x, y, seed_x, seed_y):
        """Best of seven, home court to the better seed (2-2-1-1-1). Playing all seven games picks
        the same winner as stopping at four wins, since each game is drawn on its own."""
        top, other = np.where(seed_x < seed_y, x, y), np.where(seed_x < seed_y, y, x)
        top_wins = 0
        for top_home in (True, True, False, False, True, False, True):
            top_wins = top_wins + (game(top, other) if top_home else ~game(other, top))
        return np.where(top_wins >= 4, top, other)

    counts = {c: np.zeros(n) for c in ODDS}

    def count(column, team):
        counts[column] += np.bincount(team, minlength=n)

    champions = []
    for name in CONFERENCES:
        members = np.flatnonzero(conf == name)
        seeded = members[np.argsort(-key[:, members], axis=1)]       # n_sims x conference, best first
        count("TOP6", seeded[:, :6].ravel())
        count("PLAY_IN", seeded[:, 6:10].ravel())
        s = [seeded[:, k] for k in range(10)]
        # Play-in: 7 hosts 8 (the winner is the 7 seed); 9 hosts 10; the 7-8 loser hosts the 9-10 winner for 8.
        won78 = game(s[6], s[7])
        seventh, loser78 = np.where(won78, s[6], s[7]), np.where(won78, s[7], s[6])
        ninth_tenth = np.where(game(s[8], s[9]), s[8], s[9])
        eighth = np.where(game(loser78, ninth_tenth), loser78, ninth_tenth)
        bracket = s[:6] + [seventh, eighth]
        for team in bracket:
            count("PLAYOFFS", team)
        # Round one: 1-8, 4-5, 3-6, 2-7; then the winners of 1-8 and 4-5 meet, and of 3-6 and 2-7.
        alive = []
        for x, y in ((0, 7), (3, 4), (2, 5), (1, 6)):
            winner = series(bracket[x], bracket[y], np.full(n_sims, x), np.full(n_sims, y))
            alive.append((winner, np.where(winner == bracket[x], x, y)))
            count("ROUND2", winner)
        final_two = []
        for (x, sx), (y, sy) in ((alive[0], alive[1]), (alive[2], alive[3])):
            winner = series(x, y, sx, sy)
            final_two.append((winner, np.where(winner == x, sx, sy)))
            count("CONF_FINALS", winner)
        (x, sx), (y, sy) = final_two
        winner = series(x, y, sx, sy)
        count("FINALS", winner)
        champions.append(winner)
    # The Finals: home court to the better regular-season record.
    east, west = champions
    count("TITLE", series(east, west, -key[sims, east], -key[sims, west]))

    low = np.percentile(wins, 100 * (1 - RANGE) / 2, axis=0, method="lower")
    high = np.percentile(wins, 100 * (1 + RANGE) / 2, axis=0, method="higher")
    return pd.DataFrame({"TEAM_ID": teams, "PLAYED": played_n, "GAMES": games_n, "WINS_NOW": wins_now,
                         "WINS": wins.mean(0), "WINS_LOW": low.astype(int), "WINS_HIGH": high.astype(int),
                         **{c: v / n_sims for c, v in counts.items()}})


def today(games: pd.DataFrame, players: pd.DataFrame, impact: pd.DataFrame, season: str, day: pd.Timestamp,
          fitted: dict, schedule: pd.DataFrame, abbr: dict, probs: dict, rosters: dict) -> pd.DataFrame | None:
    """Every team's odds from the day on. schedule has the season's regular-season games (GAME_ID,
    GAME_DATE, HOME_ID, AWAY_ID); finished games come from games, and the rest are simulated with
    the method and spread saved in fitted["season"]. None once every scheduled game is finished."""
    played = regular_games(games, season)
    sched = with_rest(schedule[["GAME_ID", "GAME_DATE", "HOME_ID", "AWAY_ID"]])
    done = set(played["GAME_ID"].astype(str).str.zfill(10))
    remaining = sched[~sched["GAME_ID"].astype(str).str.zfill(10).isin(done)]
    if remaining.empty:
        return None
    teams = sorted(abbr)
    remaining, pairs = snapshot(games, players, impact, season, day, fitted, teams, remaining, probs, rosters)
    params = fitted["season"]
    conference = conferences(abbr)
    odds = simulate(teams, conference, played, remaining, pairs, *spreads_for(params["method"], params["spread"]))
    return odds.assign(TEAM=odds["TEAM_ID"].map(abbr), CONFERENCE=odds["TEAM_ID"].map(conference))


# The backtest (decision 14's test).

def snapshots(feats: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, impact: dict,
              seasons: list[str]) -> list[dict]:
    """Each snapshot day of the given seasons, with what was known that day: the games played so
    far, and every remaining game's prediction from a regression fit on the games before the day
    and ratings and rosters as of the day (all players healthy). impact is player impact by season."""
    out = []
    for season in seasons:
        sched = with_rest(regular_games(games, season))
        teams = sorted(sched["HOME_ID"].unique().tolist())
        for day in snapshot_dates(season):
            train = feats[feats["GAME_DATE"] < day]
            later = sched["GAME_DATE"] >= day
            if train.empty or not later.any():
                continue
            remaining, pairs = snapshot(games, players, impact[season], season, day, model.fit(train), teams,
                                        sched[later])
            out.append({"season": season, "day": day, "teams": teams, "played": sched[~later],
                        "remaining": remaining, "pairs": pairs})
    return out


def frozen_record(snaps: list[dict]) -> pd.DataFrame:
    """Every frozen prediction of the snapshots, with its season and snapshot day."""
    return pd.concat([s["remaining"].assign(SEASON=s["season"], SNAPSHOT=s["day"]) for s in snaps],
                     ignore_index=True)


def fit_live(feats: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, impact: dict) -> dict:
    """What the daily season odds need: the method in use, and how far frozen predictions missed
    over the whole record (never the test season)."""
    seasons = probability.record_seasons()
    record = frozen_record(snapshots(evaluate.validation_rows(feats), games, players, impact, seasons))
    return {"method": config.SEASON_SIMULATION, "spread": fit_spread(record), "seasons": seasons}


def outcomes(games: pd.DataFrame, season: str) -> pd.DataFrame:
    """What happened: each team's regular-season wins, whether it made the playoffs (play-in games
    are not playoff games) and whether it won the title (the last playoff game)."""
    reg = regular_games(games, season)
    teams = sorted(reg["HOME_ID"].unique().tolist())
    wins = pd.concat([reg.loc[reg["MARGIN"] > 0, "HOME_ID"], reg.loc[reg["MARGIN"] < 0, "AWAY_ID"]]).value_counts()
    playoff = games[(games["SEASON"] == season) & games["PLAYOFF"].astype(bool)]
    champion = None
    if len(playoff):
        final = playoff[playoff["GAME_ID"] == playoff.sort_values(["GAME_DATE", "GAME_ID"])["GAME_ID"].iloc[-1]]
        champion = final.loc[final["PTS"] > final["OPP_PTS"], "TEAM_ID"].iloc[0]
    return pd.DataFrame({"TEAM_ID": teams, "ACTUAL_WINS": wins.reindex(teams, fill_value=0).to_numpy(),
                         "MADE_PLAYOFFS": np.isin(teams, playoff["TEAM_ID"]).astype(int),
                         "CHAMPION": (np.asarray(teams) == champion).astype(int)})


def backtest(feats: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, impact: dict,
             n_sims: int | None = None) -> list[str]:
    """Markdown for results/season.md: each validation season simulated with each method from the
    first of every month in SNAPSHOT_MONTHS, its spread fit only on the seasons before it."""
    latest = games.sort_values("GAME_DATE").drop_duplicates("TEAM_ID", keep="last")
    abbr = dict(zip(latest["TEAM_ID"], latest["TEAM_ABBREVIATION"]))
    conference = conferences(abbr)
    snaps = snapshots(evaluate.validation_rows(feats), games, players, impact, probability.record_seasons())
    record = frozen_record(snaps)
    spreads, rows = {}, []
    for k, s in enumerate(snaps):
        season = s["season"]
        if season not in config.VALIDATION_SEASONS:
            continue
        if season not in spreads:
            earlier = record[record["SEASON"] < season]
            if earlier.empty:
                raise ValueError(f"No record season before {season} to fit the spread on")
            spreads[season] = fit_spread(earlier) | {"seasons": sorted(earlier["SEASON"].unique())}
        for method in METHODS:  # the same seed for both, so they differ by method, not luck
            sim = simulate(s["teams"], conference, s["played"], s["remaining"], s["pairs"],
                           *spreads_for(method, spreads[season]), n_sims=n_sims, seed=k)
            rows.append(sim.assign(SEASON=season, SNAPSHOT=s["day"], METHOD=method))
    actual = pd.concat([outcomes(games, season).assign(SEASON=season) for season in spreads])
    odds = pd.concat(rows, ignore_index=True).merge(actual, on=["SEASON", "TEAM_ID"])
    return report(odds, spreads, abbr, n_sims or config.SEASON_SIMULATIONS)


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def _first_of(month: int) -> str:
    return f"{pd.Timestamp(2000, month, 1):%B} 1"


def _pct(v) -> str:
    return f"{v:.1%}" if pd.notna(v) else "-"


def report(odds: pd.DataFrame, spreads: dict, abbr: dict, n_sims: int) -> list[str]:
    """The tables of results/season.md from every simulated forecast and what happened."""
    k = config.NOISE_STANDARD_ERRORS
    floor = config.PROBABILITY_FLOOR
    current = config.SEASON_SIMULATION
    order = [current] + [m for m in METHODS if m != current]
    by = {m: odds[odds["METHOD"] == m].sort_values(["SEASON", "SNAPSHOT", "TEAM_ID"]).reset_index(drop=True)
          for m in order}
    base = by[current]
    made = base["MADE_PLAYOFFS"]
    team_seasons = base["SEASON"] + " " + base["TEAM_ID"].astype(str)
    in_use = METHODS[current][0]
    months = ", ".join(f"{pd.Timestamp(2000, m, 1):%B}" for m in SNAPSHOT_MONTHS[:-1])
    months += f" and {pd.Timestamp(2000, SNAPSHOT_MONTHS[-1], 1):%B}"
    seasons = sorted(spreads)
    intro = ["# Season and playoff odds", "",
             f"Decision 14 in docs/decisions.md. Each validation season ({' and '.join(seasons)}) is played out "
             f"{n_sims:,} times from the first of {months}, with only what was known that day: the games played so "
             "far, team ratings as of the day, each team's roster as the players who had played for it that season "
             "(all assumed healthy), and a regression fit on earlier games. Every remaining game's prediction stays "
             "frozen for the rest of the simulated season. How widely simulated games land around their predictions "
             "comes from how far frozen predictions missed in earlier seasons only, so no season helps score itself.",
             "",
             "Log loss scores the odds of making the playoffs (directly or through the play-in): minus the log of the "
             f"chance given to what happened, averaged over teams and snapshot days, with every chance kept between "
             f"{floor:.0%} and {1 - floor:.0%}. Lower is better. Changes are against the method in use on the same "
             f"forecasts; noise is {k} standard errors, with each team-season (its {len(SNAPSHOT_MONTHS)} snapshot "
             "days) as one block. A method replaces the one in use only if it beats the noise. Brier score, win "
             "totals, calibration and the champion's odds are reported but don't decide.", ""]

    # Playoff odds: the test.
    probs = {m: np.clip(by[m]["PLAYOFFS"].to_numpy(float), floor, 1 - floor) for m in order}
    rows, verdicts = probability.noise_table("", probs, made, team_seasons, current, "current", METHODS)
    losses = {m: probability.log_loss(p, made).mean() for m, p in probs.items()}
    picked = probability.best(verdicts, losses, current)
    result = f"{in_use} stay" if picked == current else f"{METHODS[picked][0]} beat {in_use.lower()}"
    rate = 16 / 30
    out = [*intro, "## Playoff odds", "",
           "| Method | Forecasts | Log loss | Brier score | Log loss change | Noise | Verdict |", "|" + " --- |" * 7,
           *rows, "",
           f"For scale: giving every team {rate:.0%} (16 of 30 teams make the playoffs) scores a log loss of "
           f"{probability.log_loss(np.full(len(made), rate), made).mean():.4f}.", "", f"**{result}.**", "",
           "### Simulation methods", "", "| Method | What it is |", "| --- | --- |",
           *[f"| {name} | {about} |" for name, about in METHODS.values()], ""]

    # The spread each validation season simulated with.
    out += ["## How far frozen predictions miss", "",
            "Each validation season's spread is fit on the frozen predictions of the seasons before it, from the "
            "same snapshot days. A game's miss is the home team's lasting part minus the away team's, plus that "
            "night's luck, so the typical miss is the square root of twice the lasting part squared plus the "
            "nightly part squared. Independent games simulate every game with the typical miss; lasting team "
            "errors draw each team's lasting part once per simulated season, plus nightly luck.", "",
            "| Validation season | Fit on | Frozen predictions | Typical miss, points | Lasting team part | "
            "Nightly part |", "|" + " --- |" * 6]
    for season, sp in spreads.items():
        out.append(f"| {season} | {', '.join(sp['seasons'])} | {sp['games']:,} | {sp['sd_total']:.2f} | "
                   f"{sp['sd_team']:.2f} | {sp['sd_game']:.2f} |")

    # Win totals.
    pace = base["WINS_NOW"] / base["PLAYED"].where(base["PLAYED"] > 0) * base["GAMES"]
    out += ["", "## Win totals", "",
            "Reported only. The average simulated win total against actual wins, and how often actual wins landed "
            f"inside the middle {RANGE:.0%} of simulated seasons (rounded outward to whole wins). Current pace "
            "scales each team's record so far to its full schedule.", "",
            "| Forecast | Forecasts | Average miss, wins | Inside the range |", "| --- | --- | --- | --- |"]
    for m in order:
        d = by[m]
        inside = ((d["WINS_LOW"] <= d["ACTUAL_WINS"]) & (d["ACTUAL_WINS"] <= d["WINS_HIGH"])).mean()
        out.append(f"| {METHODS[m][0]} | {len(d)} | {(d['WINS'] - d['ACTUAL_WINS']).abs().mean():.2f} | "
                   f"{inside:.1%} |")
    out.append(f"| Current pace | {len(base)} | {(pace - base['ACTUAL_WINS']).abs().mean():.2f} | - |")

    # Calibration.
    short = {m: METHODS[m][0].split()[0] for m in order}
    bins = {m: probability.calibration_rows(probs[m], made) for m in order}
    out += ["", "## Calibration", "",
            "Reported only. The chance of making the playoffs, grouped in tenths, against how often those teams "
            "made it.", "",
            "| Chance of playoffs | " + " | ".join(f"{short[m]} forecasts | {short[m]} said | Made it, "
                                                 f"{short[m].lower()}" for m in order) + " |",
            "|" + " --- |" * (1 + 3 * len(order))]
    for b in range(10):
        cells = [f"{bins[m][b]['games']} | {_pct(bins[m][b]['said'])} | "
                 f"{_pct(bins[m][b]['happened'])}" for m in order]
        out.append(f"| {10 * b}-{10 * b + 10}% | " + " | ".join(cells) + " |")

    # By snapshot day.
    out += ["", "## By snapshot day", "",
            "Reported only. Both validation seasons together, by the day the simulation started.", "",
            "| Simulated from | Forecasts | " + " | ".join(f"Log loss, {short[m].lower()}" for m in order) + " | "
            + " | ".join(f"Wins miss, {short[m].lower()}" for m in order) + " |",
            "|" + " --- |" * (2 + 2 * len(order))]
    month = base["SNAPSHOT"].dt.month
    for m_num in SNAPSHOT_MONTHS:
        at = (month == m_num).to_numpy()
        if not at.any():
            continue
        loss = [f"{probability.log_loss(probs[m][at], made[at]).mean():.4f}" for m in order]
        miss = [f"{(by[m]['WINS'] - by[m]['ACTUAL_WINS'])[at].abs().mean():.2f}" for m in order]
        out.append(f"| {_first_of(m_num)} | {at.sum()} | " + " | ".join(loss + miss) + " |")

    # The champion's title odds.
    days = sorted(base["SNAPSHOT"].dt.month.unique(), key=SNAPSHOT_MONTHS.index)
    out += ["", "## The champion", "",
            f"Reported only. The eventual champion's title odds under {in_use.lower()} (the method in use) from each "
            "snapshot day, with its rank among the 30 teams.", "",
            "| Season | Champion | " + " | ".join(_first_of(d) for d in days) + " |",
            "|" + " --- |" * (2 + len(days))]
    for season in seasons:
        s = base[base["SEASON"] == season]
        champ = s.loc[s["CHAMPION"] == 1, "TEAM_ID"]
        if champ.empty:
            continue
        cells = []
        for d in days:
            snap = s[s["SNAPSHOT"].dt.month == d]
            p = float(snap.loc[snap["TEAM_ID"] == champ.iloc[0], "TITLE"].iloc[0]) if len(snap) else np.nan
            cells.append(f"{p:.1%} ({_ordinal(int((snap['TITLE'] > p).sum()) + 1)})" if len(snap) else "-")
        out.append(f"| {season} | {abbr.get(champ.iloc[0], champ.iloc[0])} | " + " | ".join(cells) + " |")
    return out
