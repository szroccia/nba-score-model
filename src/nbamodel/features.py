"""Build model inputs: one row per team per game, predicting that team's points."""

import pandas as pd

from . import config, lineup, ratings

FEATURES = [
    "BASE_PTS",          # pace x (team offense + opponent defense - league average) / 100
    "LINEUP_OFF_PTS",    # points gained or lost from who plays for this team tonight
    "LINEUP_DEF_PTS",    # points gained or lost from who plays for the opponent tonight
    "IS_HOME",
    "REST",              # days off before this game, capped
    "OPP_REST",
    "B2B",               # second night of a back-to-back
    "OPP_B2B",
]
# Four-factor matchup edges, in percentage points: this team's offense plus the opponent's
# defense, minus twice the league average. Used only when config.FOUR_FACTORS is on.
FACTOR_FEATURES = {
    "EFG_EDGE": "EFG_PCT",   # shooting: effective field goal %
    "TOV_EDGE": "TOV_PCT",   # turnovers per possession
    "ORB_EDGE": "ORB_PCT",   # share of its own misses the offense rebounds
    "FT_EDGE": "FT_RATE",    # free throws made per field goal attempt
}


def inputs() -> list[str]:
    """The regression's inputs under the current settings."""
    return FEATURES + (list(FACTOR_FEATURES) if config.FOUR_FACTORS else [])


def settings() -> dict:
    """Every setting that changes the features. Saved features are reused only when all of
    these match, so changing any of them (to tune it, say) rebuilds the features."""
    return {
        "feature_version": config.FEATURE_VERSION,
        "playoff_games": config.PLAYOFF_GAMES,
        "half_life": config.RATING_HALF_LIFE_GAMES,
        "prior_season_weight": config.PRIOR_SEASON_GAME_WEIGHT,
        "shrink_games": config.SHRINK_GAMES,
        # Names the method, so features saved under the first version (opponents compared
        # with the league average) are never reused; features with it off are unchanged.
        "opponent_adjust": "vs average team" if config.OPPONENT_ADJUST else False,
        "replacement_obpm": config.REPLACEMENT_OBPM,
        "replacement_dbpm": config.REPLACEMENT_DBPM,
        "bpm_prior_minutes": config.BPM_PRIOR_MINUTES,
        "minutes_half_life": config.MINUTES_HALF_LIFE_GAMES,
        "rotation_minutes": config.ROTATION_MINUTES,
        "max_rest_days": config.MAX_REST_DAYS,
        "active_window_games": lineup.ACTIVE_WINDOW_GAMES,
        "team_minutes": lineup.TEAM_MINUTES,
    }


def matchup_rows(games: pd.DataFrame, player_games: pd.DataFrame, impact: pd.DataFrame,
                 date: pd.Timestamp, season: str, home_id: int, away_id: int,
                 home_probs: dict, away_probs: dict, league: dict,
                 home_roster: set | None = None, away_roster: set | None = None) -> list[dict]:
    """Feature rows for both teams in one game, using only information from before the date.

    Rosters are sets of player ids; without them, everyone from recent games is assumed
    to still be on the team."""
    rating = {t: ratings.team_rating(games, t, date, season, league) for t in (home_id, away_id)}
    rest = {t: ratings.rest_days(games, t, date) for t in (home_id, away_id)}
    adj = {
        team: lineup.lineup_adjustment(lineup.recent_minutes(player_games, team, date, roster), probs, impact)
        for team, roster, probs in ((home_id, home_roster, home_probs), (away_id, away_roster, away_probs))
    }
    pace = rating[home_id]["PACE"] * rating[away_id]["PACE"] / league["PACE"]
    rows = []
    for team, opp, is_home in ((home_id, away_id, 1), (away_id, home_id, 0)):
        rows.append(
            {
                "TEAM_ID": team,
                "OPP_TEAM_ID": opp,
                "PACE_EXP": pace,
                "ORTG": rating[team]["ORTG"],
                "OPP_DRTG": rating[opp]["DRTG"],
                "GAMES_PLAYED": rating[team]["GAMES"],
                "SEASON_GAMES": rating[team]["CURRENT_GAMES"],
                "BASE_PTS": pace / 100 * (rating[team]["ORTG"] + rating[opp]["DRTG"] - league["ORTG"]),
                "LINEUP_OFF_PTS": pace / 100 * adj[team]["ORTG_DELTA"],
                "LINEUP_DEF_PTS": pace / 100 * adj[opp]["DRTG_DELTA"],
                "IS_HOME": is_home,
                "REST": rest[team],
                "OPP_REST": rest[opp],
                "B2B": int(rest[team] == 0),
                "OPP_B2B": int(rest[opp] == 0),
                **{name: 100 * (rating[team][f] + rating[opp]["OPP_" + f] - 2 * league[f])
                   for name, f in FACTOR_FEATURES.items()},
                "MISSING": adj[team]["MISSING"],
            }
        )
    return rows


def historical_features(games: pd.DataFrame, player_games: pd.DataFrame, impact_by_season: dict,
                        seasons: list[str]) -> pd.DataFrame:
    """Features for every past game in the given seasons, plus the actual points scored.

    Lineups use who actually played as that night's roster (perfect knowledge of
    availability, but not of minutes)."""
    out = []
    played = player_games.groupby(["GAME_ID", "TEAM_ID"])["PLAYER_ID"].apply(set)
    target = games[games["SEASON"].isin(seasons) & (games["IS_HOME"] == 1)]
    league_cache: dict = {}
    for g in target.itertuples():
        date, season = g.GAME_DATE, g.SEASON
        if (date, season) not in league_cache:
            league_cache[(date, season)] = ratings.league_average(games, date, season)
        rows = matchup_rows(games, player_games, impact_by_season[season], date, season,
                            g.TEAM_ID, g.OPP_TEAM_ID, {}, {}, league_cache[(date, season)],
                            played.get((g.GAME_ID, g.TEAM_ID), set()),
                            played.get((g.GAME_ID, g.OPP_TEAM_ID), set()))
        actual = {g.TEAM_ID: g.PTS, g.OPP_TEAM_ID: g.OPP_PTS}
        for row in rows:
            row.update({"GAME_ID": g.GAME_ID, "GAME_DATE": date, "SEASON": season,
                        "PTS": actual[row["TEAM_ID"]]})
            out.append(row)
    return pd.DataFrame(out)
