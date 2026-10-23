"""Team pace and offensive/defensive ratings as of any date, using only earlier games."""

import numpy as np
import pandas as pd

from . import config


def team_game_table(team_games: pd.DataFrame) -> pd.DataFrame:
    """Turn the league game log into one row per team per game with possessions and ratings."""
    df = team_games.copy()
    df["POSS_EST"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    # Playoff game ids start with 004; regular season with 002.
    df["PLAYOFF"] = df["GAME_ID"].astype(str).str.zfill(10).str.startswith("004")
    opp = df[["GAME_ID", "TEAM_ID", "PTS", "POSS_EST"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "POSS_EST": "OPP_POSS_EST"}
    )
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()
    df["POSS"] = (df["POSS_EST"] + df["OPP_POSS_EST"]) / 2
    # Overtime adds possessions without changing efficiency; scale pace to 48 minutes.
    df["PACE"] = df["POSS"] * 240 / df["MIN"].clip(lower=240)
    df["ORTG"] = 100 * df["PTS"] / df["POSS"]
    df["DRTG"] = 100 * df["OPP_PTS"] / df["POSS"]
    cols = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "OPP_TEAM_ID",
            "IS_HOME", "PLAYOFF", "PTS", "OPP_PTS", "PACE", "ORTG", "DRTG"]
    df = df[cols].sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
    return playoff_relative(df) if config.PLAYOFF_GAMES == "relative" else df


def _seasons_in_window(season: str) -> set:
    start = int(season[:4]) - 1
    return {season, f"{start}-{str(start + 1)[2:]}"}


def _rating_games(games: pd.DataFrame) -> pd.DataFrame:
    """Games that feed ratings: all of them, unless PLAYOFF_GAMES is "exclude"."""
    if config.PLAYOFF_GAMES != "exclude" or "PLAYOFF" not in games:
        return games
    return games[~games["PLAYOFF"].astype(bool)]


def playoff_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale playoff games to the regular season's scoring environment.

    Each playoff game's pace and ratings are shifted by how far the playoffs so far
    (through that date) sit from that season's regular-season average, so a team is
    judged against other playoff teams rather than penalized for the slower game."""
    df = df.copy()
    for season, rows in df.groupby("SEASON").groups.items():
        season_df = df.loc[rows]
        regular = season_df[~season_df["PLAYOFF"]]
        playoff = season_df[season_df["PLAYOFF"]].sort_values("GAME_DATE")
        if regular.empty or playoff.empty:
            continue
        running = playoff[["PACE", "ORTG"]].expanding().mean()
        df.loc[playoff.index, "PACE"] = playoff["PACE"] * regular["PACE"].mean() / running["PACE"]
        shift = regular["ORTG"].mean() - running["ORTG"]
        df.loc[playoff.index, "ORTG"] = playoff["ORTG"] + shift
        df.loc[playoff.index, "DRTG"] = playoff["DRTG"] + shift
    return df


def league_average(games: pd.DataFrame, date: pd.Timestamp, season: str) -> dict:
    """League average pace and rating over this season and last, before the date."""
    games = _rating_games(games)
    past = games[(games["GAME_DATE"] < date) & games["SEASON"].isin(_seasons_in_window(season))]
    current = past[past["SEASON"] == season]
    # Weight this season's games fully and last season's lightly, like team ratings do.
    w = np.where(past["SEASON"] == season, 1.0, config.PRIOR_SEASON_GAME_WEIGHT)
    if len(past) == 0:
        return {"PACE": 99.0, "ORTG": 114.0}
    if len(current) == 0:
        w = np.ones(len(past))
    return {
        "PACE": float(np.average(past["PACE"], weights=w)),
        "ORTG": float(np.average(past["ORTG"], weights=w)),
    }


def team_rating(games: pd.DataFrame, team_id: int, date: pd.Timestamp, season: str,
                league: dict) -> dict:
    """Recency-weighted pace, offensive and defensive rating for one team before a date."""
    games = _rating_games(games)
    past = games[
        (games["TEAM_ID"] == team_id)
        & (games["GAME_DATE"] < date)
        & games["SEASON"].isin(_seasons_in_window(season))
    ].sort_values("GAME_DATE", ascending=False)
    games_ago = np.arange(len(past))
    w = 0.5 ** (games_ago / config.RATING_HALF_LIFE_GAMES)
    w = w * np.where(past["SEASON"] == season, 1.0, config.PRIOR_SEASON_GAME_WEIGHT)
    shrink = config.SHRINK_GAMES
    total = w.sum() + shrink

    def blend(col, prior):
        return float((np.dot(w, past[col]) + shrink * prior) / total)

    return {
        "PACE": blend("PACE", league["PACE"]),
        "ORTG": blend("ORTG", league["ORTG"]),
        "DRTG": blend("DRTG", league["ORTG"]),
        "GAMES": int(len(past)),
        "CURRENT_GAMES": int((past["SEASON"] == season).sum()),
    }


def rest_days(games: pd.DataFrame, team_id: int, date: pd.Timestamp) -> int:
    """Days since the team's last game, capped. Day after a game = 0 (back-to-back)."""
    past = games[(games["TEAM_ID"] == team_id) & (games["GAME_DATE"] < date)]
    if past.empty:
        return config.MAX_REST_DAYS
    days = (date - past["GAME_DATE"].max()).days - 1
    return int(min(days, config.MAX_REST_DAYS))
