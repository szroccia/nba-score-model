"""Turn who is playing tonight into a change in team offensive and defensive rating.

A team's recent ratings already reflect the players who have been playing.
We compare tonight's expected minutes with those recent minutes and value the
difference with each player's Box Plus/Minus (points per 100 possessions):

    change in ORtg = sum over players of OBPM x (expected minutes - recent minutes) / 48
    change in DRtg = -sum over players of DBPM x (expected minutes - recent minutes) / 48
"""

import numpy as np
import pandas as pd

from . import config
from .names import normalize_name

TEAM_MINUTES = 240
ACTIVE_WINDOW_GAMES = 20


def player_impact(bpm_prior: pd.DataFrame, bpm_current: pd.DataFrame | None = None) -> pd.DataFrame:
    """OBPM and DBPM per player, blending last season with this season by minutes played.

    Pass bpm_current=None for backtests: Basketball-Reference only has season-to-date
    BPM for today, so past games can only use the previous season without leaking.
    """
    def prep(df):
        out = df.copy()
        out["NAME_KEY"] = out["Player"].map(normalize_name)
        return out.drop_duplicates("NAME_KEY").set_index("NAME_KEY")[["MP", "OBPM", "DBPM"]]

    prior = prep(bpm_prior)
    # Players with few minutes last season are pulled toward replacement level.
    k = config.BPM_PRIOR_MINUTES / 2
    prior_obpm = (prior["OBPM"] * prior["MP"] + config.REPLACEMENT_OBPM * k) / (prior["MP"] + k)
    prior_dbpm = (prior["DBPM"] * prior["MP"] + config.REPLACEMENT_DBPM * k) / (prior["MP"] + k)
    impact = pd.DataFrame({"OBPM": prior_obpm, "DBPM": prior_dbpm})
    if bpm_current is None:
        return impact

    cur = prep(bpm_current)
    keys = impact.index.union(cur.index)
    impact = impact.reindex(keys)
    impact["OBPM"] = impact["OBPM"].fillna(config.REPLACEMENT_OBPM)
    impact["DBPM"] = impact["DBPM"].fillna(config.REPLACEMENT_DBPM)
    cur = cur.reindex(keys)
    mp = cur["MP"].fillna(0)
    p = config.BPM_PRIOR_MINUTES
    impact["OBPM"] = (impact["OBPM"] * p + cur["OBPM"].fillna(0) * mp) / (p + mp)
    impact["DBPM"] = (impact["DBPM"] * p + cur["DBPM"].fillna(0) * mp) / (p + mp)
    return impact


def recent_minutes(player_games: pd.DataFrame, team_id: int, date: pd.Timestamp,
                   roster_ids: set | None = None) -> pd.DataFrame:
    """For each player in the team's recent games: RECENT_MIN (missed games count as 0,
    matching what the team ratings saw) and TYPICAL_MIN (average when he does play).

    With a roster, players no longer on it are marked ON_ROSTER = False (they will not
    play), and newcomers are added with RECENT_MIN = 0 and TYPICAL_MIN from their recent
    games for any team. Early in a season this is what captures offseason moves."""
    pg = player_games[(player_games["TEAM_ID"] == team_id) & (player_games["GAME_DATE"] < date)]
    team_game_ids = (
        pg.drop_duplicates("GAME_ID").sort_values("GAME_DATE", ascending=False)["GAME_ID"]
        .head(ACTIVE_WINDOW_GAMES).tolist()
    )
    out = pd.DataFrame(columns=["PLAYER_NAME", "RECENT_MIN", "TYPICAL_MIN"])
    if team_game_ids:
        pg = pg[pg["GAME_ID"].isin(team_game_ids)]
        grid = pg.pivot_table(index="PLAYER_ID", columns="GAME_ID", values="MIN", aggfunc="sum")
        grid = grid.reindex(columns=team_game_ids)
        w = 0.5 ** (np.arange(len(team_game_ids)) / config.MINUTES_HALF_LIFE_GAMES)
        played = grid.notna().to_numpy()
        mins = grid.fillna(0).to_numpy()
        names = pg.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["PLAYER_NAME"]
        out = pd.DataFrame(
            {
                "PLAYER_NAME": names.reindex(grid.index).values,
                "RECENT_MIN": mins @ w / w.sum(),
                "TYPICAL_MIN": (mins * w).sum(axis=1) / np.maximum((played * w).sum(axis=1), 1e-9),
            },
            index=grid.index,
        )
    if roster_ids is None:
        return out

    out["ON_ROSTER"] = out.index.isin(roster_ids)
    newcomers = set(roster_ids) - set(out.index)
    if newcomers:
        elsewhere = player_games[player_games["PLAYER_ID"].isin(newcomers) & (player_games["GAME_DATE"] < date)]
        elsewhere = elsewhere.sort_values("GAME_DATE", ascending=False).groupby("PLAYER_ID").head(ACTIVE_WINDOW_GAMES)
        elsewhere = elsewhere.assign(
            RANK=elsewhere.groupby("PLAYER_ID").cumcount(),
        )
        elsewhere["W"] = 0.5 ** (elsewhere["RANK"] / config.MINUTES_HALF_LIFE_GAMES)
        elsewhere["WMIN"] = elsewhere["W"] * elsewhere["MIN"]
        agg = elsewhere.groupby("PLAYER_ID").agg(PLAYER_NAME=("PLAYER_NAME", "first"), WMIN=("WMIN", "sum"), W=("W", "sum"))
        added = pd.DataFrame(
            {"PLAYER_NAME": agg["PLAYER_NAME"], "RECENT_MIN": 0.0, "TYPICAL_MIN": agg["WMIN"] / agg["W"], "ON_ROSTER": True},
            index=agg.index,
        )
        out = pd.concat([out, added]) if not out.empty else added
    return out


def lineup_adjustment(minutes: pd.DataFrame, play_prob: dict, impact: pd.DataFrame) -> dict:
    """Change in team ORtg and DRtg given each player's chance of playing tonight.

    play_prob maps normalized player name -> probability; players not listed are
    assumed to play. Rotation players (usually 20+ minutes when they play) are expected
    at their usual minutes, so a starter back from injury counts in full; everyone else
    at their recent average including games they sat out, so a deep bench of rarely used
    players doesn't squeeze the starters. Missing minutes go to teammates in proportion.
    """
    if minutes.empty:
        return {"ORTG_DELTA": 0.0, "DRTG_DELTA": 0.0, "MISSING": ""}
    df = minutes.copy()
    df["NAME_KEY"] = df["PLAYER_NAME"].map(normalize_name)
    df["P_PLAY"] = df["NAME_KEY"].map(play_prob).fillna(1.0)
    if "ON_ROSTER" in df:
        df.loc[~df["ON_ROSTER"].astype(bool), "P_PLAY"] = 0.0
    rotation = df["TYPICAL_MIN"] >= config.ROTATION_MINUTES
    newcomer = df["RECENT_MIN"] == 0
    base = df["RECENT_MIN"].where(~(rotation | newcomer), df["TYPICAL_MIN"])
    expected = base * df["P_PLAY"]
    if expected.sum() > 0:
        expected = expected * TEAM_MINUTES / expected.sum()
    # Cap any one player at 40 minutes and spread the excess.
    for _ in range(5):
        over = (expected - 40).clip(lower=0)
        if over.sum() < 1e-6:
            break
        expected = expected.clip(upper=40)
        room = (expected < 40) & (expected > 0)
        expected[room] += over.sum() * expected[room] / expected[room].sum()
    df["EXPECTED_MIN"] = expected
    df = df.join(impact, on="NAME_KEY")
    df["OBPM"] = df["OBPM"].fillna(config.REPLACEMENT_OBPM)
    df["DBPM"] = df["DBPM"].fillna(config.REPLACEMENT_DBPM)
    change = (df["EXPECTED_MIN"] - df["RECENT_MIN"]) / 48
    on_roster = df["ON_ROSTER"].astype(bool) if "ON_ROSTER" in df else True
    missing = df[(df["P_PLAY"] < 1) & (df["TYPICAL_MIN"] >= 10) & on_roster].sort_values("TYPICAL_MIN", ascending=False)
    return {
        "ORTG_DELTA": float((df["OBPM"] * change).sum()),
        "DRTG_DELTA": float(-(df["DBPM"] * change).sum()),
        "MISSING": "; ".join(f"{n} ({p:.0%})" for n, p in zip(missing["PLAYER_NAME"], missing["P_PLAY"])),
    }
