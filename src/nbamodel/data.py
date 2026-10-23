"""Download and cache game, player and schedule data.

Team and player box scores and the schedule come from stats.nba.com through
nba_api. Box Plus/Minus comes from Basketball-Reference (one page per season).
stats.nba.com blocks most cloud servers, so run this from a home connection.
"""

import io
import time
import urllib.error
import urllib.request

import pandas as pd

from . import config

_NBA_API_PAUSE_SECONDS = 0.8


def season_end_year(season: str) -> int:
    """'2025-26' -> 2026."""
    return int(season[:4]) + 1


def season_for_date(date: pd.Timestamp) -> str:
    """Seasons start in October: 2026-10-20 -> '2026-27', 2026-03-01 -> '2025-26'."""
    start = date.year if date.month >= 9 else date.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[2:]}"


def _cached(name: str, fetch, refresh: bool) -> pd.DataFrame:
    path = config.DATA_DIR / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = fetch()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _league_game_log(season: str, player_or_team: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    time.sleep(_NBA_API_PAUSE_SECONDS)
    frames = []
    for season_type in ["Regular Season", "Playoffs"]:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation=player_or_team,
            timeout=60,
        )
        frames.append(log.get_data_frames()[0])
    return pd.concat(frames, ignore_index=True)


def team_games(season: str, refresh: bool = False) -> pd.DataFrame:
    """One row per team per game, with box score totals."""
    df = _cached(f"team_games_{season}", lambda: _league_game_log(season, "T"), refresh)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = season
    return df


def player_games(season: str, refresh: bool = False) -> pd.DataFrame:
    """One row per player per game played, with minutes."""
    df = _cached(f"player_games_{season}", lambda: _league_game_log(season, "P"), refresh)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = season
    return df


def schedule(season: str, refresh: bool = False) -> pd.DataFrame:
    """Every game of the season with home and away team ids."""
    from nba_api.stats.endpoints import scheduleleaguev2

    def fetch():
        time.sleep(_NBA_API_PAUSE_SECONDS)
        raw = scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0]
        return pd.DataFrame(
            {
                "GAME_ID": raw["gameId"],
                "GAME_DATE": pd.to_datetime(raw["gameDateEst"].astype(str).str[:10]),
                "GAME_TIME_UTC": raw["gameDateTimeUTC"],
                "HOME_TEAM_ID": raw["homeTeam_teamId"],
                "HOME_TEAM": raw["homeTeam_teamTricode"],
                "AWAY_TEAM_ID": raw["awayTeam_teamId"],
                "AWAY_TEAM": raw["awayTeam_teamTricode"],
            }
        )

    df = _cached(f"schedule_{season}", fetch, refresh)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # Preseason and All-Star games have ids that don't start with 002 (regular) or 004 (playoffs).
    return df[df["GAME_ID"].astype(str).str.zfill(10).str[:3].isin(["002", "004"])]


def bpm(season: str, refresh: bool = False) -> pd.DataFrame:
    """Season Box Plus/Minus per player from Basketball-Reference."""

    def fetch():
        url = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year(season)}_advanced.html"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            html = urllib.request.urlopen(request, timeout=60).read().decode()
        except urllib.error.HTTPError as err:
            if err.code == 404:  # the season's page appears after its first games
                return pd.DataFrame(columns=["Player", "MP", "OBPM", "DBPM", "BPM"])
            raise
        try:
            table = pd.read_html(io.StringIO(html), attrs={"id": "advanced"}, flavor="lxml")[0]
        except ValueError:  # page exists but no stats yet
            return pd.DataFrame(columns=["Player", "MP", "OBPM", "DBPM", "BPM"])
        if not {"OBPM", "DBPM", "BPM"} <= set(table.columns):
            # Early in a season the page lists players before Box Plus/Minus is published.
            return pd.DataFrame(columns=["Player", "MP", "OBPM", "DBPM", "BPM"])
        table = table[table["Player"].notna() & (table["Player"] != "Player")]
        table = table[~table["Player"].str.contains("League Average", na=False)]
        for col in ["MP", "OBPM", "DBPM", "BPM"]:
            table[col] = pd.to_numeric(table[col], errors="coerce")
        # Traded players have a combined row plus one per team; keep the combined one.
        table = table.sort_values("MP", ascending=False).drop_duplicates("Player")
        return table[["Player", "MP", "OBPM", "DBPM", "BPM"]].reset_index(drop=True)

    return _cached(f"bpm_{season}", fetch, refresh)


def rosters(season: str, refresh: bool = False) -> pd.DataFrame:
    """Current roster of every team (one call per team)."""
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams

    def fetch():
        frames = []
        for team in teams.get_teams():
            time.sleep(_NBA_API_PAUSE_SECONDS)
            roster = commonteamroster.CommonTeamRoster(team_id=team["id"], season=season, timeout=60)
            frames.append(roster.common_team_roster.get_data_frame())
        df = pd.concat(frames, ignore_index=True)
        return df.rename(columns={"TeamID": "TEAM_ID"})[["TEAM_ID", "PLAYER_ID", "PLAYER"]]

    return _cached(f"rosters_{season}", fetch, refresh)
