"""Historical closing Vegas lines from ESPN, used as the benchmark the model has to beat.

ESPN's public API keeps each game's pregame spread and over/under from several
sportsbooks. The scoreboard gives the day's games; a second call per game gives
its lines. Both load from cloud servers and home connections alike.
"""

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config, data

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day}"
ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
            "events/{event}/competitions/{event}/odds")
# Consensus first; then single books ESPN has carried across all seasons.
PROVIDER_ORDER = ["consensus", "ESPN BET", "DraftKings", "Caesars Sportsbook", "MGM"]
# ESPN abbreviations that differ from nba.com's.
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def _get_json(url: str, tries: int = 3) -> dict:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def nba_abbreviation(espn: str) -> str:
    return ESPN_TO_NBA.get(espn, espn)


def home_line(details: str, home_espn: str) -> float | None:
    """Expected home margin from a line like 'NY -5.5' (NY favored by 5.5). 'EVEN' is 0."""
    if not details:
        return None
    if details.strip().upper() in {"EVEN", "PK", "PICK"}:
        return 0.0
    m = re.match(r"^\s*(\S+)\s+(-?\d+(?:\.\d+)?)\s*$", details)
    if not m:
        return None
    favorite, points = m.group(1), abs(float(m.group(2)))
    return points if favorite == home_espn else -points


def pick_line(items: list[dict], home_espn: str) -> dict | None:
    """The preferred pregame (not live) line among the books ESPN lists."""
    pregame = [i for i in items if "live" not in i.get("provider", {}).get("name", "").lower()]
    rank = {name: k for k, name in enumerate(PROVIDER_ORDER)}
    pregame.sort(key=lambda i: rank.get(i.get("provider", {}).get("name"), len(rank)))
    for item in pregame:
        margin = home_line(item.get("details", ""), home_espn)
        total = item.get("overUnder")
        if margin is not None and total:
            return {"VEGAS_HOME_MARGIN": margin, "VEGAS_TOTAL": float(total),
                    "BOOK": item["provider"]["name"]}
    return None


def _day_games(day: pd.Timestamp) -> list[dict]:
    board = _get_json(SCOREBOARD_URL.format(day=day.strftime("%Y%m%d")))
    games = []
    for event in board.get("events", []):
        # Season type 2 is the regular season, 3 the playoffs; skip preseason and All-Star.
        if event.get("season", {}).get("type") not in (2, 3, 5):
            continue
        sides = {c["homeAway"]: c for c in event["competitions"][0]["competitors"]}
        games.append({"EVENT_ID": event["id"], "GAME_DATE": day,
                      "HOME_ESPN": sides["home"]["team"]["abbreviation"],
                      "AWAY_ESPN": sides["away"]["team"]["abbreviation"],
                      # Final scores, kept as a cross-check on nba.com's box scores.
                      "HOME_PTS": pd.to_numeric(sides["home"].get("score"), errors="coerce"),
                      "AWAY_PTS": pd.to_numeric(sides["away"].get("score"), errors="coerce")})
    return games


def _with_line(game: dict) -> dict:
    try:
        items = _get_json(ODDS_URL.format(event=game["EVENT_ID"])).get("items", [])
    except Exception:
        items = []
    line = pick_line(items, game["HOME_ESPN"]) or {}
    return {**game, **line}


def fetch_season(season: str, workers: int = 8) -> pd.DataFrame:
    """Every game of a season (October to June) with its closing spread and total."""
    end = data.season_end_year(season)
    days = pd.date_range(f"{end - 1}-10-01", f"{end}-06-30")
    today = pd.Timestamp.today().normalize()
    days = days[days < today]
    with ThreadPoolExecutor(workers) as pool:
        games = [g for day_games in pool.map(_day_games, days) for g in day_games]
        rows = list(pool.map(_with_line, games))
    df = pd.DataFrame(rows, columns=["EVENT_ID", "GAME_DATE", "HOME_ESPN", "AWAY_ESPN", "HOME_PTS",
                                     "AWAY_PTS", "VEGAS_HOME_MARGIN", "VEGAS_TOTAL", "BOOK"])
    df["HOME"] = df["HOME_ESPN"].map(nba_abbreviation)
    df["AWAY"] = df["AWAY_ESPN"].map(nba_abbreviation)
    df["SEASON"] = season
    return df


def lines(season: str, refresh: bool = False) -> pd.DataFrame:
    """Cached closing lines for a season (data/vegas_<season>.csv)."""
    path = config.DATA_DIR / f"vegas_{season}.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path)
    else:
        print(f"Downloading Vegas lines for {season} from ESPN (a few minutes)...")
        df = fetch_season(season)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def team_abbreviations() -> dict:
    """nba.com team id -> abbreviation (bundled with nba_api, no download)."""
    from nba_api.stats.static import teams

    return {t["id"]: t["abbreviation"] for t in teams.get_teams()}


def attach(games: pd.DataFrame, vegas: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas columns to a one-row-per-game frame with GAME_DATE, HOME and AWAY."""
    keep = vegas.dropna(subset=["VEGAS_HOME_MARGIN"])[
        ["GAME_DATE", "HOME", "AWAY", "VEGAS_HOME_MARGIN", "VEGAS_TOTAL"]]
    return games.merge(keep.drop_duplicates(["GAME_DATE", "HOME", "AWAY"]),
                       on=["GAME_DATE", "HOME", "AWAY"], how="left")
