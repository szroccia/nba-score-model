"""Sanity checks on the downloaded data, plus games to spot-check by hand on nba.com."""

import pandas as pd

from . import config, data
from .names import normalize_name

REGULAR_SEASON_GAMES = 1230


def _game_id(gid) -> str:
    return str(gid).zfill(10)


def check_season(season: str, complete: bool) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for one season's team and player game logs."""
    teams = data.team_games(season)
    players = data.player_games(season)
    results = []
    if teams.empty:
        return [("games downloaded", not complete, "no games yet")]

    ids = teams["GAME_ID"].map(_game_id)
    regular = ids[ids.str.startswith("002")].nunique()
    playoff = ids[ids.str.startswith("004")].nunique()
    if complete:
        results.append(("regular season game count", regular == REGULAR_SEASON_GAMES,
                        f"{regular} games (expected {REGULAR_SEASON_GAMES}), {playoff} playoff games"))
    else:
        results.append(("games so far", True, f"{regular} regular season games played"))

    results.append(("30 teams", teams["TEAM_ID"].nunique() == 30, f"{teams['TEAM_ID'].nunique()} teams"))

    per_game = teams.groupby("GAME_ID")["TEAM_ID"].nunique()
    results.append(("two teams in every game", bool((per_game == 2).all()),
                    f"{int((per_game != 2).sum())} games without exactly two teams"))

    dupes = int(teams.duplicated(["GAME_ID", "TEAM_ID"]).sum())
    results.append(("no duplicate rows", dupes == 0, f"{dupes} duplicates"))

    made = 2 * teams["FGM"] + teams["FG3M"] + teams["FTM"]
    bad = int((made != teams["PTS"]).sum())
    results.append(("points = 2 x FGM + 3PM + FTM", bad == 0, f"{bad} rows where the box score doesn't add up"))

    # Team totals and player totals are downloaded separately; they should agree.
    player_pts = players.groupby(["GAME_ID", "TEAM_ID"])["PTS"].sum()
    joined = teams.set_index(["GAME_ID", "TEAM_ID"])["PTS"].to_frame("TEAM").join(player_pts.rename("PLAYERS"))
    mismatch = int((joined["TEAM"] != joined["PLAYERS"]).sum())
    results.append(("team points = sum of player points", mismatch == 0,
                    f"{mismatch} of {len(joined)} team-games differ"))

    pts = teams["PTS"]
    results.append(("scores in a realistic range", bool(pts.between(60, 180).all()),
                    f"lowest {pts.min()}, highest {pts.max()}, average {pts.mean():.1f}"))
    return results


def check_bpm(season: str) -> list[tuple[str, bool, str]]:
    """How many of the season's minutes belong to players we found a BPM for."""
    bpm = data.bpm(season)
    players = data.player_games(season)
    if bpm.empty or players.empty:
        return [("BPM downloaded", True, "no BPM yet for this season")]
    keys = set(bpm["Player"].map(normalize_name))
    mins = players.groupby("PLAYER_NAME")["MIN"].sum()
    matched = mins[[normalize_name(n) in keys for n in mins.index]].sum() / mins.sum()
    missing = mins[[normalize_name(n) not in keys for n in mins.index]].sort_values(ascending=False)
    detail = f"{matched:.1%} of minutes matched to a BPM ({len(bpm)} players)"
    if len(missing):
        detail += "; biggest misses: " + ", ".join(missing.index[:3])
    return [("BPM names match box scores", matched >= 0.98, detail)]


def spot_check_games(season: str, n: int = 3, seed: int | None = None) -> pd.DataFrame:
    """A few random games with links to the official box scores."""
    teams = data.team_games(season)
    if teams.empty:
        return pd.DataFrame()
    picks = teams.drop_duplicates("GAME_ID").sample(min(n, teams["GAME_ID"].nunique()), random_state=seed)
    rows = []
    for gid in picks["GAME_ID"]:
        game = teams[teams["GAME_ID"] == gid]
        rows.append({
            "date": pd.Timestamp(game["GAME_DATE"].iloc[0]).date(),
            "score": " - ".join(f"{r.TEAM_ABBREVIATION} {r.PTS}" for r in game.itertuples()),
            "box score": f"https://www.nba.com/game/{_game_id(gid)}",
        })
    return pd.DataFrame(rows)


def run(seasons: list[str], current: str) -> bool:
    all_passed = True
    for season in seasons:
        print(f"\n{season}")
        for name, passed, detail in check_season(season, complete=season != current) + check_bpm(season):
            all_passed &= passed
            print(f"  [{'PASS' if passed else 'WARN'}] {name}: {detail}")
    print("\nSpot-check these against the official box scores:")
    for season in seasons[-2:]:
        games = spot_check_games(season)
        if not games.empty:
            print(games.to_string(index=False))
    print(f"\nRaw files are in {config.DATA_DIR}")
    return all_passed
