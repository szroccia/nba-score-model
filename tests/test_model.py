import numpy as np
import pandas as pd
import pytest

from nbamodel import features, injuries, lineup, model, ratings
from nbamodel.names import normalize_name


def test_normalize_name_matches_across_sources():
    assert normalize_name("Dončić, Luka") == normalize_name("Luka Doncic") == "luka doncic"
    assert normalize_name("Jackson Jr., Jaren") == normalize_name("Jaren Jackson Jr.")
    assert normalize_name("Gilgeous-Alexander, Shai") == normalize_name("Shai Gilgeous-Alexander")


def test_parse_injury_report_text():
    names = {"Boston Celtics": "BOS", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL"}
    text = "\n".join([
        "Game Date Game Time Matchup Team Player Name Current Status Reason",
        "10/21/2026 07:30 (ET) LAL@BOS Boston Celtics Tatum, Jayson Out Injury/Illness - Right Achilles",
        "Brown, Jaylen Questionable Injury/Illness - Left Knee; Soreness",
        "Los Angeles Lakers James, LeBron Probable Rest",
        "LA Clippers NOT YET SUBMITTED",
    ])
    df = injuries.parse_report_text(text, names)
    assert df[["TEAM", "PLAYER", "STATUS"]].values.tolist() == [
        ["BOS", "Tatum, Jayson", "out"],
        ["BOS", "Brown, Jaylen", "questionable"],
        ["LAL", "James, LeBron", "probable"],
    ]
    probs = injuries.play_probabilities(df)
    assert probs["BOS"]["jayson tatum"] == 0.0
    assert probs["LAL"]["lebron james"] == 0.9


def _fake_league(seasons=("2023-24", "2024-25"), n_teams=4, seed=0):
    """Round-robin seasons where team strength is known, with box scores to match."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(range(1, n_teams + 1), np.linspace(-6, 6, n_teams))}
    team_rows, player_rows = [], []
    gid = 0
    for si, season in enumerate(seasons):
        day = pd.Timestamp(f"{2023 + si}-10-20")
        for rnd in range(30):
            for x in strength:
                for y in strength:
                    if x >= y:
                        continue
                    h, a = (x, y) if rnd % 2 else (y, x)
                    gid += 1
                    day += pd.Timedelta(days=1)
                    for team, opp, home in ((h, a, True), (a, h, False)):
                        pts = 112 + strength[team] - strength[opp] / 2 + (1.5 if home else -1.5) + rng.normal(0, 8)
                        team_rows.append({
                            "SEASON_ID": season, "TEAM_ID": team, "TEAM_ABBREVIATION": f"T{team}",
                            "GAME_ID": gid, "GAME_DATE": day,
                            "MATCHUP": f"T{team} vs. T{opp}" if home else f"T{team} @ T{opp}",
                            "MIN": 240, "FGA": 88, "FTA": 22, "OREB": 10, "TOV": 14, "PTS": round(pts),
                            "SEASON": season,
                        })
                        for p in range(8):
                            player_rows.append({
                                "TEAM_ID": team, "GAME_ID": gid, "GAME_DATE": day, "SEASON": season,
                                "PLAYER_ID": team * 100 + p, "PLAYER_NAME": f"Player {chr(65 + team)}{chr(65 + p)}", "MIN": 30,
                            })
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def test_ratings_use_only_past_games():
    team_games, _ = _fake_league()
    games = ratings.team_game_table(team_games)
    first = games["GAME_DATE"].min()
    league = ratings.league_average(games, first, "2023-24")
    r = ratings.team_rating(games, 1, first, "2023-24", league)
    assert r["GAMES"] == 0 and r["ORTG"] == pytest.approx(league["ORTG"])


def test_lineup_adjustment_when_star_is_out():
    minutes = pd.DataFrame({
        "PLAYER_NAME": ["Star Player", "Bench Guy"] + [f"Other {c}" for c in "ABCDEF"],
        "RECENT_MIN": [36, 12] + [32] * 6,
        "TYPICAL_MIN": [36, 12] + [32] * 6,
    })
    impact = pd.DataFrame({"OBPM": [6.0, -2.0] + [0] * 6, "DBPM": [2.0, -1.0] + [0] * 6},
                          index=[normalize_name(n) for n in minutes["PLAYER_NAME"]])
    healthy = lineup.lineup_adjustment(minutes, {}, impact)
    assert healthy["ORTG_DELTA"] == pytest.approx(0, abs=1e-9)
    out = lineup.lineup_adjustment(minutes, {"star player": 0.0}, impact)
    assert out["ORTG_DELTA"] < -3       # offense gets worse
    assert out["DRTG_DELTA"] > 0.5      # defense allows more
    assert "Star Player" in out["MISSING"]


def test_end_to_end_training_learns_home_edge():
    team_games, player_games = _fake_league()
    games = ratings.team_game_table(team_games)
    impact = pd.DataFrame(columns=["OBPM", "DBPM"])
    feats = features.historical_features(games, player_games, {"2024-25": impact}, ["2024-25"])
    fitted = model.fit(feats)
    assert fitted["coefficients"]["IS_HOME"] > 0
    feats["PRED_PTS"] = model.predict(fitted, feats)
    s = model.score(feats)
    assert s["team_points_mae"] < 10
    assert s["winner_pct"] > 0.5


def test_predict_date_with_fake_data(monkeypatch, tmp_path):
    from nbamodel import cli, config, data

    team_games, player_games = _fake_league(seasons=("2023-24", "2024-25"))
    day = pd.Timestamp("2025-01-15")
    monkeypatch.setattr(config, "MANUAL_STATUS_PATH", tmp_path / "manual_status.csv")
    (tmp_path / "manual_status.csv").write_text("TEAM,PLAYER,STATUS\nT1,Player BA,out\n")
    monkeypatch.setattr(data, "team_games", lambda s, refresh=False: team_games[team_games["SEASON"] == s])
    monkeypatch.setattr(data, "player_games", lambda s, refresh=False: player_games[player_games["SEASON"] == s])
    monkeypatch.setattr(data, "bpm", lambda s, refresh=False: pd.DataFrame(
        {"Player": ["Player BA"], "MP": [2000], "OBPM": [5.0], "DBPM": [1.0], "BPM": [6.0]}))
    monkeypatch.setattr(data, "schedule", lambda s, refresh=False: pd.DataFrame({
        "GAME_ID": [999], "GAME_DATE": [day], "HOME_TEAM_ID": [1], "HOME_TEAM": ["T1"],
        "AWAY_TEAM_ID": [4], "AWAY_TEAM": ["T4"]}))
    # Team 1 traded Player BB to team 4 for Player EH.
    roster = player_games[["TEAM_ID", "PLAYER_ID"]].drop_duplicates()
    roster.loc[roster["PLAYER_ID"] == 111, "TEAM_ID"] = 4
    roster.loc[roster["PLAYER_ID"] == 407, "TEAM_ID"] = 1
    monkeypatch.setattr(data, "rosters", lambda s, refresh=False: roster.assign(PLAYER=""))
    games = ratings.team_game_table(team_games)
    feats = features.historical_features(games, player_games, {"2024-25": pd.DataFrame(columns=["OBPM", "DBPM"])}, ["2024-25"])
    monkeypatch.setattr(model, "load", lambda: model.fit(feats))

    preds = cli.predict_date(day, use_injury_report=False)
    assert len(preds) == 1
    row = preds.iloc[0]
    assert 90 < row["home_pts"] < 140 and 90 < row["away_pts"] < 140
    assert "Player BA" in row["home_missing"]
    assert row["home_lineup_pts"] < 0


def test_roster_moves_count_departures_and_newcomers():
    _, player_games = _fake_league()
    day = pd.Timestamp("2025-01-15")
    base = lineup.recent_minutes(player_games, 1, day)
    roster = (set(base.index) - {100}) | {400}       # Player BA left, Player EA joined
    moved = lineup.recent_minutes(player_games, 1, day, roster)
    assert not moved.loc[100, "ON_ROSTER"]
    assert moved.loc[400, "RECENT_MIN"] == 0 and moved.loc[400, "TYPICAL_MIN"] == pytest.approx(30)
    impact = pd.DataFrame({"OBPM": [6.0, -3.0], "DBPM": [0.0, 0.0]}, index=["player ba", "player ea"])
    adj = lineup.lineup_adjustment(moved, {}, impact)
    assert adj["ORTG_DELTA"] < -3       # lost a +6 scorer, gained a -3 one
    assert adj["MISSING"] == ""          # departed players aren't listed as injured


def test_report_chart_from_backtest_table(tmp_path):
    from nbamodel import report

    md = tmp_path / "backtest.md"
    md.write_text("| Season | Model | Games | Team points MAE | Margin MAE | Total MAE | Winner % |\n"
                  "| --- | --- | --- | --- | --- | --- | --- |\n"
                  "| 2024-25 | Ridge (B) | 1309 | 9.23 | 10.92 | 14.72 | 67.7% |\n"
                  "| 2024-25 | Formula (A) | 1309 | 9.3 | 11.12 | 14.83 | 66.5% |\n")
    bt = report.read_backtest(md)
    assert bt.loc[0, "margin_mae"] == 10.92 and bt.loc[1, "winner_pct"] == 66.5
    fitted = {"coefficients": {k: 0.5 for k in features.FEATURES}, "seasons": ["2023-24"]}
    out = report.make_chart(bt, fitted, tmp_path / "chart.png")
    assert out.stat().st_size > 10_000


def test_healthy_team_with_deep_bench_has_no_lineup_change():
    # Eight regulars plus six rarely used players who averaged 10 minutes in the few games
    # they played. Nobody is out, so tonight's lineup should match recent games.
    names = [f"Regular {c}" for c in "ABCDEFGH"] + [f"Fringe {c}" for c in "ABCDEF"]
    minutes = pd.DataFrame({"PLAYER_NAME": names, "RECENT_MIN": [29] * 8 + [1.3] * 6,
                            "TYPICAL_MIN": [30] * 8 + [10] * 6, "ON_ROSTER": True})
    impact = pd.DataFrame({"OBPM": [3] * 8 + [-3] * 6, "DBPM": [1] * 8 + [-1] * 6},
                          index=[normalize_name(n) for n in names])
    adj = lineup.lineup_adjustment(minutes, {}, impact)
    assert abs(adj["ORTG_DELTA"]) < 0.5 and abs(adj["DRTG_DELTA"]) < 0.5


def test_parse_real_report_layout():
    # Lines as pdfplumber reads the 2026-10-20 6 PM report: reasons can wrap onto their
    # own line, and the team name only appears on a team's first row.
    names = {"Philadelphia 76ers": "PHI", "Oklahoma City Thunder": "OKC", "San Antonio Spurs": "SAS"}
    text = "\n".join([
        "07:00 (ET) PHI@NYK Philadelphia 76ers Barlow, Dominick Out",
        "Reaction",
        "Bona, Adem Questionable Injury/Illness - Left Foot Sprain",
        "Injury/Illness - Right Achilles Tendon",
        "Hukporti, Ariel Out",
        "Rupture/Tear",
        "09:30 (ET) OKC@SAS Oklahoma City Thunder Oweh, Otega Out",
        "San Antonio Spurs Harris, Tobias Out Injury/Illness - Left Calf Strain",
    ])
    df = injuries.parse_report_text(text, names)
    assert df[["TEAM", "PLAYER", "STATUS"]].values.tolist() == [
        ["PHI", "Barlow, Dominick", "out"],
        ["PHI", "Bona, Adem", "questionable"],
        ["PHI", "Hukporti, Ariel", "out"],
        ["OKC", "Oweh, Otega", "out"],
        ["SAS", "Harris, Tobias", "out"],
    ]


def test_playoff_games_left_out_of_ratings(monkeypatch):
    from nbamodel import config

    monkeypatch.setattr(config, "PLAYOFF_GAMES", "exclude")
    team_games, _ = _fake_league(seasons=("2024-25",))
    playoffs = team_games[team_games["GAME_ID"] <= 12].copy()
    playoffs["GAME_ID"] = playoffs["GAME_ID"] + 42400000          # 004... ids
    playoffs["GAME_DATE"] = playoffs["GAME_DATE"] + pd.Timedelta(days=400)
    playoffs["PTS"] = 60                                            # absurdly low scoring
    team_games = team_games.assign(GAME_ID=team_games["GAME_ID"] + 22400000)
    games = ratings.team_game_table(pd.concat([team_games, playoffs], ignore_index=True))
    day = playoffs["GAME_DATE"].max() + pd.Timedelta(days=1)
    league = ratings.league_average(games, day, "2024-25")
    assert league["ORTG"] > 100
    assert ratings.team_rating(games, 1, day, "2024-25", league)["ORTG"] > 100


def test_playoff_relative_matches_regular_season_level(monkeypatch):
    from nbamodel import config

    team_games, _ = _fake_league(seasons=("2024-25",))
    playoffs = team_games[team_games["GAME_ID"] <= 12].copy()
    playoffs["GAME_ID"] = playoffs["GAME_ID"] + 42400000
    playoffs["GAME_DATE"] = playoffs["GAME_DATE"] + pd.Timedelta(days=400)
    playoffs["PTS"] = playoffs["PTS"] - 10
    team_games = team_games.assign(GAME_ID=team_games["GAME_ID"] + 22400000)
    monkeypatch.setattr(config, "PLAYOFF_GAMES", "relative")
    games = ratings.team_game_table(pd.concat([team_games, playoffs], ignore_index=True))
    regular, playoff = games[~games["PLAYOFF"]], games[games["PLAYOFF"]]
    assert playoff["ORTG"].mean() == pytest.approx(regular["ORTG"].mean(), abs=1.5)
    # Teams keep their order within the playoffs: the strongest still rates highest.
    by_team = playoff.groupby("TEAM_ID")["ORTG"].mean()
    assert by_team.idxmax() == 4


def test_backtest_comparison_runs(monkeypatch, tmp_path):
    from nbamodel import cli, config

    team_games, player_games = _fake_league(seasons=("2023-24", "2024-25", "2025-26"))
    games = ratings.team_game_table(team_games)
    empty = pd.DataFrame(columns=["OBPM", "DBPM"])
    feats = features.historical_features(games, player_games, {s: empty for s in ("2024-25", "2025-26")},
                                         ["2024-25", "2025-26"])
    monkeypatch.setattr(config, "TRAIN_SEASONS", ["2024-25", "2025-26"])
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_historical", lambda seasons: feats)
    cli.main(["backtest", "--playoffs", "all"])
    table = (tmp_path / "results" / "playoff_comparison.md").read_text()
    assert "Ridge (B), first 15 games, playoffs relative" in table
    assert table.count("| 2025-26 |") == 9
