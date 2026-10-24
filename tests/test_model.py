import numpy as np
import pandas as pd
import pytest

from nbamodel import features, injuries, lineup, model, ratings
from nbamodel.names import normalize_name


def test_normalize_name_matches_across_sources():
    assert normalize_name("Dončić, Luka") == normalize_name("Luka Doncic") == "luka doncic"
    assert normalize_name("Jackson Jr., Jaren") == normalize_name("Jaren Jackson Jr.")
    assert normalize_name("Gilgeous-Alexander, Shai") == normalize_name("Shai Gilgeous-Alexander")
    assert normalize_name("Ron Holland") == normalize_name("Holland II, Ronald") == "ronald holland"


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
        ["LAC", "", injuries.NOT_SUBMITTED],
    ]
    assert (df["GAME_DATE"] == pd.Timestamp("2026-10-21")).all()
    probs = injuries.play_probabilities(df)
    assert probs["BOS"]["jayson tatum"] == 0.0
    assert probs["LAL"]["lebron james"] == 0.9
    assert "LAC" not in probs


def test_injury_report_uses_only_the_day_predicted():
    """A report covers two days. A team on a back-to-back must get tonight's status, not tomorrow's."""
    names = {"Boston Celtics": "BOS", "Detroit Pistons": "DET", "Los Angeles Lakers": "LAL"}
    text = "\n".join([
        "Injury Report: 10/20/26 06:00 PM",
        "10/20/2026 07:00 (ET) BOS@DET Boston Celtics Brown, Jaylen Out Injury/Illness - Left Ankle Sprain",
        "10/21/2026 07:30 (ET) LAL@BOS Boston Celtics Brown, Jaylen Questionable Injury/Illness - Left Ankle Sprain",
        "Los Angeles Lakers NOT YET SUBMITTED",
    ])
    df = injuries.parse_report_text(text, names)
    tonight = injuries.play_probabilities(injuries.for_date(df, pd.Timestamp("2026-10-20")))
    tomorrow = injuries.for_date(df, pd.Timestamp("2026-10-21"))
    assert tonight["BOS"]["jaylen brown"] == 0.0
    assert injuries.play_probabilities(tomorrow)["BOS"]["jaylen brown"] == 0.5
    assert tomorrow.loc[tomorrow["TEAM"] == "LAL", "STATUS"].tolist() == [injuries.NOT_SUBMITTED]
    # Manual overrides have no date and always apply.
    manual = pd.DataFrame({"TEAM": ["BOS"], "PLAYER": ["Jaylen Brown"], "STATUS": ["available"]})
    both = injuries.for_date(pd.concat([df, manual], ignore_index=True), pd.Timestamp("2026-10-20"))
    assert injuries.play_probabilities(both)["BOS"]["jaylen brown"] == 1.0


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


def test_opponent_adjustment_credits_tough_schedule(monkeypatch):
    from nbamodel import config

    monkeypatch.setattr(config, "OPPONENT_ADJUST", True)
    # Teams 1 and 2 score 112 every night. Team 1 always plays team 4, whose defense also
    # gives up 125 to team 5; team 2 always plays team 5, which holds team 4 to 100.
    # Team 2 scored the same against a much better defense, so it should rate higher.
    rows = []
    day = pd.Timestamp("2024-10-20")
    gid = 0
    for rnd in range(40):
        day += pd.Timedelta(days=1)
        for team, opp, pts, opp_pts in ((1, 4, 112, 112), (2, 5, 112, 112), (4, 5, 100, 125)):
            gid += 1
            for t, o, p, home in ((team, opp, pts, True), (opp, team, opp_pts, False)):
                rows.append({"TEAM_ID": t, "TEAM_ABBREVIATION": f"T{t}", "GAME_ID": gid, "GAME_DATE": day,
                             "MATCHUP": f"T{t} vs. T{o}" if home else f"T{t} @ T{o}", "MIN": 240,
                             "FGA": 88, "FTA": 22, "OREB": 10, "TOV": 14, "PTS": p, "SEASON": "2024-25"})
    games = ratings.opponent_adjust(ratings.team_game_table(pd.DataFrame(rows)))
    end = day + pd.Timedelta(days=1)
    league = ratings.league_average(games, end, "2024-25")
    a = ratings.team_rating(games, 1, end, "2024-25", league)
    b = ratings.team_rating(games, 2, end, "2024-25", league)
    assert b["ORTG"] > a["ORTG"] + 3

    # Opening night of the next season: team 4's leaky defense is known from last season,
    # so 112 against it counts for less than 112 raw.
    opener = pd.DataFrame([{**rows[0], "GAME_ID": 999, "GAME_DATE": pd.Timestamp("2025-10-21"), "SEASON": "2025-26",
                            "PTS": 112, "TEAM_ID": t, "TEAM_ABBREVIATION": f"T{t}",
                            "MATCHUP": f"T{t} vs. T{o}" if t == 1 else f"T{t} @ T{o}"} for t, o in ((1, 4), (4, 1))])
    both = pd.concat([ratings.team_game_table(pd.DataFrame(rows)), ratings.team_game_table(opener)], ignore_index=True)
    first = ratings.opponent_adjust(both).query("GAME_ID == 999 and TEAM_ID == 1").iloc[0]
    assert first["ORTG_ADJ"] < first["ORTG"] - 1


def test_opponent_adjustment_ignores_league_wide_scoring_changes():
    """Equal teams, so there is no schedule to correct for, while league scoring jumps six
    points between seasons. The adjustment must leave the scoring level alone."""
    rng = np.random.default_rng(0)
    rows, gid = [], 0
    for season, level in (("2023-24", 110.0), ("2024-25", 116.0)):
        for d in range(50):
            day = pd.Timestamp(f"{season[:4]}-10-20") + pd.Timedelta(days=d)
            teams = rng.permutation(10)
            for a, b in zip(teams[::2], teams[1::2]):
                gid += 1
                pa, pb = level + rng.normal(0, 11), level + rng.normal(0, 11)
                for t, o, f, ag in ((a, b, pa, pb), (b, a, pb, pa)):
                    rows.append({"GAME_ID": gid, "GAME_DATE": day, "SEASON": season, "TEAM_ID": t,
                                 "OPP_TEAM_ID": o, "PLAYOFF": False, "PACE": 99.0, "ORTG": f, "DRTG": ag})
    adj = ratings.opponent_adjust(pd.DataFrame(rows)).query("SEASON == '2024-25'")
    raw_level = adj["ORTG"].mean() + adj["DRTG"].mean()
    assert abs(adj["ORTG_ADJ"].mean() + adj["DRTG_ADJ"].mean() - raw_level) < 0.5


def test_validate_flags_bad_box_scores(monkeypatch):
    from nbamodel import data, validate

    def row(gid, team, pts, fgm, fg3m, ftm):
        return {"GAME_ID": gid, "TEAM_ID": team, "TEAM_ABBREVIATION": f"T{team}", "GAME_DATE": "2025-01-01",
                "PTS": pts, "FGM": fgm, "FG3M": fg3m, "FTM": ftm}
    teams = pd.DataFrame([row(22400001, 1, 110, 40, 12, 18), row(22400001, 2, 105, 40, 10, 15),
                          row(22400002, 1, 99, 40, 12, 18), row(22400002, 3, 100, 38, 9, 15)])
    players = teams[["GAME_ID", "TEAM_ID", "PTS"]].copy()
    players.loc[0, "PTS"] = 100          # player totals disagree with the team total once
    monkeypatch.setattr(data, "team_games", lambda s, refresh=False: teams)
    monkeypatch.setattr(data, "player_games", lambda s, refresh=False: players)
    results = {name: passed for name, passed, _ in validate.check_season("2024-25", complete=False)}
    assert results["two teams in every game"]
    assert not results["points = 2 x FGM + 3PM + FTM"]          # 99 != 2*40+12+18
    assert not results["team points = sum of player points"]
    assert not results["30 teams"]
    links = validate.spot_check_games("2024-25", n=2, seed=0)["box score"]
    assert set(links) == {"https://www.nba.com/game/0022400001", "https://www.nba.com/game/0022400002"}


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
    assert report.coefficients(fitted, tmp_path / "coef.png").stat().st_size > 10_000
    assert report.model_vs_formula(bt, tmp_path / "formula.png").stat().st_size > 10_000


def test_contributions_add_up_to_the_prediction():
    fitted = {"intercept": 1.5, "coefficients": dict(zip(features.FEATURES, [0.98, 0.6, 1.2, 2.1, 0.1, -0.1, -0.8, 1.2]))}
    row = {"BASE_PTS": 113.0, "LINEUP_OFF_PTS": -2.0, "LINEUP_DEF_PTS": 1.0, "IS_HOME": 1,
           "REST": 2, "OPP_REST": 0, "B2B": 0, "OPP_B2B": 1}
    parts = model.contributions(fitted, row)
    assert list(parts) == model.STEPS
    assert sum(parts.values()) == pytest.approx(model.predict(fitted, pd.DataFrame([row]))[0])
    assert parts["Home court"] == pytest.approx(2.1)


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
    monkeypatch.setattr(config, "TEST_SEASON", "2026-27")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_historical", lambda seasons: feats)
    cli.main(["backtest", "--playoffs", "all"])
    table = (tmp_path / "results" / "playoff_comparison.md").read_text()
    assert "Ridge (B), first 15 games, playoffs relative" in table
    assert table.count("| 2025-26 |") == 9
    # The locked test season never shows up in a backtest.
    monkeypatch.setattr(config, "TEST_SEASON", "2025-26")
    cli.main(["backtest"])
    assert "| 2025-26 |" not in (tmp_path / "results" / "backtest.md").read_text()


def _fake_features(seasons=("2022-23", "2023-24", "2024-25", "2025-26"), n_teams=6, seed=1):
    """Feature rows like historical_features makes, where points = BASE_PTS + home edge + noise."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for season in seasons:
        start = pd.Timestamp(f"{season[:4]}-10-22")
        for day in range(170):
            date = start + pd.Timedelta(days=day)
            teams = rng.permutation(n_teams)
            for h, a in zip(teams[::2], teams[1::2]):
                gid += 1
                for team, opp, home in ((h, a, 1), (a, h, 0)):
                    base = 112 + rng.normal(0, 4)
                    rows.append({"GAME_ID": gid, "GAME_DATE": date, "SEASON": season, "TEAM_ID": int(team) + 1,
                                 "OPP_TEAM_ID": int(opp) + 1, "GAMES_PLAYED": 50, "SEASON_GAMES": day,
                                 "BASE_PTS": base, "LINEUP_OFF_PTS": rng.normal(0, 1), "LINEUP_DEF_PTS": 0.0,
                                 "IS_HOME": home, "REST": 1, "OPP_REST": 1, "B2B": 0, "OPP_B2B": 0,
                                 "PTS": round(base + (1.5 if home else -1.5) + rng.normal(0, 9))})
    return pd.DataFrame(rows)


def test_folds_only_train_on_the_past_and_skip_the_test_season(monkeypatch):
    from nbamodel import config, evaluate

    monkeypatch.setattr(config, "TEST_SEASON", "2025-26")
    feats = evaluate.validation_rows(_fake_features())
    assert "2025-26" not in set(feats["SEASON"])
    folds = list(evaluate.monthly_folds(feats, ["2023-24", "2024-25"]))
    assert len(folds) == 14  # October through April, two seasons
    for _, _, train, test in folds:
        assert train["GAME_DATE"].max() < test["GAME_DATE"].min()
        assert test["GAME_DATE"].dt.to_period("M").nunique() == 1


def test_vegas_line_parsing():
    from nbamodel import vegas

    assert vegas.home_line("NY -5.5", "PHI") == -5.5
    assert vegas.home_line("PHI -3", "PHI") == 3.0
    assert vegas.home_line("EVEN", "PHI") == 0.0
    items = [
        {"provider": {"name": "Caesars Sportsbook (New Jersey) - Live Odds"}, "details": "PHI -9", "overUnder": 250},
        {"provider": {"name": "DraftKings"}, "details": "NY -6", "overUnder": 220},
        {"provider": {"name": "consensus"}, "details": "NY -5.5", "overUnder": 219.5},
    ]
    assert vegas.pick_line(items, "PHI") == {"VEGAS_HOME_MARGIN": -5.5, "VEGAS_TOTAL": 219.5, "BOOK": "consensus"}
    assert vegas.pick_line(items[:1], "PHI") is None
    assert vegas.nba_abbreviation("GS") == "GSW" and vegas.nba_abbreviation("BOS") == "BOS"


def test_rank_prefers_simpler_model_on_a_tie():
    from nbamodel import evaluate

    table = pd.DataFrame({"alpha": [1, 100, 10], "margin_mae": [11.001, 11.004, 11.2],
                          "total_mae": [15, 15, 14], "abs_total_bias": [0.1, 0.1, 0.0]})
    assert evaluate.rank(table)["alpha"].tolist() == [100, 1, 10]


def test_evaluation_report_and_locked_test_log(monkeypatch, tmp_path):
    from nbamodel import config, evaluate, report

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "VALIDATION_SEASONS", ["2023-24", "2024-25"])
    monkeypatch.setattr(config, "TEST_SEASON", "2025-26")
    monkeypatch.setattr(config, "ALPHA_GRID", [0.1, 10, 1000])
    monkeypatch.setattr(config, "LEARNING_CURVE_SIZES", [50, 200, 800])
    feats = _fake_features()
    abbr = {i: f"T{i}" for i in range(1, 7)}
    # Vegas lines: the true expected margin and total, so Vegas should beat a fitted model slightly.
    games = evaluate.game_frame(feats, feats["BASE_PTS"] + 1.5 * (2 * feats["IS_HOME"] - 1), abbr)
    lines = games.rename(columns={"PRED_MARGIN": "VEGAS_HOME_MARGIN", "PRED_TOTAL": "VEGAS_TOTAL"})

    text = "\n".join(evaluate.run(feats, lines, abbr))
    assert "## Against the Vegas closing line" in text and "## Learning curve" in text
    assert "| 2025-1" not in text and "| 2026-" not in text  # no test-season months
    tables = report.read_tables(tmp_path / "results" / "evaluation.md")
    assert {"Current settings", "Against the Vegas closing line", "Learning curve"} <= set(tables)
    assert tables["Complexity curve (ridge alpha)"]["Alpha"].tolist() == [0.1, 10, 1000]
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.json")
    made = {p.name for p in report.run(tmp_path / "figures")}
    assert made == {"evaluation_scheme.png", "vegas_by_month.png", "ridge_alpha.png", "learning_curve.png"}
    curve = evaluate.complexity_curve(evaluate.validation_rows(feats), ["2024-25"], [0.1, 1e6])
    assert curve.loc[0, "margin_mae"] < curve.loc[1, "margin_mae"]  # crushing the coefficients hurts

    evaluate.final(feats, lines, abbr)
    evaluate.final(feats, lines, abbr)
    log = (tmp_path / "results" / "test_log.md").read_text()
    assert log.count("# Test season log") == 1 and log.count("## ") == 2
    assert "Vegas closing line" in log


def test_evaluate_command_compares_playoff_settings(monkeypatch, tmp_path):
    from nbamodel import cli, config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "PLAYOFF_GAMES", "include")
    monkeypatch.setattr(config, "VALIDATION_SEASONS", ["2023-24", "2024-25"])
    monkeypatch.setattr(config, "TEST_SEASON", "2025-26")
    monkeypatch.setattr(config, "ALPHA_GRID", [1, 100])
    monkeypatch.setattr(config, "LEARNING_CURVE_SIZES", [100, 400])
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.json")
    feats = _fake_features()
    monkeypatch.setattr(cli, "_historical", lambda seasons: feats)
    monkeypatch.setattr(cli, "_vegas_lines", lambda: None)
    cli.main(["evaluate", "--playoffs", "all"])
    assert (tmp_path / "docs" / "figures" / "playoff_settings.png").exists()
    report = (tmp_path / "results" / "evaluation.md").read_text()
    assert report.count("| playoffs ") == 3
    assert config.PLAYOFF_GAMES == "include"  # restored after the comparison


def test_explain_draws_one_game(monkeypatch, tmp_path):
    from nbamodel import cli, config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    fitted = {"intercept": 0.2, "coefficients": dict(zip(features.FEATURES, [0.99, 0.6, 1.2, 2.1, 0.1, 0, -0.8, 1.2]))}
    rows = []
    for team, opp, home in (("BOS", "DET", 0), ("DET", "BOS", 1)):
        x = {"BASE_PTS": 112.0, "LINEUP_OFF_PTS": -1.0, "LINEUP_DEF_PTS": 0.5, "IS_HOME": home,
             "REST": 1, "OPP_REST": 1, "B2B": 0, "OPP_B2B": 0}
        rows.append({"date": "2026-10-22", "game_id": 7, "team": team, "opponent": opp, **x,
                     **model.contributions(fitted, x), "predicted_pts": model.predict(fitted, pd.DataFrame([x]))[0],
                     "missing": "Jayson Tatum (0%)" if team == "BOS" else ""})
    (tmp_path / "predictions").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "predictions" / "2026-10-22_details.csv", index=False)
    cli.main(["explain", "--date", "2026-10-22", "--game", "det"])
    assert (tmp_path / "predictions" / "2026-10-22_BOS-at-DET.png").stat().st_size > 10_000
    with pytest.raises(SystemExit):
        cli.main(["explain", "--date", "2026-10-22", "--game", "LAL"])


def test_bpm_before_it_is_published(monkeypatch, tmp_path):
    """Early in a season Basketball-Reference lists players without BPM columns."""
    import io

    from nbamodel import config, data

    html = ("<table id='advanced'><thead><tr><th>Rk</th><th>Player</th><th>MP</th><th>PER</th></tr></thead>"
            "<tbody><tr><td>1</td><td>Paolo Banchero</td><td>41</td><td>29.5</td></tr></tbody></table>")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(html.encode()))
    df = data.bpm("2026-27", refresh=True)
    assert df.empty and list(df.columns) == ["Player", "MP", "OBPM", "DBPM", "BPM"]
    assert lineup.player_impact(data.bpm("2025-26", refresh=True), df).empty


def test_games_in_progress_are_left_out(monkeypatch, tmp_path):
    """A download during a game saves its partial score with no win or loss yet."""
    from nbamodel import config, data

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    rows = "GAME_ID,GAME_DATE,TEAM_ID,PTS,WL\n1,2026-10-21,10,128,W\n1,2026-10-21,20,116,L\n2,2026-10-23,30,34,\n2,2026-10-23,40,39,\n"
    (tmp_path / "team_games_2026-27.csv").write_text(rows)
    (tmp_path / "player_games_2026-27.csv").write_text(rows.replace("TEAM_ID", "PLAYER_ID"))
    assert set(data.team_games("2026-27")["GAME_ID"]) == {1}
    assert set(data.player_games("2026-27")["GAME_ID"]) == {1}


def test_warns_when_report_does_not_cover_a_team(capsys):
    from nbamodel import cli

    names = {"Boston Celtics": "BOS", "Los Angeles Lakers": "LAL"}
    df = injuries.parse_report_text("\n".join([
        "10/21/2026 07:30 (ET) LAL@BOS Boston Celtics Brown, Jaylen Out Injury/Illness - Left Ankle Sprain",
        "Los Angeles Lakers NOT YET SUBMITTED",
    ]), names)
    day = pd.Timestamp("2026-10-21")
    cli._warn_unreported(df, injuries.for_date(df, day), {"BOS", "LAL"}, day)
    assert "not yet submitted to the injury report: LAL" in capsys.readouterr().out
    other = pd.Timestamp("2026-10-23")
    cli._warn_unreported(df, injuries.for_date(df, other), {"BOS", "LAL"}, other)
    assert "covers 2026-10-21, not 2026-10-23" in capsys.readouterr().out


def test_saved_features_rebuild_when_a_setting_changes(monkeypatch, tmp_path):
    from nbamodel import cli, config

    builds = []

    def build(games, players, impact, seasons):
        builds.append(config.SHRINK_GAMES)
        return pd.DataFrame({"SEASON": seasons, "GAME_DATE": pd.Timestamp("2025-01-01")})

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_load", lambda seasons: (None, None))
    monkeypatch.setattr(cli.data, "bpm", lambda season: None)
    monkeypatch.setattr(cli.lineup, "player_impact", lambda bpm: None)
    monkeypatch.setattr(cli.features, "historical_features", build)
    cli._historical(["2024-25"])
    cli._historical(["2024-25"])          # same settings: reuses the saved file
    monkeypatch.setattr(config, "SHRINK_GAMES", 8)
    cli._historical(["2024-25"])          # a tuning change: rebuilt, not the stale file
    assert builds == [4, 8]


def test_push_predictions_uses_its_own_branch(monkeypatch, tmp_path):
    """Predictions go to their own branch on the remote; the checked-out branch is untouched."""
    import subprocess

    from nbamodel import cli, config

    def git(*cmd, cwd):
        return subprocess.run(["git", *cmd], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    for var, value in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"),
                       ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")):
        monkeypatch.setenv(var, value)
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    git("init", "-q", "--bare", str(remote), cwd=tmp_path)
    git("init", "-q", str(repo), cwd=tmp_path)
    (repo / "code.py").write_text("x = 1\n")
    git("add", "code.py", cwd=repo)
    git("commit", "-q", "-m", "code", cwd=repo)
    git("remote", "add", "dev", str(remote), cwd=repo)
    head = git("rev-parse", "HEAD", cwd=repo)
    monkeypatch.setattr(config, "ROOT", repo)
    monkeypatch.setattr(config, "PREDICTIONS_DIR", repo / "predictions")
    (repo / "predictions").mkdir()
    (repo / "predictions" / "2026-10-22.csv").write_text("a\n")
    cli.push_predictions("2026-10-22")
    (repo / "predictions" / "2026-10-23.csv").write_text("b\n")
    cli.push_predictions("2026-10-23")
    cli.push_predictions("2026-10-23")      # nothing new: no extra commit
    files = git("ls-tree", "-r", "--name-only", "predictions", cwd=remote).splitlines()
    assert files == ["predictions/2026-10-22.csv", "predictions/2026-10-23.csv"]
    assert git("rev-list", "--count", "predictions", cwd=remote) == "2"
    assert git("rev-parse", "HEAD", cwd=repo) == head          # the code branch didn't move
    assert git("status", "--porcelain", "--", "code.py", cwd=repo) == ""


def _games_with_errors(margin_err, total_err, folds):
    """A game frame like evaluate.game_frame where predictions miss by the given amounts."""
    n = len(margin_err)
    return pd.DataFrame({"GAME_ID": range(n), "FOLD": folds, "MARGIN": 0.0, "TOTAL": 220.0,
                         "PRED_MARGIN": margin_err, "PRED_TOTAL": 220.0 + np.asarray(total_err)})


def test_noise_test_needs_a_change_bigger_than_the_noise():
    from nbamodel import evaluate

    rng = np.random.default_rng(0)
    folds = np.repeat([f"2024-{m:02d}" for m in range(1, 11)], 100)
    base = _games_with_errors(rng.normal(0, 12, 1000), rng.normal(0, 18, 1000), folds)
    # Every game a point better on the margin: beats the noise.
    better = base.assign(PRED_MARGIN=base["PRED_MARGIN"] * (1 - 1 / base["PRED_MARGIN"].abs().clip(lower=1)))
    assert evaluate.noise_test(base, better)["verdict"] == "better"
    # The same predictions: a tie.
    assert evaluate.noise_test(base, base.copy())["verdict"] == "tie"
    # Two points better in half the months and two points worse in the rest averages out:
    # month-to-month blocks make that a tie, even though every single game changed.
    sign = np.where(pd.Series(folds).str[-2:].astype(int) % 2 == 0, -2.0, 2.0)
    mixed = base.assign(PRED_MARGIN=np.sign(base["PRED_MARGIN"]) * (base["PRED_MARGIN"].abs() + sign).clip(lower=0))
    change, se = evaluate.paired_change(base, mixed, "MARGIN")
    assert abs(change) < 2 * se
    # Worse on total error, with no margin change: worse.
    worse = base.assign(PRED_TOTAL=220.0 + np.sign(base["PRED_TOTAL"] - 220) * ((base["PRED_TOTAL"] - 220).abs() + 3))
    assert evaluate.noise_test(base, worse)["verdict"] == "worse"


def test_evaluate_compares_strength_of_schedule(monkeypatch, tmp_path):
    from nbamodel import cli, config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "OPPONENT_ADJUST", False)
    monkeypatch.setattr(config, "VALIDATION_SEASONS", ["2023-24", "2024-25"])
    monkeypatch.setattr(config, "TEST_SEASON", "2025-26")
    monkeypatch.setattr(config, "ALPHA_GRID", [1, 100])
    monkeypatch.setattr(config, "LEARNING_CURVE_SIZES", [100, 400])
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.json")
    plain = _fake_features()
    # With the adjustment on, the formula sits closer to what actually happened.
    sharper = plain.assign(BASE_PTS=(plain["BASE_PTS"] + plain["PTS"] - 1.5 * (2 * plain["IS_HOME"] - 1)) / 2)
    monkeypatch.setattr(cli, "_historical", lambda seasons: sharper if config.OPPONENT_ADJUST else plain)
    monkeypatch.setattr(cli, "_vegas_lines", lambda: None)
    cli.main(["evaluate", "--compare", "sos"])
    tables = __import__("nbamodel.report", fromlist=["read_tables"]).read_tables(tmp_path / "results" / "evaluation.md")
    sos = tables["Strength of schedule"]
    assert sos["Setting"].tolist() == ["off", "on"]
    assert sos["Verdict"].tolist() == ["current", "better"]
    assert sos.loc[1, "Margin change"] < 0
    assert (tmp_path / "docs" / "figures" / "strength_of_schedule.png").exists()
    assert config.OPPONENT_ADJUST is False  # restored after the comparison
