"""Model settings. Every number here is a modeling choice you can tune."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PREDICTIONS_DIR = ROOT / "predictions"
MODEL_PATH = DATA_DIR / "model.json"
MANUAL_STATUS_PATH = ROOT / "manual_status.csv"

# Seasons used to train the regression (nba_api season strings).
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# Team ratings: exponentially weighted over past games.
RATING_HALF_LIFE_GAMES = 24        # a game 24 games ago counts half as much as the last one (12 before decision 10)
PRIOR_SEASON_GAME_WEIGHT = 0.4     # last season's games count less once a new season starts
# How playoff games feed team ratings (they are slower and lower scoring than the regular season):
#   "include"  - use them as they are
#   "exclude"  - leave them out
#   "relative" - use them, measured against the playoff league average instead of the regular season's
PLAYOFF_GAMES = "include"   # original behavior; change once the comparison backtest picks a winner
SHRINK_GAMES = 4                   # pseudo-games of league average added to every rating
# Decision 12 tests three changes to how ratings are built. Each is off (the values below) until
# the test settles it (docs/decisions.md):
# - Season-start prior: share of a team's edge over the league average that last season's games
#   keep once a new season starts (1 = all of it). Below 1, last season is pulled toward average
#   first: regression to the mean.
SEASON_CARRYOVER = 1.0
# - Trend: share of the recent trend a rating follows (0 = a flat recency-weighted average). Above
#   0, the rating moves toward where a weighted straight line through the same games is heading.
RATING_TREND = 0.0
# - Change detection: a CUSUM on each team's net rating (points per 100 possessions) against its
#   rating going in. Residuals beyond the slack add up; past the threshold, a real change is
#   flagged and games before it count CHANGE_GAME_WEIGHT as much. None = off.
# Decision 12 kept the first two off and turned change detection on at 80 (it beat the noise on
# total error).
CUSUM_THRESHOLD = 80
CUSUM_SLACK = 4.0
CHANGE_GAME_WEIGHT = 0.4
# Strength of schedule: correct each game for how good the opponent was going into it.
# On since decision 8 (docs/decisions.md): it beat the noise on margin error.
OPPONENT_ADJUST = True

# Chance a player plays, by official injury report status.
PLAY_PROBABILITY = {
    "out": 0.0,
    "doubtful": 0.25,
    "questionable": 0.5,
    "probable": 0.9,
    "available": 1.0,
}

# Player impact (Box Plus/Minus) for players with no BPM on record: -3 in all since decision 10
# (it was -1.5 / -0.5).
REPLACEMENT_OBPM = -2.25
REPLACEMENT_DBPM = -0.75
# Minutes of current-season play at which this season's BPM counts as much as last season's.
# Half of it also pulls players with few minutes last season toward replacement level
# (250 minutes last season: half his own BPM, half replacement).
BPM_PRIOR_MINUTES = 500

# Recent minutes: exponentially weighted over a player's last games for his team.
MINUTES_HALF_LIFE_GAMES = 12       # 5 before decision 10
# Players who usually play at least this many minutes when available count as rotation players.
ROTATION_MINUTES = 20

# `predict --push` (which run_daily uses) commits the predictions folder to this branch of
# this git remote, on its own, apart from the code and whatever branch is checked out.
PREDICTIONS_REMOTE = "dev"
PREDICTIONS_BRANCH = "predictions"

# Bump when feature calculations change, so cached features are rebuilt.
FEATURE_VERSION = 6

# Four factors (shooting, turnovers, offensive rebounding, free throws) as extra regression
# inputs next to the pace x efficiency formula. The inputs are always built; this only says
# whether the regression uses them. Off until `evaluate --compare factors` settles it
# (docs/decisions.md, decision 9).
FOUR_FACTORS = False

# Rest days are capped: 3+ days off is treated the same as 3.
MAX_REST_DAYS = 3

# Ridge regression strength. Every alpha from 0.01 to 100 scored the same on validation
# (results/evaluation.md); the selection rule breaks ties toward the simpler model, which is 10.
RIDGE_ALPHA = 10.0

# Evaluation (python -m nbamodel evaluate). Settings are compared on the validation seasons
# only, refitting before each month and predicting that month. The test season stays locked:
# it is scored only by `evaluate --final`, and every such run is logged in results/test_log.md.
VALIDATION_SEASONS = ["2023-24", "2024-25"]
TEST_SEASON = "2025-26"
# Fixed before looking at results: settings are ranked by margin MAE, then total MAE,
# then how far the average total sits from the actual (closest to zero wins).
SELECTION_METRICS = ["margin_mae", "total_mae", "abs_total_bias"]
# On/off comparisons (`evaluate --compare`) use a noise test, fixed before decision 8's results:
# a change counts only if it moves average error by more than this many standard errors,
# measured month to month (docs/decisions.md, decision 8).
NOISE_STANDARD_ERRORS = 2
# From decision 9 on: a change that beats the noise is still turned down if it moves the
# average total more than this many points further from the actual (decision 8's lesson).
BIAS_GUARD_POINTS = 0.5
# Ridge strengths tried for the complexity curve, and training sizes (team-games) for the learning curve.
ALPHA_GRID = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
LEARNING_CURVE_SIZES = [100, 250, 500, 1000, 2000, 4000]
