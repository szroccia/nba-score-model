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
RATING_HALF_LIFE_GAMES = 12        # a game 12 games ago counts half as much as the last one
PRIOR_SEASON_GAME_WEIGHT = 0.4     # last season's games count less once a new season starts
# How playoff games feed team ratings (they are slower and lower scoring than the regular season):
#   "include"  - use them as they are
#   "exclude"  - leave them out
#   "relative" - use them, measured against the playoff league average instead of the regular season's
PLAYOFF_GAMES = "include"   # original behavior; change once the comparison backtest picks a winner
SHRINK_GAMES = 4                   # pseudo-games of league average added to every rating

# Chance a player plays, by official injury report status.
PLAY_PROBABILITY = {
    "out": 0.0,
    "doubtful": 0.25,
    "questionable": 0.5,
    "probable": 0.9,
    "available": 1.0,
}

# Player impact (Box Plus/Minus) for players with no BPM on record.
REPLACEMENT_OBPM = -1.5
REPLACEMENT_DBPM = -0.5
# Minutes of current-season play needed before current BPM outweighs last season's.
BPM_PRIOR_MINUTES = 500

# Recent minutes: exponentially weighted over a player's last games for his team.
MINUTES_HALF_LIFE_GAMES = 5
# Players who usually play at least this many minutes when available count as rotation players.
ROTATION_MINUTES = 20

# Bump when feature calculations change, so cached features are rebuilt.
FEATURE_VERSION = 4

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
# Ridge strengths tried for the complexity curve, and training sizes (team-games) for the learning curve.
ALPHA_GRID = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
LEARNING_CURVE_SIZES = [100, 250, 500, 1000, 2000, 4000]
