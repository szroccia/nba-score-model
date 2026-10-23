"""Ridge regression from features to a team's points, saved as readable JSON."""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config
from .features import FEATURES

# Early-season ratings are mostly last season's; skip a team's first games when training.
MIN_GAMES_FOR_TRAINING = 10


def fit(train: pd.DataFrame, alpha: float | None = None) -> dict:
    rows = train[train["GAMES_PLAYED"] >= MIN_GAMES_FOR_TRAINING]
    alpha = config.RIDGE_ALPHA if alpha is None else alpha
    ridge = Ridge(alpha=alpha).fit(rows[FEATURES], rows["PTS"])
    return {
        "intercept": float(ridge.intercept_),
        "coefficients": dict(zip(FEATURES, map(float, ridge.coef_))),
        "alpha": alpha,
        "trained_on_rows": int(len(rows)),
        "seasons": sorted(rows["SEASON"].unique().tolist()),
    }


def predict(model: dict, X: pd.DataFrame) -> np.ndarray:
    coef = np.array([model["coefficients"][f] for f in FEATURES])
    return model["intercept"] + X[FEATURES].to_numpy() @ coef


# A prediction broken into the steps that add up to it, in plain words, in order.
STEPS = ["Pace x efficiency formula", "Regression calibration", "Own lineup tonight",
         "Opponent's lineup tonight", "Home court", "Rest and back-to-backs"]


def contributions(model: dict, row: dict) -> dict:
    """Points each step adds for one team. The steps sum to that team's predicted points.

    The formula enters at face value; "regression calibration" is the intercept plus the
    small correction from its coefficient not being exactly 1."""
    c = model["coefficients"]
    return {
        STEPS[0]: row["BASE_PTS"],
        STEPS[1]: model["intercept"] + (c["BASE_PTS"] - 1) * row["BASE_PTS"],
        STEPS[2]: c["LINEUP_OFF_PTS"] * row["LINEUP_OFF_PTS"],
        STEPS[3]: c["LINEUP_DEF_PTS"] * row["LINEUP_DEF_PTS"],
        STEPS[4]: c["IS_HOME"] * row["IS_HOME"],
        STEPS[5]: sum(c[f] * row[f] for f in ("REST", "OPP_REST", "B2B", "OPP_B2B")),
    }


def save(model: dict, path=config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load(path=config.MODEL_PATH) -> dict:
    return json.loads(path.read_text())


def score(df: pd.DataFrame, pred_col: str = "PRED_PTS") -> dict:
    """Average errors in points: per team score, margin and game total."""
    games = df.pivot_table(index="GAME_ID", columns="IS_HOME", values=["PTS", pred_col])
    actual_margin = games["PTS"][1] - games["PTS"][0]
    pred_margin = games[pred_col][1] - games[pred_col][0]
    actual_total = games["PTS"][1] + games["PTS"][0]
    pred_total = games[pred_col][1] + games[pred_col][0]
    return {
        "games": int(len(games)),
        "team_points_mae": round(float((df["PTS"] - df[pred_col]).abs().mean()), 2),
        "margin_mae": round(float((actual_margin - pred_margin).abs().mean()), 2),
        "total_mae": round(float((actual_total - pred_total).abs().mean()), 2),
        "winner_pct": round(float((actual_margin * pred_margin > 0).mean()), 3),
        # Average of predicted minus actual total: near 0 means no systematic lean high or low.
        "total_bias": round(float((pred_total - actual_total).mean()), 2),
    }
