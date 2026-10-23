"""Chart of backtest accuracy and model coefficients, for the README."""

import re
import textwrap

import pandas as pd

from . import config, model

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

MULTIPLIERS = {
    "BASE_PTS": "Pace x efficiency formula",
    "LINEUP_OFF_PTS": "Own lineup changes",
    "LINEUP_DEF_PTS": "Opponent's lineup changes",
}
POINT_EFFECTS = {
    "IS_HOME": "Playing at home",
    "OPP_B2B": "Opponent on a back-to-back",
    "REST": "Per day of rest",
    "OPP_REST": "Per opponent rest day",
    "B2B": "On a back-to-back",
}


def read_backtest(path=None) -> pd.DataFrame:
    """Parse the markdown table that `backtest` writes."""
    path = path or config.ROOT / "results" / "backtest.md"
    rows = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) in (7, 8) and re.match(r"\d{4}-\d{2}", cells[0]):
            rows.append({
                "season": cells[0], "model": cells[1], "games": int(cells[2]),
                "team_mae": float(cells[3]), "margin_mae": float(cells[4]),
                "total_mae": float(cells[5]), "winner_pct": float(cells[6].rstrip("%")),
            })
    return pd.DataFrame(rows)


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _season_dots(ax, bt, col, title, fmt):
    _style(ax, title)
    seasons = sorted(bt["season"].unique())
    x = range(len(seasons))
    for name, color, offset in (("Ridge (B)", BLUE, -0.08), ("Formula (A)", ORANGE, 0.08)):
        vals = bt[bt["model"] == name].set_index("season").reindex(seasons)[col]
        ax.plot([i + offset for i in x], vals, "o", color=color, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name.replace("Ridge", "Regression"))
        for i, v in zip(x, vals):
            ax.annotate(fmt(v), (i + offset, v), textcoords="offset points",
                        xytext=(-8 if offset < 0 else 8, 0), ha="right" if offset < 0 else "left",
                        va="center", fontsize=9, color=MUTED)
    ax.set_xticks(list(x), seasons)
    ax.set_xlim(-0.6, len(seasons) - 0.4)
    lo, hi = bt[col].min(), bt[col].max()
    pad = max((hi - lo) * 0.6, 0.3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left", ncol=2)


def _bars(ax, names, values, title, xlabel, reference=None):
    """Horizontal bars from zero. With a reference, all bars share one color and a dashed
    line marks the reference; without, positive and negative bars get the two colors."""
    _style(ax, title)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = list(range(len(names)))[::-1]
    colors = [BLUE] * len(values) if reference is not None else [BLUE if v >= 0 else ORANGE for v in values]
    ax.barh(y, values, color=colors, height=0.55)
    ax.axvline(0, color=MUTED, linewidth=1)
    if reference is not None:
        ax.axvline(reference, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("1.0 = taken at face value", (reference, max(y) + 0.55), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=MUTED)
    for yi, v in zip(y, values):
        label = f"{v:.2f}" if reference is not None else f"{v:+.2f}"
        ax.annotate(label, (v, yi), textcoords="offset points", xytext=(6 if v >= 0 else -6, 0),
                    ha="left" if v >= 0 else "right", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    lo, hi = min(min(values), 0), max(max(values), reference or 0)
    span = hi - lo
    ax.set_xlim(lo - (span * 0.2 if lo < 0 else 0), hi + span * 0.18)
    ax.set_ylim(min(y) - 0.6, max(y) + 0.9)


def make_chart(bt: pd.DataFrame, fitted: dict, out=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or config.ROOT / "results" / "backtest.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55, "wspace": 0.45})
    _season_dots(axes[0, 0], bt, "margin_mae", "Average miss on the final margin (points, lower is better)",
                 lambda v: f"{v:.1f}")
    _season_dots(axes[0, 1], bt, "winner_pct", "Games where the predicted winner won (%)",
                 lambda v: f"{v:.1f}%")

    coefs = fitted["coefficients"]
    _bars(axes[1, 0], list(MULTIPLIERS.values()), [coefs[k] for k in MULTIPLIERS],
          "How much the model trusts each input", "Weight given by the regression", 1.0)
    _bars(axes[1, 1], list(POINT_EFFECTS.values()), [coefs[k] for k in POINT_EFFECTS],
          "Effect on a team's points", "Points added to the team's score")

    ridge = bt[bt["model"] == "Ridge (B)"]
    fig.text(0.04, 0.965, "NBA score model: backtest", ha="left", va="top",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.04, 0.925, textwrap.fill(
             f"Each season predicted by a model trained only on earlier seasons ({ridge['games'].sum():,} games). "
             f"Regression averages {ridge['team_mae'].mean():.1f} points off per team score "
             f"and {ridge['total_mae'].mean():.1f} off the game total. "
             f"Coefficients from the model trained on {', '.join(fitted['seasons'])}.", 150),
             ha="left", va="top", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.81, bottom=0.08)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def run():
    return make_chart(read_backtest(), model.load())
