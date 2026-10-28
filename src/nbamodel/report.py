"""Charts that explain the model and the decisions behind it, for the README and docs/decisions.md.

Every chart is drawn from files already in the repo (results/*.md, data/model.json and the
settings in config.py), so `python -m nbamodel report` can redraw them all at any time.
"""

import re

import pandas as pd

from . import config, model

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2ac"
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
# Four-factor edges (only in a model trained with them): points per percentage point of edge,
# beyond what the pace x efficiency formula already counts.
FACTOR_EFFECTS = {
    "EFG_EDGE": "Per point of shooting edge",
    "TOV_EDGE": "Per point of turnover edge",
    "ORB_EDGE": "Per point of rebounding edge",
    "FT_EDGE": "Per point of free throw edge",
}


def _number(cell: str):
    """'66.8%' -> 66.8, '+1.64' -> 1.64, '2023-10' stays text."""
    text = cell.strip().rstrip("%")
    try:
        return float(text)
    except ValueError:
        return cell.strip()


def read_tables(path) -> dict[str, pd.DataFrame]:
    """Every markdown table in a file, keyed by the heading above it ('' before any heading)."""
    tables, heading, rows = {}, "", []

    def flush():
        if len(rows) > 2:
            header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
            body = [[_number(c) for c in r.strip().strip("|").split("|")] for r in rows[2:]]
            tables[heading] = pd.DataFrame(body, columns=header)
        rows.clear()

    for line in path.read_text().splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        elif line.strip().startswith("|"):
            rows.append(line)
        else:
            flush()
    flush()
    return tables


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


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure(ncols=1, width=7.0, height=4.2, **kw):
    plt = _plt()
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, **kw)
    return fig, axes


def _style(ax, title, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _headline(fig, title, subtitle):
    fig.text(0.02, 0.98, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED, wrap=True)


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    _plt().close(fig)
    return out


def _line(ax, x, y, color, label):
    ax.plot(x, y, "-o", color=color, linewidth=2, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label=label)


def _legend(ax, **kw):
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, **kw)


def _bars(ax, names, values, title, xlabel, reference=None):
    """Horizontal bars from zero. With a reference, all bars share one color and a dashed
    line marks the reference; without, positive and negative bars get the two colors."""
    _style(ax, title, grid_axis="x")
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


# --- One chart per point the README or decision log makes ---------------------------------

def coefficients(fitted: dict, out):
    """What the regression learned: how much it trusts each input, and the fixed point effects."""
    fig, axes = _figure(2, 12, 4.4, gridspec_kw={"wspace": 0.75})
    coefs = fitted["coefficients"]
    _bars(axes[0], list(MULTIPLIERS.values()), [coefs[k] for k in MULTIPLIERS],
          "How much the model trusts each input", "Weight given by the regression", 1.0)
    effects = {**POINT_EFFECTS, **{k: v for k, v in FACTOR_EFFECTS.items() if k in coefs}}
    _bars(axes[1], list(effects.values()), [coefs[k] for k in effects],
          "Effect on a team's points", "Points added to the team's score")
    _headline(fig, "What the model learned",
              f"Ridge regression coefficients, trained on {', '.join(fitted['seasons'])} "
              f"({fitted['trained_on_rows']:,} team-games)." if "trained_on_rows" in fitted else
              f"Ridge regression coefficients, trained on {', '.join(fitted['seasons'])}.")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.14)
    return _save(fig, out)


def model_vs_formula(bt: pd.DataFrame, out):
    """Decision 1: the regression against the plain possession formula, season by season."""
    fig, ax = _figure(1, 7, 4.2)
    _style(ax, "")
    seasons = sorted(bt["season"].unique())
    for name, color, label in (("Ridge (B)", BLUE, "Ratings + ridge regression (chosen)"),
                               ("Formula (A)", ORANGE, "Possession formula only")):
        vals = bt[bt["model"] == name].set_index("season").reindex(seasons)["margin_mae"]
        _line(ax, range(len(seasons)), vals, color, label)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 9 if name == "Formula (A)" else -15),
                        textcoords="offset points", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(seasons)), seasons)
    ax.set_xlim(-0.4, len(seasons) - 0.6)
    ax.set_ylabel("Margin error (points)", fontsize=9, color=MUTED)
    lo, hi = bt["margin_mae"].min(), bt["margin_mae"].max()
    ax.set_ylim(lo - 0.15, hi + 0.15)
    _legend(ax, loc="upper left")
    _headline(fig, "The regression beats the formula in every validation season",
              "Average miss on the final margin. Each season predicted by a model trained only on earlier seasons.")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.1)
    return _save(fig, out)


def vegas_by_month(vegas: pd.DataFrame, out):
    """Headline accuracy: the model against the Vegas closing line, month by month."""
    months = vegas[vegas["Month"] != "All"]
    total = vegas[vegas["Month"] == "All"].iloc[0]
    fig, ax = _figure(1, 10, 4.2)
    _style(ax, "")
    x = range(len(months))
    _line(ax, x, months["Margin MAE, model"], BLUE, f"Model ({total['Margin MAE, model']:.2f} overall)")
    _line(ax, x, months["Margin MAE, Vegas"], ORANGE, f"Vegas closing line ({total['Margin MAE, Vegas']:.2f} overall)")
    ax.set_xticks(list(x), months["Month"], rotation=60, fontsize=8)
    ax.set_ylabel("Margin error (points)", fontsize=9, color=MUTED)
    _legend(ax, loc="upper left")
    _headline(fig, "Within a third of a point of Vegas",
              f"Average miss on the final margin by month, {int(total['Games']):,} validation games "
              "(2023-24 and 2024-25). May and June are playoffs: few games, so they swing more.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.2)
    return _save(fig, out)


def ridge_alpha(curve: pd.DataFrame, chosen: float, out):
    """Decision: ridge strength. A flat curve means the penalty barely matters at sensible values."""
    fig, ax = _figure(1, 7, 4.2)
    _style(ax, "")
    _line(ax, curve["Alpha"], curve["Margin MAE, validation"], BLUE, "Validation (unseen months)")
    _line(ax, curve["Alpha"], curve["Margin MAE, training"], ORANGE, "Training (games it was fit on)")
    ax.set_xscale("log")
    ax.set_xticks(list(curve["Alpha"]), [f"{a:g}" for a in curve["Alpha"]])
    ax.minorticks_off()
    ax.axvline(chosen, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"chosen: {chosen:g}", (chosen, ax.get_ylim()[1]), xytext=(5, -12),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Ridge alpha (log scale; larger = coefficients pulled harder toward zero)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("Margin error (points)", fontsize=9, color=MUTED)
    _legend(ax, loc="center left")
    _headline(fig, "Regularization strength: flat until it starts to hurt",
              "Alphas 0.01 to 100 tie; the rule breaks ties toward the simpler model. "
              "Training and validation error stay close, so the model is not overfitting.")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.15)
    return _save(fig, out)


def learning_curve(learn: pd.DataFrame, out):
    """How much history the regression needs before more stops helping."""
    fig, ax = _figure(1, 7, 4.2)
    _style(ax, "")
    x = learn["Training team-games"]
    _line(ax, x, learn["Margin MAE, validation"], BLUE, "Validation (unseen months)")
    _line(ax, x, learn["Margin MAE, training"], ORANGE, "Training (games it was fit on)")
    ax.set_xscale("log")
    ax.set_xticks(list(x), [f"{int(v):,}" for v in x])
    ax.minorticks_off()
    ax.set_xlabel("Most recent team-games used to fit (log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("Margin error (points)", fontsize=9, color=MUTED)
    _legend(ax, loc="upper right")
    _headline(fig, "More history stops helping after about half a season",
              "Both curves level off together: the model is limited by its inputs (bias), not by data "
              "(variance). Better features are the next lever.")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.15)
    return _save(fig, out)


def playoff_settings(table: pd.DataFrame, chosen: str, out):
    """Decision: how playoff games feed ratings. Three metrics, chosen setting highlighted."""
    metrics = [("Margin MAE", "Margin error"), ("Total MAE", "Total error"),
               ("October total bias", "October total bias")]
    fig, axes = _figure(3, 12, 3.6, gridspec_kw={"wspace": 0.35})
    names = [s.replace("playoffs ", "") for s in table["Setting"]]
    y = list(range(len(names)))[::-1]
    for ax, (col, title) in zip(axes, metrics):
        _style(ax, title, grid_axis="x")
        vals = table[col].astype(float)
        colors = [BLUE if n == chosen else GRAY for n in names]
        ax.scatter(vals, y, s=90, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:+.2f}" if "bias" in col else f"{v:.2f}", (v, yi), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        if "bias" in col:
            ax.axvline(0, color=MUTED, linewidth=1)
        if "bias" in col:
            pad = max((vals.max() - vals.min()) * 0.35, 0.05)
            ax.set_xlim(min(vals.min(), 0) - pad, vals.max() + pad)
        else:
            # Same half-point window on both error panels, so gaps are drawn to the same scale.
            mid = (vals.max() + vals.min()) / 2
            ax.set_xlim(mid - 0.25, mid + 0.25)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.set_yticks(y, names if ax is axes[0] else [""] * len(names))
        ax.tick_params(axis="y", labelcolor=INK)
    axes[0].set_xlabel("points, lower is better", fontsize=8, color=MUTED)
    axes[1].set_xlabel("points, lower is better", fontsize=8, color=MUTED)
    axes[2].set_xlabel("points, closer to 0 is better", fontsize=8, color=MUTED)
    _headline(fig, f"Playoff games in ratings: {chosen} (blue)",
              "Same validation months for each setting. Margin error is a near tie; including playoffs "
              "keeps totals more accurate and early-season totals unbiased.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.17)
    return _save(fig, out)


VERDICT_PHRASES = {"better": "beats the noise", "worse": "is worse beyond the noise",
                   "tie": "is within the noise",
                   "fails bias guard": "beats the noise but fails the bias guard"}


def noise_test(table: pd.DataFrame, heading: str, out):
    """An on/off decision under the noise test: each change in average error, with a bar for
    the noise (2 standard errors, month to month). A bar that crosses zero is a tie."""
    current = table.loc[table["Verdict"] == "current", "Setting"].iloc[0]
    others = table[table["Verdict"] != "current"]
    items = []
    for _, r in others.iterrows():
        prefix = f"{r['Setting']}: " if len(others) > 1 else ""
        for metric in ("Margin", "Total"):
            items.append((f"{prefix}{metric} error", float(r[f"{metric} change"]), float(r[f"{metric} noise"])))
    fig, ax = _figure(1, 8, 3.4)
    _style(ax, "", grid_axis="x")
    y = list(range(len(items)))[::-1]
    for yi, (_, change, noise) in zip(y, items):
        # Square ends, so a bar stops exactly at the noise and whether it reaches zero is readable.
        ax.plot([change - noise, change + noise], [yi, yi], color=GRAY, linewidth=6, solid_capstyle="butt")
        ax.scatter([change], [yi], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{change:+.3f} \u00b1 {noise:.3f}", (change, yi), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9, color=INK, zorder=4,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
    ax.axvline(0, color=MUTED, linewidth=1)
    reach = max(max(abs(c) + n for _, c, n in items) * 1.6, 0.1)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_yticks(y, [name for name, _, _ in items])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("Change in average error, points (below zero is better)", fontsize=9, color=MUTED)
    verdict = others["Verdict"].iloc[0]
    games = int(others["Games"].iloc[0])
    # Name the setting being judged, so the headline still reads right once it becomes current
    # ("Strength of schedule off is worse beyond the noise").
    judged = f" {others['Setting'].iloc[0]}" if len(others) == 1 else ""
    _headline(fig, f"{heading}{judged} {VERDICT_PHRASES.get(verdict, verdict)}",
              f"Against the current setting ({current}) on the same {games:,} validation games. Margin error "
              "decides; total error counts only if margin is a tie. Gray bars are the noise (2 standard errors, "
              "month to month); a bar crossing zero is a tie, and a tie keeps the simpler model.")
    fig.subplots_adjust(left=0.17, right=0.95, top=0.7, bottom=0.2)
    return _save(fig, out)


def _label(value) -> str:
    """A table cell back as the label it was written as: 12.0 -> '12', 0.2 -> '0.2'."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def tuning(tables: dict, out):
    """Decision 10: one panel per hand-picked setting, in the order tested. Each value's change in
    margin error against the value in use at that step, with the noise. All panels share one
    scale, so a big effect looks big and a small one small."""
    plt = _plt()
    summary = tables["Summary"].set_index("Setting")["Test picked"]
    panels = [(h, t) for h, t in tables.items() if h != "Summary" and "Value" in t]
    cols = 4
    rows = -(-len(panels) // cols)
    height = 2.9 * rows + 1.6
    fig, axes = plt.subplots(rows, cols, figsize=(11, height), facecolor=SURFACE, sharey=True, squeeze=False)
    reach = 0.05
    changed = 0
    for ax, (heading, table) in zip(axes.flat, panels):
        name, _, unit = heading.partition(" (")
        _style(ax, name)
        labels = [_label(v) for v in table["Value"]]
        current = labels[table["Verdict"].tolist().index("current")]
        picked = _label(summary.get(name, current))
        changed += picked != current
        for x, (label, (_, r)) in enumerate(zip(labels, table.iterrows())):
            if label == current:
                ax.scatter([x], [0], s=90, facecolor=SURFACE, edgecolor=INK, linewidth=2, zorder=3)
                continue
            change, noise = float(r["Margin change"]), float(r["Margin noise"])
            reach = max(reach, abs(change) + noise)
            ax.plot([x, x], [change - noise, change + noise], color=GRAY, linewidth=6, solid_capstyle="butt")
            ax.scatter([x], [change], s=70, color=ORANGE if label == picked else BLUE,
                       edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_xlabel(unit.rstrip(")"), fontsize=9, color=MUTED)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes[0][0].set_ylim(-reach * 1.25, reach * 1.25)
    for row in axes:
        row[0].set_ylabel("Change in margin error, points", fontsize=9, color=MUTED)
    title = (f"{changed} of {len(panels)} hand-picked settings beat the noise" if changed
             else "None of the hand-picked settings beat the noise")
    _headline(fig, title,
              "Each value against the value in use (open circle) on the same validation games; below zero is "
              "better. Gray bars are the noise (2 standard errors, month to month): a bar crossing zero is a tie, "
              "which keeps the value in use. Orange marks a value the test picked. Settings were tested in "
              "reading order, each with the earlier winners in place.")
    fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.75 / height, bottom=0.1, hspace=0.75, wspace=0.12)
    return _save(fig, out)


def cv_scheme(out):
    """How evaluation splits time: rolling monthly folds on validation seasons, test season locked."""
    plt = _plt()
    seasons = config.VALIDATION_SEASONS
    first = config.TRAIN_SEASONS[0]
    start = pd.Timestamp(f"{first[:4]}-10-01")
    end = pd.Timestamp(f"{int(config.TEST_SEASON[:4]) + 1}-06-30")
    folds = [pd.Timestamp(f"{s[:4]}-{m}-01") for s in seasons for m in ("11",)] + \
            [pd.Timestamp(f"{int(s[:4]) + 1}-03-01") for s in seasons]
    folds = sorted(folds)
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    _style(ax, "", grid_axis="x")
    test_start = pd.Timestamp(f"{config.TEST_SEASON[:4]}-10-01")
    for i, f in enumerate(folds):
        yi = len(folds) - i
        ax.barh(yi, (f - start).days, left=start, height=0.5, color=BLUE)
        ax.barh(yi, 30, left=f, height=0.5, color=ORANGE)
        ax.barh(yi, (end - test_start).days, left=test_start, height=0.5, color=GRAY)
        ax.annotate(f"predict {f:%b %Y}", (f + pd.Timedelta(days=34), yi), va="center",
                    fontsize=8.5, color=INK)
    ax.annotate("locked test season\n(scored once, logged)", (test_start + (end - test_start) / 2, 0.15),
                ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(-0.4, len(folds) + 0.6)
    ax.set_xlim(start, end)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=BLUE, label="train on every earlier game"),
                       Patch(color=ORANGE, label="predict the next month"),
                       Patch(color=GRAY, label=f"{config.TEST_SEASON}: never used for choices")],
              frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left", ncol=3, bbox_to_anchor=(0, -0.12))
    _headline(fig, "How the model is evaluated",
              "Rolling-origin cross-validation: every month of the validation seasons is a fold (four shown). "
              "No fold ever trains on a later game.")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.25)
    return _save(fig, out)


def prediction_breakdown(details: pd.DataFrame, out):
    """One game, step by step: how each team's predicted points are built."""
    fig, axes = _figure(2, 12, 4.6, gridspec_kw={"wspace": 0.55})
    rows = details.sort_values("IS_HOME")  # away first, home second
    reached = []
    for ax, (_, r) in zip(axes, rows.iterrows()):
        _style(ax, f"{r['team']} {'vs' if r['IS_HOME'] else 'at'} {r['opponent']}: "
                   f"{r['predicted_pts']:.1f} points", grid_axis="x")
        steps = [s for s in model.STEPS if s in r.index and pd.notna(r[s])]  # "Four factors" only if used
        labels = steps + ["Predicted points"]
        y = list(range(len(labels)))[::-1]
        running = 0.0
        for i, step in enumerate(steps):
            v = float(r[step])
            if i == 0:
                running = v
                ax.scatter([v], [y[i]], s=90, color=INK, zorder=3)
                ax.annotate(f"{v:.1f}", (v, y[i]), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=9, color=INK)
                continue
            ax.barh(y[i], v, left=running, height=0.5, color=BLUE if v >= 0 else ORANGE)
            ax.annotate(f"{v:+.1f}", (max(running, running + v), y[i]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
            running += v
        final = float(r["predicted_pts"])  # equals the running total; label the saved prediction itself
        ax.scatter([final], [y[-1]], s=90, color=INK, zorder=3)
        ax.annotate(f"{final:.1f}", (final, y[-1]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", labelcolor=INK)
        cum = float(r[steps[0]])
        reached.append(cum)
        for step in steps[1:]:
            cum += float(r[step])
            reached.append(cum)
        ax.set_xlabel("Points", fontsize=9, color=MUTED)
    for ax in axes:  # one scale for both teams, so their steps compare directly
        ax.set_xlim(min(reached) - 2, max(reached) + 2)
    home = rows[rows["IS_HOME"] == 1].iloc[0]
    away = rows[rows["IS_HOME"] == 0].iloc[0]
    missing = "; ".join(f"{r['team']}: {r['missing']}" for _, r in rows.iterrows()
                        if isinstance(r["missing"], str) and r["missing"])
    _headline(fig, f"How the prediction is built: {away['team']} {away['predicted_pts']:.1f}, "
                   f"{home['team']} {home['predicted_pts']:.1f} ({home['date']})",
              "Start from the pace x efficiency formula, then add each adjustment the regression applies. "
              + (f"Listed on the injury report: {missing}." if missing else "No rotation players listed as out."))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.75, bottom=0.13)
    return _save(fig, out)


def run(fig_dir=None) -> list:
    """Redraw every chart that the files in results/ and data/model.json support
    (into docs/figures unless told otherwise)."""
    fig_dir = fig_dir or config.ROOT / "docs" / "figures"
    made = []
    results = config.ROOT / "results"
    if config.MODEL_PATH.exists():
        made.append(coefficients(model.load(config.MODEL_PATH), fig_dir / "coefficients.png"))
    if (results / "backtest.md").exists():
        made.append(model_vs_formula(read_backtest(), fig_dir / "model_vs_formula.png"))
    made.append(cv_scheme(fig_dir / "evaluation_scheme.png"))
    if (results / "tuning.md").exists():
        made.append(tuning(read_tables(results / "tuning.md"), fig_dir / "tuning.png"))
    if (results / "evaluation.md").exists():
        tables = read_tables(results / "evaluation.md")
        if "Against the Vegas closing line" in tables:
            made.append(vegas_by_month(tables["Against the Vegas closing line"], fig_dir / "vegas_by_month.png"))
        if "Complexity curve (ridge alpha)" in tables:
            made.append(ridge_alpha(tables["Complexity curve (ridge alpha)"], config.RIDGE_ALPHA,
                                    fig_dir / "ridge_alpha.png"))
        if "Learning curve" in tables:
            made.append(learning_curve(tables["Learning curve"], fig_dir / "learning_curve.png"))
        for heading, name in (("Strength of schedule", "strength_of_schedule"), ("Four factors", "four_factors")):
            if heading in tables:
                made.append(noise_test(tables[heading], heading, fig_dir / f"{name}.png"))
        if "Playoff games in ratings" in tables:
            made.append(playoff_settings(tables["Playoff games in ratings"], config.PLAYOFF_GAMES,
                                         fig_dir / "playoff_settings.png"))
    return made
