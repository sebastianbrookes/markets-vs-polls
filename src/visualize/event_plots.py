"""Event-response visualizations for campaign events."""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.events.event_response import (
    EVENTS,
    build_reaction_summary,
    compute_538_swing_timeseries,
    compute_raw_indexed_window,
    compute_swing_average,
    load_data,
)


@dataclass(frozen=True)
class Theme:
    PM_COLOR: str = "#004276"
    F38_COLOR: str = "#D95F02"
    GRID_COLOR: str = "#E2E2E2"
    TEXT_MAIN: str = "#303030"
    TEXT_MUTED: str = "#6F6F6F"
    EVENT_BAND: str = "#F3F3F3"
    FONT_STACK: tuple[str, ...] = ("Inter", "DejaVu Sans", "sans-serif")
    DPI: int = 300


THEME = Theme()
FIGURES_DIR = Path(PROJECT_ROOT) / "figures" / "events"
TIMELINE_START = pd.Timestamp("2024-06-01")
TIMELINE_END = pd.Timestamp("2024-09-12")
EVENT_LABEL_LEVELS = {
    "Biden Drops Out": 1,
}
SOURCES = (
    ("Polymarket", "Polymarket", THEME.PM_COLOR, -0.175),
    ("538", "FiveThirtyEight", THEME.F38_COLOR, 0.175),
)


def apply_style():
    """Apply shared matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": THEME.DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": THEME.TEXT_MAIN,
            "axes.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "text.color": THEME.TEXT_MAIN,
            "font.family": list(THEME.FONT_STACK),
            "font.size": 10,
            "xtick.color": THEME.TEXT_MUTED,
            "ytick.color": THEME.TEXT_MUTED,
            "xtick.major.size": 4,
            "xtick.major.width": 0.5,
            "ytick.left": False,
            "grid.color": THEME.GRID_COLOR,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def format_header(fig, title, subtitle):
    """Add a left-aligned title block."""
    fig.text(0.05, 0.98, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.05, 0.93, subtitle, fontsize=11, color=THEME.TEXT_MUTED, ha="left", va="top")


def _filter_window(series, start, end):
    return series[(series.index >= start) & (series.index <= end)]


def _format_month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b."))


def _map_y_between_axes(value, src_ax, dst_ax):
    src_min, src_max = src_ax.get_ylim()
    dst_min, dst_max = dst_ax.get_ylim()
    if src_max == src_min:
        return dst_min
    return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)


def _align_dual_axes(ax_left, ax_right):
    """Align the y=0 line between twinned axes."""
    l_min, l_max = ax_left.get_ylim()
    r_min, r_max = ax_right.get_ylim()
    if l_max - l_min == 0 or r_max - r_min == 0:
        return

    l_ratio = abs(l_min) / (l_max - l_min)
    r_ratio = abs(r_min) / (r_max - r_min)
    ratio = max(l_ratio, r_ratio)

    l_span = l_max if l_max > 0 else abs(l_min)
    r_span = r_max if r_max > 0 else abs(r_min)
    if ratio >= 1:
        ax_left.set_ylim(-l_span, l_span)
        ax_right.set_ylim(-r_span, r_span)
        return

    scale = ratio / (1 - ratio)
    ax_left.set_ylim(-scale * l_span, l_span)
    ax_right.set_ylim(-scale * r_span, r_span)


def _add_event_bands(ax, y_anchor, start, end):
    for event in EVENTS:
        event_date = pd.Timestamp(event["date"])
        if event_date < start or event_date > end:
            continue

        ax.axvspan(
            event_date - pd.Timedelta(days=1),
            event_date + pd.Timedelta(days=1),
            color=THEME.EVENT_BAND,
            zorder=1,
        )

        level = EVENT_LABEL_LEVELS.get(event["name"], 0)
        ax.annotate(
            event["name"],
            xy=(event_date, y_anchor),
            xytext=(0, 6 + 14 * level),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=THEME.TEXT_MAIN,
            annotation_clip=False,
        )


def plot_event_timeline(pm_swing, p538_swing):
    """Plot campaign movement in Polymarket and FiveThirtyEight series."""
    pm_filtered = _filter_window(pm_swing, TIMELINE_START, TIMELINE_END)
    p538_filtered = _filter_window(p538_swing, TIMELINE_START, TIMELINE_END)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    fig.subplots_adjust(top=0.78, right=0.82)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    
    # Plot the lines and assign labels for the legend
    line1 = ax1.plot(
        pm_filtered.index, pm_filtered.values, 
        color=THEME.PM_COLOR, linewidth=2.5, zorder=4, 
        label="Polymarket (Win Prob.)"
    )
    line2 = ax2.plot(
        p538_filtered.index, p538_filtered.values, 
        color=THEME.F38_COLOR, linewidth=2.2, zorder=3, 
        label="FiveThirtyEight (Vote Share %)"
    )
    ax2.spines["right"].set_visible(False)

    # Color-code tick labels and add explicit axis labels
    ax1.tick_params(axis="y", colors=THEME.PM_COLOR, labelsize=11)
    ax1.set_ylabel("Win Probability", color=THEME.PM_COLOR, 
                   fontweight="bold", fontsize=11, labelpad=10)
    
    ax2.tick_params(axis="y", colors=THEME.F38_COLOR, labelsize=11)
    ax2.set_ylabel("Vote Share %", color=THEME.F38_COLOR, 
                   fontweight="bold", fontsize=11, rotation=-90, labelpad=20)

    _align_dual_axes(ax1, ax2)
    ax1.axhline(0, color=THEME.TEXT_MAIN, linewidth=1.2, zorder=2)

    # --- UPDATED: Stacked top-right legend aligned with header ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # loc="upper right" and anchor y=1.32 pushes it up into the margin space 
    # created by subplots_adjust(top=0.78), aligning it with the title.
    leg = ax1.legend(
        lines, labels, 
        loc="upper right", 
        bbox_to_anchor=(1.0, 1.32), 
        ncol=1, 
        frameon=False, 
        fontsize=11,
        handletextpad=0.5,
        labelspacing=0.6
    )
    
    # Color-code the legend text to perfectly match the lines
    for text, color in zip(leg.get_texts(), [THEME.PM_COLOR, THEME.F38_COLOR]):
        text.set_color(color)
        text.set_fontweight("bold")
    # --------------------------------------------------------------

    if not pm_filtered.empty:
        start_date = pm_filtered.index.min()
    elif not p538_filtered.empty:
        start_date = p538_filtered.index.min()
    else:
        start_date = TIMELINE_START
        
    ax1.annotate(
        "Trump leads",
        xy=(start_date, 0),
        xytext=(10, 8),
        textcoords="offset points",
        color=THEME.TEXT_MUTED,
        fontsize=9,
        va="bottom",
    )
    ax1.annotate(
        "Biden / Harris leads",
        xy=(start_date, 0),
        xytext=(10, -8),
        textcoords="offset points",
        color=THEME.TEXT_MUTED,
        fontsize=9,
        va="top",
    )

    _, y_max = ax1.get_ylim()
    _add_event_bands(ax1, y_max, TIMELINE_START, TIMELINE_END)
    _format_month_axis(ax1)

    format_header(
        fig,
        "How Prediction Markets and Polls Tracked the Campaign",
        "Swing-state average Trump lead, showing Polymarket probability vs. FiveThirtyEight vote share.",
    )
    fig.savefig(FIGURES_DIR / "timeline.png", bbox_inches="tight")
    plt.close(fig)


def plot_reaction_scoreboard(summary):
    """Compare event reaction size and direction by source."""
    event_names = [event["name"] for event in EVENTS]
    y_pos = np.arange(len(event_names))
    bar_h = 0.35

    values = (
        summary[["event", "source", "shift_z", "direction_match"]]
        .drop_duplicates(subset=["event", "source"], keep="last")
        .set_index(["event", "source"])
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(top=0.78, left=0.25)
    ax.axvline(0, color=THEME.TEXT_MAIN, linewidth=1, zorder=2)

    max_abs_z = 0.0
    for i, event_name in enumerate(event_names):
        for source, _, color, y_offset in SOURCES:
            key = (event_name, source)
            if key not in values.index:
                continue
            row = values.loc[key]
            z_score = row["shift_z"]
            if pd.isna(z_score):
                continue

            is_correct = row["direction_match"] == "correct"
            face_color = color if is_correct else "white"
            alpha = 0.95 if is_correct else 0.65

            ax.barh(
                y_pos[i] + y_offset,
                z_score,
                height=bar_h * 0.86,
                facecolor=face_color,
                edgecolor=color,
                linewidth=1.4,
                alpha=alpha,
                zorder=3,
            )

            label_pad = 0.08 if z_score >= 0 else -0.08
            ax.text(
                z_score + label_pad,
                y_pos[i] + y_offset,
                f"{z_score:+.1f}σ",
                ha="left" if z_score >= 0 else "right",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )
            max_abs_z = max(max_abs_z, abs(z_score))

    max_abs_z = max(max_abs_z, 1.0)
    x_pad = max(0.25, 0.15 * max_abs_z)
    ax.set_xlim(-max_abs_z - x_pad, max_abs_z + x_pad)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(event_names)
    ax.invert_yaxis()
    ax.set_xlabel("Reaction intensity (z-score, normalized to daily volatility)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    legend_elements = [
        Patch(facecolor=THEME.TEXT_MAIN, edgecolor=THEME.TEXT_MAIN, label="Expected direction"),
        Patch(facecolor="white", edgecolor=THEME.TEXT_MAIN, linewidth=1.4, label="Unexpected direction"),
        Line2D([0], [0], color=THEME.PM_COLOR, lw=4, label="Polymarket"),
        Line2D([0], [0], color=THEME.F38_COLOR, lw=4, label="FiveThirtyEight"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    format_header(
        fig,
        "Event Reaction Scoreboard",
        "Which source reacted more intensely, and did they move in the expected direction?",
    )
    fig.savefig(FIGURES_DIR / "scoreboard.png", bbox_inches="tight")
    plt.close(fig)


def _steepest_drop_day(series):
    diffs = series.diff().dropna()
    if diffs.empty:
        return None
    return diffs.idxmin()


def plot_indexed_event_study(pm_swing, p538_swing):
    """Plot response timing around Biden dropping out."""
    hero_date = next(event["date"] for event in EVENTS if event["name"] == "Biden Drops Out")
    pm_window = compute_raw_indexed_window(pm_swing, hero_date, pre_days=1, scale=100)
    p538_window = compute_raw_indexed_window(p538_swing, hero_date, pre_days=1, scale=1)

    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(top=0.82, right=0.95)
    ax1.grid(axis="y", alpha=0.5)

    ax2 = ax1.twinx()
    ax1.step(pm_window.index, pm_window.values, where="post", color=THEME.PM_COLOR, linewidth=2.5)
    ax2.plot(p538_window.index, p538_window.values, color=THEME.F38_COLOR, linewidth=2.3)
    ax2.spines["right"].set_visible(False)

    _align_dual_axes(ax1, ax2)
    ax1.axvline(0, color=THEME.TEXT_MAIN, linewidth=1, linestyle=":", zorder=2)
    ax1.axhline(0, color=THEME.TEXT_MAIN, linewidth=1, zorder=2)
    ax1.text(0, ax1.get_ylim()[1] * 0.95, " Biden withdraws", color=THEME.TEXT_MAIN, fontsize=9)

    ax1.legend(
        handles=[
            Line2D([0], [0], color=THEME.PM_COLOR, lw=3, label="Polymarket (percentage points)"),
            Line2D([0], [0], color=THEME.F38_COLOR, lw=3, label="FiveThirtyEight (vote share points)"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=10,
        borderaxespad=1.5,
    )

    pm_lag = _steepest_drop_day(pm_window)
    p538_lag = _steepest_drop_day(p538_window)
    if pm_lag is not None and p538_lag is not None and p538_lag > pm_lag:
        pm_floor = float(pm_window.min()) if not pm_window.empty else 0.0
        y_bracket = 0.35 * pm_floor if pm_floor < 0 else -0.1
        ax1.annotate(
            "",
            xy=(p538_lag, y_bracket),
            xytext=(pm_lag, y_bracket),
            arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color=THEME.TEXT_MAIN, lw=1.2),
        )
        ax1.annotate(
            f"{int(p538_lag - pm_lag)}-day lag",
            xy=((pm_lag + p538_lag) / 2, y_bracket),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=THEME.TEXT_MAIN,
        )

        pm_value = pm_window.loc[pm_lag]
        ax1.plot([pm_lag, pm_lag], [y_bracket, pm_value], color=THEME.PM_COLOR, linestyle=":", lw=1.2, alpha=0.7)
        ax1.plot([pm_lag], [pm_value], marker="o", color=THEME.PM_COLOR, markersize=5)

        p538_value = p538_window.loc[p538_lag]
        y_bracket_ax2 = _map_y_between_axes(y_bracket, ax1, ax2)
        ax2.plot(
            [p538_lag, p538_lag],
            [y_bracket_ax2, p538_value],
            color=THEME.F38_COLOR,
            linestyle=":",
            lw=1.2,
            alpha=0.7,
        )
        ax2.plot([p538_lag], [p538_value], marker="o", color=THEME.F38_COLOR, markersize=5)

    ax1.set_xlabel("Days from event")

    format_header(
        fig,
        "Market Traders Spotted the Harris Surge Before Polling Caught Up",
        "Change in probability and vote share, indexed to the day before Biden dropped out.",
    )
    fig.savefig(FIGURES_DIR / "dropout-response.png", bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate all event-response figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("Loading data...")
    polymarket, fivethirtyeight = load_data()

    print("Computing swing-state averages...")
    pm_swing = compute_swing_average(polymarket)
    p538_swing = compute_538_swing_timeseries(fivethirtyeight)

    print("Building reaction summary...")
    summary = build_reaction_summary(polymarket, fivethirtyeight)

    print("Generating plots...")
    plot_event_timeline(pm_swing, p538_swing)
    print("  Saved event_timeline.png")
    plot_reaction_scoreboard(summary)
    print("  Saved event_scoreboard.png")
    plot_indexed_event_study(pm_swing, p538_swing)
    print("  Saved event_dropout_response.png")
    print("Success. All figures saved to figures/events/")


if __name__ == "__main__":
    main()
