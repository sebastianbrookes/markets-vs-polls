"""
Visualize event response analysis: how Polymarket and FiveThirtyEight
reacted to major 2024 campaign events.

Generates three plots:
    1. Event timeline — full March-Sept 12 with event bands
    2. Reaction scoreboard — grouped horizontal bars per event
    3. Indexed event study — small multiples showing response lag per event

Run from project root:
    python -m src.visualize.event_plots
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.clean.utils import PROCESSED_DIR, SWING_STATES
from src.analysis.events.event_response import (
    EVENTS,
    load_data,
    compute_swing_average,
    compute_538_swing_timeseries,
    compute_raw_indexed_window,
    detect_price_in_day,
    build_reaction_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(_project_root) / "figures" / "events"

CLR_PM = "#1f4e79"
CLR_538 = "#ca5800"
CLR_CORRECT = "#2E8B57"
CLR_WRONG = "#C53030"

DPI = 200


def _configure_plot_style():
    """Apply a consistent style baseline for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _style_axis(ax, y_grid_alpha=0.3):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=y_grid_alpha)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Plot 1: Event Timeline (full March–Sept 12)
# ---------------------------------------------------------------------------
def plot_event_timeline(pm_swing, p538_swing):
    """Full-period line chart with vertical event bands."""
    # Clip both series to the overlap period (through Sept 12)
    cutoff = pd.Timestamp("2024-09-12")
    pm_swing = pm_swing[pm_swing.index <= cutoff]
    p538_swing = p538_swing[p538_swing.index <= cutoff]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Polymarket (left y-axis, probability scale)
    ax1.plot(pm_swing.index, pm_swing.values,
             color=CLR_PM, linewidth=2, label="Polymarket (prob.)", zorder=3)
    ax1.set_ylabel("Polymarket: Trump Lead (probability)", color=CLR_PM)
    ax1.tick_params(axis="y", labelcolor=CLR_PM)
    ax1.set_ylim(pm_swing.min() - 0.03, pm_swing.max() + 0.08)

    # 538 (right y-axis, vote-share % scale)
    ax2 = ax1.twinx()
    ax2.plot(p538_swing.index, p538_swing.values,
             color=CLR_538, linewidth=2, linestyle="--", dashes=(6, 3),
             label="FiveThirtyEight (vote %)", zorder=3)
    ax2.set_ylabel("538: Trump Lead (vote share %)", color=CLR_538)
    ax2.tick_params(axis="y", labelcolor=CLR_538)
    ax2.set_ylim(p538_swing.min() - 1, p538_swing.max() + 2)

    # Background shading: faint red above 0 (Trump leads), faint blue below
    y_lo, y_hi = ax1.get_ylim()
    ax1.axhspan(0, y_hi, color="#E53935", alpha=0.04, zorder=0)
    ax1.axhspan(y_lo, 0, color="#1565C0", alpha=0.04, zorder=0)
    ax1.set_ylim(y_lo, y_hi)  # restore limits after axhspan

    # Emphasized tie line at y=0
    ax1.axhline(0, color="#555555", linewidth=1.4, linestyle="-", zorder=2)
    x_start = pm_swing.index.min()
    ax1.text(x_start + pd.Timedelta(days=5), 0.01, "Tied",
             color="#555555", fontsize=10, va="bottom", ha="left",
             fontstyle="italic", zorder=4)

    # Event bands
    event_colors = ["#FFE0B2", "#FFCDD2", "#C8E6C9", "#B3E5FC", "#E1BEE7"]
    for i, event in enumerate(EVENTS):
        edate = pd.Timestamp(event["date"])
        ax1.axvspan(edate - pd.Timedelta(hours=12),
                    edate + pd.Timedelta(hours=12),
                    color=event_colors[i], alpha=0.5, zorder=1)

        # Horizontal label, staggered to avoid overlap
        y_top = ax1.get_ylim()[1]
        level = i % 2  # 0 = low, 1 = high
        offset_y = [6, 22][level]

        annot_kw = dict(
            xy=(edate, y_top),
            xytext=(0, offset_y), textcoords="offset points",
            fontsize=8,
            ha="center", va="bottom",
            color="#444444",
            zorder=5,
        )
        if level:
            annot_kw["arrowprops"] = dict(
                arrowstyle="-", color="#AAAAAA", lw=0.7,
            )

        ax1.annotate(event["name"], **annot_kw)

    # Formatting
    ax1.set_xlabel("Date (2024)")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.set_title(
        "How Prediction Markets and Polls Tracked the Campaign\n"
        "Swing-state average Trump lead, March–Sept 12 2024",
        pad=40, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=True, framealpha=0.9)

    _style_axis(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    fig.tight_layout()
    path = FIGURES_DIR / "event_timeline.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Reaction Scoreboard (horizontal grouped bars)
# ---------------------------------------------------------------------------
def plot_reaction_scoreboard(summary):
    """Horizontal grouped bar chart — shift per event per source."""
    event_names = [e["name"] for e in EVENTS]
    n = len(event_names)
    y = np.arange(n)
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, event_name in enumerate(event_names):
        for j, (source, color, offset) in enumerate([
            ("Polymarket", CLR_PM, -bar_h / 2),
            ("538", CLR_538, bar_h / 2),
        ]):
            row = summary[(summary["event"] == event_name) &
                          (summary["source"] == source)]
            if row.empty:
                continue
            row = row.iloc[0]

            z = row["shift_z"]
            if np.isnan(z):
                ax.text(0, y[i] + offset, "no data",
                        ha="center", va="center", fontsize=8,
                        color="#999999", fontstyle="italic")
                continue

            is_correct = row["direction_match"] == "correct"
            edge_color = CLR_CORRECT if is_correct else CLR_WRONG
            edge_width = 2.5

            ax.barh(y[i] + offset, z, height=bar_h * 0.85,
                    color=color, alpha=0.8,
                    edgecolor=edge_color, linewidth=edge_width)

            # Value label (z-score with sigma)
            label_x = z + (0.08 if z >= 0 else -0.08)
            ha = "left" if z >= 0 else "right"
            ax.text(label_x, y[i] + offset, f"{z:+.1f}\u03c3",
                    ha=ha, va="center", fontsize=9, color=color,
                    fontweight="bold")

    # Background shading: blue (pro-Harris) left, red (pro-Trump) right
    x_lo, x_hi = ax.get_xlim()
    x_abs = max(abs(x_lo), abs(x_hi))
    ax.axvspan(-x_abs, 0, color="#1565C0", alpha=0.12, zorder=0)
    ax.axvspan(0, x_abs, color="#E53935", alpha=0.12, zorder=0)
    ax.set_xlim(-x_abs, x_abs)

    # Zero line
    ax.axvline(0, color="#333333", linewidth=1, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(event_names)
    ax.invert_yaxis()
    ax.set_xlabel("Reaction Intensity (z-score within source)")
    ax.set_title(
        "Event Reaction Scoreboard: Who Moved in the Right Direction?",
        fontweight="bold", pad=15,
    )
    ax.text(0.5, 1.005, "Normalized to each source\u2019s daily volatility",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#666666", fontstyle="italic")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLR_PM, alpha=0.8, label="Polymarket"),
        Patch(facecolor=CLR_538, alpha=0.8, label="FiveThirtyEight"),
        Patch(facecolor="white", edgecolor=CLR_CORRECT,
              linewidth=2, label="Correct direction"),
        Patch(facecolor="white", edgecolor=CLR_WRONG,
              linewidth=2, label="Wrong direction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              frameon=True, framealpha=0.9, fontsize=9)

    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "event_scoreboard.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Indexed Event Study (small multiples)
# ---------------------------------------------------------------------------
def _align_dual_axes(ax_left, ax_right):
    """Force y=0 to the same vertical pixel on both axes.

    Computes the union of both y-ranges and sets symmetric limits
    so zero aligns perfectly.
    """
    lo_l, hi_l = ax_left.get_ylim()
    lo_r, hi_r = ax_right.get_ylim()

    # Fraction of total range that is below zero, for each axis
    frac_l = abs(lo_l) / (abs(lo_l) + abs(hi_l)) if (abs(lo_l) + abs(hi_l)) else 0.5
    frac_r = abs(lo_r) / (abs(lo_r) + abs(hi_r)) if (abs(lo_r) + abs(hi_r)) else 0.5

    # Use the larger fraction so neither axis gets clipped
    frac = max(frac_l, frac_r)

    # Expand each axis so that frac of the range sits below zero
    range_l = max(abs(lo_l) / frac, abs(hi_l) / (1 - frac)) if frac not in (0, 1) else abs(lo_l) + abs(hi_l)
    range_r = max(abs(lo_r) / frac, abs(hi_r) / (1 - frac)) if frac not in (0, 1) else abs(lo_r) + abs(hi_r)

    ax_left.set_ylim(-frac * range_l, (1 - frac) * range_l)
    ax_right.set_ylim(-frac * range_r, (1 - frac) * range_r)


def _draw_panel(ax_left, event, pm_swing, p538_swing, is_hero=False):
    """Draw one event panel with dual y-axes, step PM, smooth 538."""
    edate = event["date"]

    # Start at Day -1 to remove pre-event noise
    pm_win = compute_raw_indexed_window(pm_swing, edate, pre_days=1,
                                         scale=100)
    p538_win = compute_raw_indexed_window(p538_swing, edate, pre_days=1,
                                           scale=1)

    # --- Left axis: Polymarket (step line) ---
    if not pm_win.empty:
        ax_left.step(pm_win.index, pm_win.values, where="post",
                     color=CLR_PM, linewidth=2.2, label="Polymarket",
                     zorder=3)
    ax_left.set_ylabel("Change in Win Prob. (% pts)", color=CLR_PM,
                        fontsize=9 if not is_hero else 11)
    ax_left.tick_params(axis="y", colors=CLR_PM, labelsize=9)

    # --- Right axis: 538 (smooth line) ---
    ax_right = ax_left.twinx()
    if not p538_win.empty:
        ax_right.plot(p538_win.index, p538_win.values,
                      color=CLR_538, linewidth=2.5, label="538",
                      zorder=3)
    ax_right.set_ylabel("Change in Vote Share (% pts)", color=CLR_538,
                         fontsize=9 if not is_hero else 11)
    ax_right.tick_params(axis="y", colors=CLR_538, labelsize=9)

    # --- Zero-align both axes ---
    _align_dual_axes(ax_left, ax_right)

    # --- Reference lines ---
    ax_left.axvline(0, color="#888888", linewidth=1, linestyle="--",
                    alpha=0.6, zorder=2)
    ax_left.axhline(0, color="#888888", linewidth=0.8, linestyle="-",
                    alpha=0.4, zorder=1)

    # --- Title & direction badge ---
    direction_labels = {
        "pro-Trump": "+Trump",
        "pro-Harris": "+Harris",
        "neutral": "neutral",
    }
    badge = direction_labels.get(event["expected_direction"], "")
    title = f"{event['name']}  ({badge})" if badge else event["name"]
    ax_left.set_title(title, fontsize=12 if is_hero else 10,
                       fontweight="bold")

    # --- Hero-only annotations ---
    if is_hero:
        pm_lag = detect_price_in_day(pm_win)
        p538_lag = detect_price_in_day(p538_win)

        # Day 0 label
        ax_left.annotate(
            "Biden\nwithdraws", xy=(0, 0),
            xytext=(0, 12), textcoords="offset points",
            fontsize=9, ha="center", color="#555555",
            fontstyle="italic", zorder=5,
        )

        # Lag bracket — positioned close to the data
        if pm_lag is not None and p538_lag is not None and p538_lag > pm_lag:
            # Place bracket at ~30% of PM's total drop (near the top of
            # the data region, not floating in white space)
            pm_day10 = pm_win.loc[10] if 10 in pm_win.index else 0
            bracket_y = pm_day10 * 0.25
            ax_left.annotate(
                "", xy=(p538_lag, bracket_y),
                xytext=(pm_lag, bracket_y),
                arrowprops=dict(arrowstyle="<->", color="#333333",
                                lw=1.5),
                zorder=5,
            )
            mid = (pm_lag + p538_lag) / 2
            lag_days = p538_lag - pm_lag
            ax_left.text(mid, bracket_y,
                          f"{lag_days}-day lag",
                          ha="center", va="bottom", fontsize=10,
                          fontweight="bold", color="#333333", zorder=5)

    ax_left.set_xlabel("Days from Event")

    _style_axis(ax_left)
    ax_right.spines["top"].set_visible(False)
    ax_right.grid(False)

    return ax_right


def plot_indexed_event_study(pm_swing, p538_swing):
    """Single hero panel: market speed vs. poll lag for Biden dropout."""
    from matplotlib.lines import Line2D

    hero_event = next(e for e in EVENTS if e["name"] == "Biden Drops Out")

    fig, ax = plt.subplots(figsize=(13, 7))
    _draw_panel(ax, hero_event, pm_swing, p538_swing, is_hero=True)

    fig.suptitle(
        "Market Traders Spotted the Harris Surge\n"
        "a Week Before Polls",
        fontsize=15, fontweight="bold", y=1.03,
    )

    # --- Combined legend ---
    legend_elements = [
        Line2D([0], [0], color=CLR_PM, linewidth=2.2,
               drawstyle="steps-post", label="Polymarket (win prob.)"),
        Line2D([0], [0], color=CLR_538, linewidth=2.5,
               label="FiveThirtyEight (vote share)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=11,
               frameon=True, framealpha=0.9)

    # --- Footnote ---
    fig.text(0.5, -0.10,
             "Left axis (blue): change in percentage-point win "
             "probability.  Right axis (orange): change in "
             "percentage-point vote share.\n"
             "Both zeroed to Day \u22121 baseline.",
             ha="center", va="top", fontsize=8, color="#777777",
             fontstyle="italic")

    path = FIGURES_DIR / "event_dropout_response.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    print("Loading data...")
    pm, p538 = load_data()

    print("Computing swing-state averages...")
    pm_swing = compute_swing_average(pm)
    p538_swing = compute_538_swing_timeseries(p538)

    print("Building reaction summary...")
    summary = build_reaction_summary(pm, p538)

    print("Generating plots...")
    plot_event_timeline(pm_swing, p538_swing)
    plot_reaction_scoreboard(summary)
    plot_indexed_event_study(pm_swing, p538_swing)

    print("Done — 3 plots saved to figures/events/")


if __name__ == "__main__":
    main()
