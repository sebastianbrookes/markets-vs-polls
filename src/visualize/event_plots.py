"""
Visualize event response analysis: how Polymarket and FiveThirtyEight
reacted to major 2024 campaign events.

Generates three plots:
    1. Event timeline (swing around each campaign event)
    2. Reaction scoreboard (head-to-head event response comparison)
    3. Indexed event study (dropout response, normalized windows)

Run from project root:
    python -m src.visualize.event_plots
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.analysis.events.event_response import (
    EVENTS,
    build_reaction_summary,
    compute_538_swing_timeseries,
    compute_raw_indexed_window,
    compute_swing_average,
    load_data,
)

# ---------------------------------------------------------------------------
# Design System
# ---------------------------------------------------------------------------
@dataclass
class Theme:
    """
    Centralized design tokens for consistent plot styling.
    """
    PM_COLOR: str = "#004276"      # Deep Navy
    F38_COLOR: str = "#D95F02"     # Burnt Orange
    CORRECT_COLOR: str = "#4A7c59" # Muted Green
    WRONG_COLOR: str = "#C24C3A"   # Muted Red
    GRID_COLOR: str = "#E0E0E0"    # Light Gray
    TEXT_MAIN: str = "#333333"     # Off-Black
    TEXT_MUTED: str = "#777777"    # Medium Gray
    EVENT_BAND: str = "#F0F0F0"    # Very faint gray for events
    FONT_FAMILY: str = "Inter"
    DPI: int = 300                 

FIGURES_DIR = Path(_project_root) / "figures" / "events"


def apply_style():
    """
    Apply a clean matplotlib style using the Theme design tokens.

    Returns
    -------
    None
    """
    plt.rcParams.update({
        "figure.dpi": Theme.DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": Theme.TEXT_MAIN,
        "axes.linewidth": 0.5,
        "axes.labelcolor": Theme.TEXT_MAIN,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,  
        "text.color": Theme.TEXT_MAIN,
        "font.family": Theme.FONT_FAMILY,
        "font.size": 10,
        "xtick.color": Theme.TEXT_MUTED,
        "ytick.color": Theme.TEXT_MUTED,
        "xtick.bottom": True,
        "xtick.major.size": 4,
        "xtick.major.width": 0.5,
        "ytick.left": False,        
        "grid.color": Theme.GRID_COLOR,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    })


def format_header(fig, title, subtitle):
    """
    Add a left-aligned headline and subheadline to a figure.
    """
    fig.text(0.05, 0.98, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.05, 0.93, subtitle, fontsize=11, color=Theme.TEXT_MUTED, ha="left", va="top")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _align_dual_axes(ax_left, ax_right):
    """
    Align the y=0 line on two overlapping dual axes.

    Parameters
    ----------
    ax_left : Axes
        The left-side (primary) matplotlib axes.
    ax_right : Axes
        The right-side (secondary) matplotlib axes.

    Returns
    -------
    None
    """
    l_min, l_max = ax_left.get_ylim()
    r_min, r_max = ax_right.get_ylim()

    if l_max - l_min == 0 or r_max - r_min == 0:
        return

    l_ratio = abs(l_min) / (l_max - l_min)
    r_ratio = abs(r_min) / (r_max - r_min)

    ratio = max(l_ratio, r_ratio)

    l_range = l_max if l_max > 0 else abs(l_min)
    r_range = r_max if r_max > 0 else abs(r_min)

    ax_left.set_ylim(-ratio / (1 - ratio) * l_range, l_range)
    ax_right.set_ylim(-ratio / (1 - ratio) * r_range, r_range)


# ---------------------------------------------------------------------------
# Plot 1: Event Timeline
# ---------------------------------------------------------------------------
def plot_event_timeline(pm_swing, p538_swing):
    """
    Plot a dual-axis line chart tracking campaign swing-state
    averages for Polymarket and FiveThirtyEight over June
    through September 2024.

    Parameters
    ----------
    pm_swing : pd.Series
        Daily Polymarket swing-state average, datetime-indexed.
    p538_swing : pd.Series
        Daily FiveThirtyEight swing-state average,
        datetime-indexed.

    Returns
    -------
    None
        Saves figure to figures/events/event_timeline.png.
    """
    start_cutoff = pd.Timestamp("2024-06-01")
    end_cutoff = pd.Timestamp("2024-09-12")
    
    # Filter data to the June–September window
    pm_filtered = pm_swing[(pm_swing.index >= start_cutoff) & (pm_swing.index <= end_cutoff)]
    p538_filtered = p538_swing[(p538_swing.index >= start_cutoff) & (p538_swing.index <= end_cutoff)]
    fig, ax1 = plt.subplots(figsize=(15, 6.5))

    # Increased 'top' from 0.72 to 0.82 to decrease padding below title/subtitle
    fig.subplots_adjust(top=0.78, right=0.82)

    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(p538_filtered.index, p538_filtered.values, color=Theme.F38_COLOR,
             linewidth=2, label="FiveThirtyEight", zorder=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.plot(pm_filtered.index, pm_filtered.values, color=Theme.PM_COLOR,
             linewidth=2.5, label="Polymarket", zorder=4)

    _align_dual_axes(ax1, ax2)

    ax1.axhline(0, color=Theme.TEXT_MAIN, linewidth=1.2, zorder=2)

    start_date = pm_filtered.index.min()
    ax1.annotate("Trump leads", xy=(start_date, 0), xytext=(10, 8),
                 textcoords="offset points", color=Theme.TEXT_MUTED, fontsize=9, va="bottom")
    ax1.annotate("Biden / Harris leads", xy=(start_date, 0), xytext=(10, -8),
                 textcoords="offset points", color=Theme.TEXT_MUTED, fontsize=9, va="top")

    last_date = pm_filtered.index.max()

    ax1.annotate("Polymarket\n(Win Prob.)",
                 xy=(last_date, pm_filtered.iloc[-1]), xytext=(70, 12), textcoords="offset points",
                 color=Theme.PM_COLOR, fontweight="bold", va="center", annotation_clip=False)

    ax2.annotate("FiveThirtyEight\n(Vote Share %)",
                 xy=(last_date, p538_filtered.iloc[-1]), xytext=(70, -12), textcoords="offset points",
                 color=Theme.F38_COLOR, fontweight="bold", va="center", annotation_clip=False)

    _, y_max = ax1.get_ylim()

    for event in EVENTS:
        edate = pd.Timestamp(event["date"])
        
        # Skip events that fall outside our window
        if edate < start_cutoff or edate > end_cutoff:
            continue
            
        # Plot the faint background band
        ax1.axvspan(edate - pd.Timedelta(days=1), edate + pd.Timedelta(days=1), 
                    color=Theme.EVENT_BAND, zorder=1)
        
        # Horizontal alignment
        name = event["name"]
        if name in ["Biden-Trump Debate", "Walz VP Pick", "Harris-Trump Debate"]:
            level = 0
        elif name == "Assassination Attempt":
            level = 0
        elif name == "Biden Drops Out":
            level = 1
        else:
            level = 3  # Default fallback for any additional events
            
        # Stagger event labels above the chart box based on manually set level
        ax1.annotate(name, 
                     xy=(edate, y_max), 
                     xytext=(0, 6 + 14 * level),  # 6 pts above axis, +14 pts per level
                     textcoords="offset points",
                     ha="center", va="bottom", 
                     fontsize=8.5, color=Theme.TEXT_MAIN, 
                     annotation_clip=False)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b."))
    
    format_header(
        fig,
        "How Prediction Markets and Polls Tracked the Campaign",
        "Swing-state average Trump lead, showing Polymarket probability vs. FiveThirtyEight vote share."
    )

    fig.savefig(FIGURES_DIR / "event_timeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Event Scoreboard
# ---------------------------------------------------------------------------
def plot_reaction_scoreboard(summary):
    """
    Plot a horizontal bar chart comparing how each source
    reacted to campaign events, scored by direction and
    intensity (z-score).

    Parameters
    ----------
    summary : pd.DataFrame
        Reaction summary with columns event, source,
        shift_z, and direction_match.

    Returns
    -------
    None
        Saves figure to figures/events/event_scoreboard.png.
    """
    event_names = [e["name"] for e in EVENTS]
    y = np.arange(len(event_names))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    fig.subplots_adjust(top=0.78, left=0.25) 

    ax.axvline(0, color=Theme.TEXT_MAIN, linewidth=1, zorder=2)

    for i, event_name in enumerate(event_names):
        for source, color, offset in [
            ("Polymarket", Theme.PM_COLOR, -bar_h / 2),
            ("538", Theme.F38_COLOR, bar_h / 2),
        ]:
            row = summary[(summary["event"] == event_name) & (summary["source"] == source)]
            if row.empty or np.isnan(row.iloc[0]["shift_z"]):
                continue
            
            z = row.iloc[0]["shift_z"]
            is_correct = row.iloc[0]["direction_match"] == "correct"
            
            face_color = color if is_correct else "white"
            edge_color = color
            alpha = 0.9 if is_correct else 0.5

            ax.barh(y[i] + offset, z, height=bar_h * 0.85,
                    facecolor=face_color, edgecolor=edge_color, linewidth=1.5, alpha=alpha, zorder=3)

            align = "left" if z >= 0 else "right"
            label_x = z + (0.1 if z >= 0 else -0.1)
            ax.text(label_x, y[i] + offset, f"{z:+.1f}σ", ha=align, va="center", 
                    fontsize=8.5, color=color, fontweight="bold")

    x_min, x_max = ax.get_xlim()
    max_abs_x = max(abs(x_min), abs(x_max))
    ax.set_xlim(-max_abs_x, max_abs_x)

    # Axis Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(event_names)
    ax.invert_yaxis()
    ax.set_xlabel("Reaction Intensity (z-score, normalized to daily volatility)", color=Theme.TEXT_MUTED)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Custom Legend
    legend_elements = [
        Patch(facecolor=Theme.TEXT_MAIN, label="Moved in expected direction"),
        Patch(facecolor="white", edgecolor=Theme.TEXT_MAIN, linewidth=1.5, label="Moved in unexpected direction"),
        Line2D([0], [0], color=Theme.PM_COLOR, lw=4, label="Polymarket"),
        Line2D([0], [0], color=Theme.F38_COLOR, lw=4, label="FiveThirtyEight"),
    ]
    
    ax.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(1.0, 1.02),
              ncol=2, fontsize=9, frameon=False)

    format_header(
        fig,
        "Event Reaction Scoreboard",
        "Which source reacted more intensely, and did they move in the correct direction?"
    )

    fig.savefig(FIGURES_DIR / "event_scoreboard.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Event Dropout Response
# ---------------------------------------------------------------------------
def plot_indexed_event_study(pm_swing, p538_swing):
    """
    Plot an indexed event study of the Biden dropout,
    showing the lag between Polymarket and FiveThirtyEight
    reactions with a bracket annotation.

    Parameters
    ----------
    pm_swing : pd.Series
        Daily Polymarket swing-state average, datetime-indexed.
    p538_swing : pd.Series
        Daily FiveThirtyEight swing-state average,
        datetime-indexed.

    Returns
    -------
    None
        Saves figure to
        figures/events/event_dropout_response.png.
    """
    hero_event = next(e for e in EVENTS if e["name"] == "Biden Drops Out")
    edate = hero_event["date"]

    pm_win = compute_raw_indexed_window(pm_swing, edate, pre_days=1, scale=100)
    p538_win = compute_raw_indexed_window(p538_swing, edate, pre_days=1, scale=1)

    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    # Reset right padding since right-side labels are removed
    fig.subplots_adjust(top=0.82, right=0.95)
    ax1.grid(axis="y", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(p538_win.index, p538_win.values, color=Theme.F38_COLOR, linewidth=2.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.step(pm_win.index, pm_win.values, where="post", color=Theme.PM_COLOR, linewidth=2.5)

    _align_dual_axes(ax1, ax2)

    ax1.axvline(0, color=Theme.TEXT_MAIN, linewidth=1, linestyle=":", zorder=2)
    ax1.axhline(0, color=Theme.TEXT_MAIN, linewidth=1, zorder=2)
    ax1.text(0, ax1.get_ylim()[1]*0.95, " Biden withdraws", color=Theme.TEXT_MAIN, fontsize=9)

    legend_elements = [
        Line2D([0], [0], color=Theme.PM_COLOR, lw=3, label="Polymarket (Percentage points)"),
        Line2D([0], [0], color=Theme.F38_COLOR, lw=3, label="FiveThirtyEight (Vote share shift)")
    ]
    ax1.legend(handles=legend_elements, loc="lower left", frameon=False, fontsize=10, borderaxespad=1.5)

    # Use the day of the steepest drop to anchor the measurement
    pm_lag = pm_win.diff().idxmin()
    p538_lag = p538_win.diff().idxmin()

    if pd.notna(pm_lag) and pd.notna(p538_lag) and p538_lag > pm_lag:
        y_annot = pm_win.loc[10] * 0.35 if 10 in pm_win.index else 0
        
        # Horizontal Bracket
        ax1.annotate("", xy=(p538_lag, y_annot), xytext=(pm_lag, y_annot),
                     arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color=Theme.TEXT_MAIN, lw=1.2))
        
        # Bracket Text
        days_diff = int(p538_lag - pm_lag)
        ax1.annotate(f"{days_diff}-day lag",
                     xy=((pm_lag + p538_lag) / 2, y_annot), xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=Theme.TEXT_MAIN)

        # Draw vertical guide lines dropping down to actual curve points to ground the floating bracket
        if pm_lag in pm_win.index:
            pm_val = pm_win.loc[pm_lag]
            ax1.plot([pm_lag, pm_lag], [y_annot, pm_val], color=Theme.PM_COLOR, linestyle=":", lw=1.2, alpha=0.7)
            ax1.plot([pm_lag], [pm_val], marker='o', color=Theme.PM_COLOR, markersize=5)
            
        if p538_lag in p538_win.index:
            p538_val = p538_win.loc[p538_lag]
            # Must map ax1's y_annot level to ax2's mapped y-limits
            l_min, l_max = ax1.get_ylim()
            r_min, r_max = ax2.get_ylim()
            y_annot_ax2 = r_min + ((y_annot - l_min) / (l_max - l_min)) * (r_max - r_min)
            
            ax2.plot([p538_lag, p538_lag], [y_annot_ax2, p538_val], color=Theme.F38_COLOR, linestyle=":", lw=1.2, alpha=0.7)
            ax2.plot([p538_lag], [p538_val], marker='o', color=Theme.F38_COLOR, markersize=5)

    ax1.set_xlabel("Days from Event", color=Theme.TEXT_MUTED)

    format_header(
        fig,
        "Market Traders Spotted the Harris Surge a Week Before Polls",
        "Change in probability and vote share zeroed to the day before Biden dropped out."
    )

    fig.savefig(FIGURES_DIR / "event_dropout_response.png", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    """
    Load data, compute swing-state averages, and generate
    all three event analysis figures.

    Returns
    -------
    None
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("Loading data...")
    pm, p538 = load_data()

    print("Computing swing-state averages...")
    pm_swing = compute_swing_average(pm)
    p538_swing = compute_538_swing_timeseries(p538)

    print("Building reaction summary...")
    summary = build_reaction_summary(pm, p538)

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