"""
Visualize event response analysis: how Polymarket and FiveThirtyEight
reacted to major 2024 campaign events.

Generates two plots:
    1. Event timeline — full March-Sept 12 with event bands
    2. Reaction scoreboard — grouped horizontal bars per event

Run from project root:
    python -m src.visualize.event_plots
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    build_reaction_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(_project_root) / "figures" / "events"

CLR_PM = "#0072B2"
CLR_538 = "#D55E00"
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

    # Zero line
    ax.axvline(0, color="#333333", linewidth=1, zorder=2)

    # Direction labels
    ax.text(ax.get_xlim()[1], -0.8, "Pro-Trump \u2192",
            ha="right", fontsize=9, color="#888888", fontstyle="italic")
    ax.text(ax.get_xlim()[0], -0.8, "\u2190 Pro-Harris",
            ha="left", fontsize=9, color="#888888", fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(event_names)
    ax.invert_yaxis()
    ax.set_xlabel("Reaction Intensity (z-score within source)")
    ax.set_title(
        "Event Reaction Scoreboard: Who Moved in the Right Direction?\n"
        "Normalized to each source\u2019s daily volatility \u2014 "
        "green border = correct, red = wrong",
        fontweight="bold",
    )

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

    print("Done — 2 plots saved to figures/events/")


if __name__ == "__main__":
    main()
