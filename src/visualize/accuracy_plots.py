"""
Visualize accuracy analysis findings: Polymarket vs. FiveThirtyEight
prediction accuracy for the 2024 U.S. presidential election.

Generates four plots:
    1. Time-series crossover (daily accuracy, Mar-Sep 12)
    2. Head-to-head snapshot (Sept 12, grouped bars)
    3. Polymarket trajectory (Sept 12 / Oct 6 / Nov 4)
    4. Electoral vote comparison (normalized %)

Run from project root:
    python -m src.visualize.accuracy_plots
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.accuracy.metrics import (
    ELECTION_EVE,
    OVERLAP_CUTOFF,
    OVERLAP_STATES,
    PERIODS,
    SWING_OVERLAP,
    THIRTY_DAYS_OUT,
    build_ev_comparison_metrics,
    build_head_to_head_metrics,
    build_polymarket_trajectory_metrics,
    compute_daily_accuracy,
    load_data,
)
from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Constants & Colors
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(PROJECT_ROOT) / "figures" / "accuracy"

CLR_PM = "#12436D"
CLR_PM_LT = "#6B9BC3"
CLR_538 = "#D85E2A"
CLR_TRUMP = "#CC3A35"
CLR_HARRIS = "#20639B"
CLR_TEXT = "#333333"
CLR_SUBTEXT = "#666666"
CLR_GRID = "#EBEBEB"

FONTS = ["Avenir Next", "Calibri", "sans-serif"]
DPI = 300


def configure_plot_style():
    """Apply a style baseline for all figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": FONTS,
            "figure.dpi": DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": CLR_TEXT,
            "axes.labelcolor": CLR_SUBTEXT,
            "xtick.color": CLR_SUBTEXT,
            "ytick.color": CLR_SUBTEXT,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#CCCCCC",
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def style_axis(ax, hide_x=False, hide_y=False):
    """Applies clean styles to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if hide_x:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    else:
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(axis="x", length=4, color="#CCCCCC", pad=6)

    if hide_y:
        ax.tick_params(axis="y", length=0, labelsize=0)
        ax.grid(axis="y", visible=False)
    else:
        ax.tick_params(axis="y", length=0, pad=4)
        ax.grid(axis="y", color=CLR_GRID, linewidth=1, linestyle="-", zorder=0)

    ax.set_axisbelow(True)


def add_titles(fig, title, subtitle, title_y=0.98, subtitle_y=0.92):
    """Adds left-aligned title and subtitle to the figure."""
    fig.text(
        0.04, title_y, title, fontsize=15, fontweight="bold", color=CLR_TEXT, ha="left"
    )
    fig.text(0.04, subtitle_y, subtitle, fontsize=11, color=CLR_SUBTEXT, ha="left")


def annotate_bars(ax, bars, color, bold=False):
    """Add percentage labels to the top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.0f}%" if height % 1 == 0 else f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=color,
            fontweight="bold" if bold else "normal",
        )


def save_figure(fig, filename, show=False):
    """Display or save a figure to the accuracy figures directory."""
    if show:
        plt.show()
    else:
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: Time-Series Crossover
# ---------------------------------------------------------------------------


def plot_timeseries_crossover(ts, show=False):
    """Plot daily Polymarket vs FiveThirtyEight accuracy over time."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(top=0.82)

    add_titles(
        fig,
        title="Polymarket Overtook FiveThirtyEight in State-Level Accuracy",
        subtitle="Tracking the daily prediction accuracy across "
        "13 overlapping states available in both sources, Mar-Sep 2024.",
    )

    ax.plot(
        ts["date"],
        ts["pm_pct"],
        color=CLR_PM,
        linewidth=2.5,
        label="Polymarket",
        zorder=4,
    )
    ax.plot(
        ts["date"],
        ts["p538_pct"],
        color=CLR_538,
        linewidth=2.2,
        linestyle="--",
        dashes=(5, 3),
        label="FiveThirtyEight",
        zorder=4,
    )

    # Subtle shading for specific periods
    shade_colors = ["#FFFFFF", "#F9FAFB", "#FFFFFF"]
    for i, (_, pstart, pend) in enumerate(PERIODS):
        ax.axvspan(
            pd.Timestamp(pstart),
            pd.Timestamp(pend),
            color=shade_colors[i],
            alpha=1,
            zorder=1,
        )

    # Event annotation with a dotted vertical line
    biden_date = pd.Timestamp("2024-07-21")
    ax.axvline(biden_date, color="#999999", linestyle=":", linewidth=1.2, zorder=3)
    ax.text(
        biden_date - pd.Timedelta(days=2),
        43,
        "Biden drops out",
        fontsize=9.5,
        color=CLR_SUBTEXT,
        ha="right",
        va="bottom",
        style="italic",
        zorder=5,
    )

    # Styling axes and limits
    style_axis(ax)
    ax.set_ylim(40, 105)

    # Custom y-tick formatting (only show % on the top tick)
    yticks = [40, 60, 80, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}" if y != 100 else "100%" for y in yticks])

    # Custom x-tick formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b."))

    # Legend horizontally placed at the top left
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.12, 1.10),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        borderaxespad=0,
    )

    save_figure(fig, "timeseries.png", show=show)


# ---------------------------------------------------------------------------
# Plot 2: Head-to-Head Snapshot (Sept 12)
# ---------------------------------------------------------------------------


def plot_head_to_head(metrics, show=False):
    """Plot grouped bar chart comparing per-state accuracy on Sept 12."""
    groups = metrics["group"].tolist()
    pm_vals = metrics["pm_pct"].to_numpy()
    p538_vals = metrics["p538_pct"].to_numpy()

    x = np.arange(len(groups))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(top=0.78)

    add_titles(
        fig,
        title="State-by-State Prediction Accuracy",
        subtitle="Snapshot of accuracy for states "
        "available in both models on Sept 12.",
    )

    bars_pm = ax.bar(
        x - width / 2, pm_vals, width, label="Polymarket", color=CLR_PM, zorder=3
    )
    bars_538 = ax.bar(
        x + width / 2,
        p538_vals,
        width,
        label="FiveThirtyEight",
        color=CLR_538,
        zorder=3,
    )

    annotate_bars(ax, bars_pm, color=CLR_PM, bold=True)
    annotate_bars(ax, bars_538, color=CLR_538, bold=True)

    style_axis(ax)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.12, 1.10),
        ncol=2,
        frameon=False,
        borderaxespad=0,
    )

    save_figure(fig, "head-to-head.png", show=show)


# ---------------------------------------------------------------------------
# Plot 3: Polymarket Trajectory (Sept 12 / Oct 6 / Nov 4)
# ---------------------------------------------------------------------------


def plot_polymarket_trajectory(metrics, show=False):
    """Plot Polymarket accuracy at three snapshots
    leading up to Election Day."""
    labels = metrics["label"].tolist()
    all_vals = metrics["all_pct"].to_numpy()
    swing_vals = metrics["swing_pct"].to_numpy()

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.subplots_adjust(top=0.78)

    final_pct = float(all_vals[-1]) if len(all_vals) else 0.0
    add_titles(
        fig,
        title=f"Polymarket Reached {final_pct:.0f}% Accuracy by Election Eve",
        subtitle="Accuracy improved progressively as the election approached.",
    )

    bars_all = ax.bar(
        x - width / 2, all_vals, width, label="Sample States", color=CLR_PM, zorder=3
    )
    bars_sw = ax.bar(
        x + width / 2,
        swing_vals,
        width,
        label="Swing States",
        color=CLR_PM_LT,
        zorder=3,
    )

    annotate_bars(ax, bars_all, color=CLR_PM, bold=True)
    annotate_bars(ax, bars_sw, color=CLR_PM_LT)

    style_axis(ax)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.12, 1.10),
        ncol=2,
        frameon=False,
        borderaxespad=0,
    )

    save_figure(fig, "pm-trajectory.png", show=show)


# ---------------------------------------------------------------------------
# Plot 4: Electoral Vote Comparison (normalized %)
# ---------------------------------------------------------------------------


def plot_ev_comparison(metrics, show=False):
    """Plot stacked bars comparing predicted vs actual Electoral Vote splits."""
    labels = metrics["label"].tolist()
    trump_pct = metrics["trump_pct"].to_numpy()
    harris_pct = metrics["harris_pct"].to_numpy()

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.subplots_adjust(top=0.82)

    add_titles(
        fig,
        title="Polymarket More Closely Reflected the Electoral Split",
        subtitle="Share of available Electoral Votes "
        "based on Sept 12 model calls vs actual outcome.",
    )

    actual_idx = len(labels) - 1

    for i in x:
        alpha = 1.0 if i == actual_idx else 0.85
        ax.bar(i, trump_pct[i], width, color=CLR_TRUMP, alpha=alpha, zorder=3)
        ax.bar(
            i,
            harris_pct[i],
            width,
            bottom=trump_pct[i],
            color=CLR_HARRIS,
            alpha=alpha,
            zorder=3,
        )

        # White internal labels for the percentages
        ax.text(
            i,
            trump_pct[i] / 2,
            f"Trump\n{trump_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        ax.text(
            i,
            trump_pct[i] + harris_pct[i] / 2,
            f"Harris\n{harris_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # 50% line overlay
    ax.axhline(50, color="#111111", linewidth=1.2, linestyle=(0, (3, 2)), zorder=5)
    ax.text(
        len(labels) - 0.45,
        50,
        "50%",
        fontsize=9,
        color="#111111",
        va="center",
        ha="right",
    )

    style_axis(ax, hide_y=True)
    ax.set_ylim(0, 100)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)

    save_figure(fig, "ev-comparison.png", show=show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Load data, compute metrics, and generate all four accuracy plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    print("Loading data...")
    pm, p538, fec = load_data(PROCESSED_DIR)

    print("Computing metrics...")
    ts = compute_daily_accuracy(pm, p538, fec, OVERLAP_CUTOFF, OVERLAP_STATES)
    groups = [
        ("All 13 States", OVERLAP_STATES),
        (
            f"{len(SWING_OVERLAP)} Swing States",
            SWING_OVERLAP,
        ),
    ]
    head_to_head = build_head_to_head_metrics(pm, p538, fec, OVERLAP_CUTOFF, groups)
    snapshots = [
        ("Sept 12", OVERLAP_CUTOFF, OVERLAP_STATES),
        ("Oct 6", THIRTY_DAYS_OUT, None),
        ("Nov 4", ELECTION_EVE, None),
    ]
    trajectory = build_polymarket_trajectory_metrics(pm, fec, snapshots, SWING_STATES)
    ev = build_ev_comparison_metrics(pm, p538, fec, OVERLAP_CUTOFF, OVERLAP_STATES)

    print("Generating plots...")
    plot_timeseries_crossover(ts)
    plot_head_to_head(head_to_head)
    plot_polymarket_trajectory(trajectory)
    plot_ev_comparison(ev)

    print("Done - 4 plots saved to figures/accuracy/")


if __name__ == "__main__":
    main()
