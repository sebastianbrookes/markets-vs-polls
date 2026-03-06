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

from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Constants & Colors
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(PROJECT_ROOT) / "figures" / "accuracy"

OVERLAP_CUTOFF = "2024-09-12"
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"
ELECTION_EVE = (pd.Timestamp(ELECTION_DATE) - pd.Timedelta(days=1)).strftime(
    "%Y-%m-%d"
)

OVERLAP_STATES = [
    "AZ", "CA", "FL", "GA", "MI", "MN", "NC",
    "NH", "NV", "OH", "PA", "TX", "WI",
]
SWING_OVERLAP = sorted(set(OVERLAP_STATES) & set(SWING_STATES))

PERIODS = [
    ("Mar-May", "2024-03-01", "2024-05-31"),
    ("Jun-Jul", "2024-06-01", "2024-07-31"),
    ("Aug-Sep 12", "2024-08-01", "2024-09-12"),
]

# Color Palette
CLR_PM = "#12436D"
CLR_PM_LT = "#6B9BC3"
CLR_538 = "#D85E2A"
CLR_TRUMP = "#CC3A35"
CLR_HARRIS = "#20639B"
CLR_TEXT = "#333333"
CLR_SUBTEXT = "#666666"
CLR_GRID = "#EBEBEB"

DPI = 300


def configure_plot_style():
    """Apply a style baseline for all figures."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "sans-serif"],
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
# Data loading and prediction helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load Polymarket, FiveThirtyEight, and FEC results from processed CSVs."""
    pm = pd.read_csv(
        PROCESSED_DIR / "polymarket_daily.csv",
        parse_dates=["date"],
    )
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv", parse_dates=["date"])
    fec = pd.read_csv(PROCESSED_DIR / "fec_results.csv")
    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538, fec


def latest_per_state(p538, as_of):
    """Return the most recent 538 row per state on or before as_of."""
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


def predict_winner_pm(df):
    """Add predicted_winner column based on Polymarket trump_prob > 0.5."""
    df = df.copy()
    df["predicted_winner"] = np.where(df["trump_prob"] > 0.5, "Trump", "Harris")
    return df


def predict_winner_538(df):
    """Add predicted_winner column based on 538 trump_pct vs dem_pct."""
    df = df.copy()
    df["predicted_winner"] = np.where(
        df["trump_pct"] > df["dem_pct"],
        "Trump", "Harris"
    )
    return df


def pm_snapshot(pm, as_of, states=None):
    """Return Polymarket predictions for a single date,
    optionally filtered by states."""
    snapshot = predict_winner_pm(pm[pm["date"] == pd.Timestamp(as_of)])
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def p538_snapshot(p538, as_of, states=None):
    """Return 538 predictions as of a date, optionally filtered by states."""
    snapshot = predict_winner_538(latest_per_state(p538, as_of))
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def compute_accuracy(snapshot, fec_winners, states=None):
    """Compare predicted winners to actual results and return accuracy stats."""
    if states is None:
        subset = snapshot
    else:
        subset = snapshot[snapshot["state"].isin(states)]
    if subset.empty:
        return {"correct": 0, "n_states": 0, "pct": 0.0}

    predicted = (
        subset[["state", "predicted_winner"]]
        .drop_duplicates(subset="state", keep="last")
        .set_index("state")["predicted_winner"]
    )
    common_states = predicted.index.intersection(fec_winners.index)
    n_states = len(common_states)
    if n_states == 0:
        return {"correct": 0, "n_states": 0, "pct": 0.0}

    correct = int(
        (predicted.loc[common_states]
         == fec_winners.loc[common_states]).sum()
    )
    return {
        "correct": correct,
        "n_states": n_states,
        "pct": correct / n_states * 100,
    }


def compute_ev_share(snapshot, fec, states=None):
    """Compute Trump/Harris Electoral Vote share from predicted winners."""
    predicted = snapshot[["state", "predicted_winner"]].drop_duplicates(
        subset="state", keep="last"
    )
    merged = predicted.merge(
        fec[["state", "electoral_votes"]],
        on="state",
        how="inner",
    )
    if states is not None:
        merged = merged[merged["state"].isin(states)]

    total_ev = int(merged["electoral_votes"].sum())
    if total_ev == 0:
        return {
            "trump_pct": 0.0, "harris_pct": 0.0,
            "trump_ev": 0, "harris_ev": 0,
            "total_ev": 0,
        }

    trump_mask = merged["predicted_winner"] == "Trump"
    trump_ev = int(
        merged.loc[trump_mask, "electoral_votes"].sum()
    )
    harris_ev = total_ev - trump_ev
    trump_pct = trump_ev / total_ev * 100
    return {
        "trump_pct": trump_pct, "harris_pct": 100 - trump_pct,
        "trump_ev": trump_ev, "harris_ev": harris_ev, "total_ev": total_ev,
    }


# ---------------------------------------------------------------------------
# Reusable computation
# ---------------------------------------------------------------------------

def compute_daily_accuracy(pm, p538, fec):
    """
    Compute daily accuracy percentages for both
    sources over the overlap period.
    """
    start = max(pm["date"].min(), p538["date"].min())
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")
    fec_winners = fec.set_index("state")["winner"]

    records = []
    for as_of_date in dates:
        pm_day = pm[
            (pm["date"] == as_of_date)
            & (pm["state"].isin(OVERLAP_STATES))
        ]
        if pm_day.empty:
            continue
        pm_day = predict_winner_pm(pm_day)

        p5_day = latest_per_state(p538, as_of_date)
        p5_day = p5_day[p5_day["state"].isin(OVERLAP_STATES)]
        if p5_day.empty:
            continue
        p5_day = predict_winner_538(p5_day)

        common_states = sorted(
            set(pm_day["state"])
            & set(p5_day["state"])
            & set(fec_winners.index)
        )
        if not common_states:
            continue

        pm_stats = compute_accuracy(pm_day, fec_winners, common_states)
        p5_stats = compute_accuracy(p5_day, fec_winners, common_states)

        records.append({
            "date": as_of_date,
            "n_states": pm_stats["n_states"],
            "pm_correct": pm_stats["correct"],
            "p538_correct": p5_stats["correct"],
            "pm_pct": pm_stats["pct"],
            "p538_pct": p5_stats["pct"],
        })

    return pd.DataFrame(records)


def build_head_to_head_metrics(pm, p538, fec):
    """
    Build per-group accuracy comparison between
    Polymarket and 538 on Sept 12.
    """
    fec_winners = fec.set_index("state")["winner"]
    pm_snap = pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p538_snap = p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)

    groups = [
        ("All 13 States", OVERLAP_STATES),
        (f"{len(SWING_OVERLAP)} Swing States", SWING_OVERLAP),
    ]

    rows = []
    for group_label, states in groups:
        pm_stats = compute_accuracy(pm_snap, fec_winners, states)
        p538_stats = compute_accuracy(p538_snap, fec_winners, states)
        rows.append({"group": group_label,
                      "pm_pct": pm_stats["pct"],
                      "p538_pct": p538_stats["pct"]})
    return pd.DataFrame(rows)


def build_polymarket_trajectory_metrics(pm, fec):
    """Build Polymarket accuracy metrics at three date snapshots."""
    fec_winners = fec.set_index("state")["winner"]
    snapshots = [
        ("Sept 12", OVERLAP_CUTOFF, OVERLAP_STATES),
        ("Oct 6", THIRTY_DAYS_OUT, None),
        ("Nov 4", ELECTION_EVE, None),
    ]

    rows = []
    for label, as_of_date, state_subset in snapshots:
        snapshot = pm_snapshot(pm, as_of_date)
        if state_subset is None:
            states = sorted(snapshot["state"].unique())
        else:
            states = state_subset
        all_stats = compute_accuracy(snapshot, fec_winners, states)
        swing_states = sorted(set(SWING_STATES) & set(snapshot["state"]))
        swing_stats = compute_accuracy(snapshot, fec_winners, swing_states)

        rows.append({
            "label": f"{label}\n({all_stats['n_states']} states)",
            "all_pct": all_stats["pct"],
            "swing_pct": swing_stats["pct"],
        })

    return pd.DataFrame(rows)


def build_ev_comparison_metrics(pm, p538, fec):
    """
    Build Electoral Vote share metrics for
    Polymarket, 538, and actual results.
    """
    pm_snap = pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p538_snap = p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)
    actual = fec[["state", "winner"]].rename(
        columns={"winner": "predicted_winner"}
    )

    rows = [
        {"label": "Polymarket\n(Sept 12)",
         **compute_ev_share(pm_snap, fec, OVERLAP_STATES)},
        {"label": "538\n(Sept 12)",
         **compute_ev_share(p538_snap, fec, OVERLAP_STATES)},
        {"label": "Actual\nResult",
         **compute_ev_share(actual, fec)},
    ]
    return pd.DataFrame(rows)


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
    fig.text(0.04, title_y, title, fontsize=15,
             fontweight="bold", color=CLR_TEXT,
             ha="left")
    fig.text(0.04, subtitle_y, subtitle,
             fontsize=11, color=CLR_SUBTEXT,
             ha="left")


def annotate_bars(ax, bars, color, bold=False):
    """Add percentage labels to the top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.0f}%" if height % 1 == 0 else f"{height:.1f}%",
            ha="center", va="bottom",
            fontsize=9.5, color=color,
            fontweight="bold" if bold else "normal"
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
        "13 overlapping states available in both sources, Mar-Sep 2024."
    )

    ax.plot(ts["date"], ts["pm_pct"], color=CLR_PM,
            linewidth=2.5, label="Polymarket", zorder=4)
    ax.plot(ts["date"], ts["p538_pct"], color=CLR_538,
            linewidth=2.2, linestyle="--", dashes=(5, 3),
            label="FiveThirtyEight", zorder=4)

    # Subtle shading for specific periods
    shade_colors = ["#FFFFFF", "#F9FAFB", "#FFFFFF"]
    for i, (_, pstart, pend) in enumerate(PERIODS):
        ax.axvspan(
            pd.Timestamp(pstart), pd.Timestamp(pend),
            color=shade_colors[i], alpha=1, zorder=1,
        )

    # Event annotation with a dotted vertical line
    biden_date = pd.Timestamp("2024-07-21")
    ax.axvline(biden_date, color="#999999",
               linestyle=":", linewidth=1.2, zorder=3)
    ax.text(
        biden_date - pd.Timedelta(days=2), 43,
        "Biden drops out", fontsize=9.5,
        color=CLR_SUBTEXT, ha="right", va="bottom",
        style="italic", zorder=5,
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
    ax.legend(loc="upper left",
              bbox_to_anchor=(-0.12, 1.10), ncol=2,
              frameon=False, fontsize=10.5,
              borderaxespad=0)

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
        "available in both models on Sept 12."
    )

    bars_pm = ax.bar(x - width / 2, pm_vals, width,
                      label="Polymarket", color=CLR_PM,
                      zorder=3)
    bars_538 = ax.bar(x + width / 2, p538_vals, width,
                       label="FiveThirtyEight",
                       color=CLR_538, zorder=3)

    annotate_bars(ax, bars_pm, color=CLR_PM, bold=True)
    annotate_bars(ax, bars_538, color=CLR_538, bold=True)

    style_axis(ax)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    
    ax.legend(loc="upper left",
              bbox_to_anchor=(-0.12, 1.10), ncol=2,
              frameon=False, borderaxespad=0)

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
        subtitle="Accuracy improved progressively as the election approached."
    )

    bars_all = ax.bar(x - width / 2, all_vals, width,
                       label="Sample States",
                       color=CLR_PM, zorder=3)
    bars_sw = ax.bar(x + width / 2, swing_vals, width,
                      label="Swing States",
                      color=CLR_PM_LT, zorder=3)

    annotate_bars(ax, bars_all, color=CLR_PM, bold=True)
    annotate_bars(ax, bars_sw, color=CLR_PM_LT)

    style_axis(ax)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    
    ax.legend(loc="upper left",
              bbox_to_anchor=(-0.12, 1.10), ncol=2,
              frameon=False, borderaxespad=0)

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
        "based on Sept 12 model calls vs actual outcome."
    )

    actual_idx = len(labels) - 1
    
    for i in x:
        alpha = 1.0 if i == actual_idx else 0.85
        ax.bar(i, trump_pct[i], width,
               color=CLR_TRUMP, alpha=alpha, zorder=3)
        ax.bar(i, harris_pct[i], width,
               bottom=trump_pct[i], color=CLR_HARRIS,
               alpha=alpha, zorder=3)

        # White internal labels for the percentages
        ax.text(
            i, trump_pct[i] / 2,
            f"Trump\n{trump_pct[i]:.1f}%",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white", zorder=4,
        )
        ax.text(
            i, trump_pct[i] + harris_pct[i] / 2,
            f"Harris\n{harris_pct[i]:.1f}%",
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white", zorder=4,
        )

    # 50% line overlay
    ax.axhline(50, color="#111111", linewidth=1.2,
                linestyle=(0, (3, 2)), zorder=5)
    ax.text(len(labels) - 0.45, 50, "50%",
            fontsize=9, color="#111111",
            va="center", ha="right")

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
    pm, p538, fec = load_data()

    print("Computing metrics...")
    ts = compute_daily_accuracy(pm, p538, fec)
    head_to_head = build_head_to_head_metrics(pm, p538, fec)
    trajectory = build_polymarket_trajectory_metrics(pm, fec)
    ev = build_ev_comparison_metrics(pm, p538, fec)

    print("Generating plots...")
    plot_timeseries_crossover(ts)
    plot_head_to_head(head_to_head)
    plot_polymarket_trajectory(trajectory)
    plot_ev_comparison(ev)

    print("Done - 4 refined plots saved to figures/accuracy/")


if __name__ == "__main__":
    main()
