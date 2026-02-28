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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(_project_root) / "figures" / "accuracy"

OVERLAP_CUTOFF = "2024-09-12"
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"
ELECTION_EVE = (pd.Timestamp(ELECTION_DATE) - pd.Timedelta(days=1)).strftime(
    "%Y-%m-%d"
)

OVERLAP_STATES = [
    "AZ",
    "CA",
    "FL",
    "GA",
    "MI",
    "MN",
    "NC",
    "NH",
    "NV",
    "OH",
    "PA",
    "TX",
    "WI",
]
SWING_OVERLAP = sorted(set(OVERLAP_STATES) & set(SWING_STATES))

PERIODS = [
    ("Mar-May", "2024-03-01", "2024-05-31"),
    ("Jun-Jul", "2024-06-01", "2024-07-31"),
    ("Aug-Sep 12", "2024-08-01", "2024-09-12"),
]


CLR_PM = "#004276"
CLR_538 = "#D95F02"
CLR_TRUMP = "#C53030"
CLR_HARRIS = "#2B6CB0"

DPI = 300


def _configure_plot_style():
    """Apply a consistent style baseline for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "figure.dpi": DPI,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


# ---------------------------------------------------------------------------
# Data loading and prediction helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load processed source files used for all accuracy visualizations."""
    pm = pd.read_csv(PROCESSED_DIR / "polymarket_daily.csv", parse_dates=["date"])
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv", parse_dates=["date"])
    fec = pd.read_csv(PROCESSED_DIR / "fec_results.csv")
    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538, fec


def _latest_per_state(p538, as_of):
    """Forward-fill 538 data by taking latest available value per state."""
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


def _predict_winner_pm(df):
    df = df.copy()
    df["predicted_winner"] = np.where(df["trump_prob"] > 0.5, "Trump", "Harris")
    return df


def _predict_winner_538(df):
    df = df.copy()
    df["predicted_winner"] = np.where(df["trump_pct"] > df["dem_pct"], "Trump", "Harris")
    return df


def _pm_snapshot(pm, as_of, states=None):
    snapshot = _predict_winner_pm(pm[pm["date"] == pd.Timestamp(as_of)])
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def _p538_snapshot(p538, as_of, states=None):
    snapshot = _predict_winner_538(_latest_per_state(p538, as_of))
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def _compute_accuracy(snapshot, fec_winners, states=None):
    """Return dict with winner-call accuracy for a prediction snapshot."""
    subset = snapshot if states is None else snapshot[snapshot["state"].isin(states)]
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

    correct = int((predicted.loc[common_states] == fec_winners.loc[common_states]).sum())
    return {"correct": correct, "n_states": n_states, "pct": correct / n_states * 100}


def _compute_ev_share(snapshot, fec, states=None):
    """Summarize EV split implied by model predictions."""
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
            "trump_pct": 0.0,
            "harris_pct": 0.0,
            "trump_ev": 0,
            "harris_ev": 0,
            "total_ev": 0,
        }

    trump_ev = int(merged.loc[merged["predicted_winner"] == "Trump", "electoral_votes"].sum())
    harris_ev = total_ev - trump_ev
    trump_pct = trump_ev / total_ev * 100
    return {
        "trump_pct": trump_pct,
        "harris_pct": 100 - trump_pct,
        "trump_ev": trump_ev,
        "harris_ev": harris_ev,
        "total_ev": total_ev,
    }


# ---------------------------------------------------------------------------
# Reusable computation
# ---------------------------------------------------------------------------

def compute_daily_accuracy(pm, p538, fec):
    """Return daily winner-call accuracy for both sources through Sept 12."""
    start = max(pm["date"].min(), p538["date"].min())
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")
    fec_winners = fec.set_index("state")["winner"]

    records = []
    for as_of_date in dates:
        pm_day = pm[(pm["date"] == as_of_date) & (pm["state"].isin(OVERLAP_STATES))]
        if pm_day.empty:
            continue
        pm_day = _predict_winner_pm(pm_day)

        p5_day = _latest_per_state(p538, as_of_date)
        p5_day = p5_day[p5_day["state"].isin(OVERLAP_STATES)]
        if p5_day.empty:
            continue
        p5_day = _predict_winner_538(p5_day)

        common_states = sorted(
            set(pm_day["state"]) & set(p5_day["state"]) & set(fec_winners.index)
        )
        if not common_states:
            continue

        pm_stats = _compute_accuracy(pm_day, fec_winners, common_states)
        p5_stats = _compute_accuracy(p5_day, fec_winners, common_states)

        records.append(
            {
                "date": as_of_date,
                "n_states": pm_stats["n_states"],
                "pm_correct": pm_stats["correct"],
                "p538_correct": p5_stats["correct"],
                "pm_pct": pm_stats["pct"],
                "p538_pct": p5_stats["pct"],
            }
        )

    return pd.DataFrame(records)


def build_head_to_head_metrics(pm, p538, fec):
    """Compute Sept 12 grouped-bar values from source data."""
    fec_winners = fec.set_index("state")["winner"]

    pm_snap = _pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p538_snap = _p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)

    groups = [
        ("All 13 States", OVERLAP_STATES),
        (f"{len(SWING_OVERLAP)} Swing States", SWING_OVERLAP),
    ]

    rows = []
    for group_label, states in groups:
        pm_stats = _compute_accuracy(pm_snap, fec_winners, states)
        p538_stats = _compute_accuracy(p538_snap, fec_winners, states)
        rows.append(
            {
                "group": group_label,
                "pm_pct": pm_stats["pct"],
                "p538_pct": p538_stats["pct"],
            }
        )
    return pd.DataFrame(rows)


def build_polymarket_trajectory_metrics(pm, fec):
    """Compute Polymarket snapshot accuracy values for bar chart."""
    fec_winners = fec.set_index("state")["winner"]
    snapshots = [
        ("Sept 12", OVERLAP_CUTOFF, OVERLAP_STATES),
        ("Oct 6", THIRTY_DAYS_OUT, None),
        ("Nov 4", ELECTION_EVE, None),
    ]

    rows = []
    for label, as_of_date, state_subset in snapshots:
        snapshot = _pm_snapshot(pm, as_of_date)
        states = (
            sorted(snapshot["state"].unique())
            if state_subset is None
            else state_subset
        )

        all_stats = _compute_accuracy(snapshot, fec_winners, states)
        swing_states = sorted(set(SWING_STATES) & set(snapshot["state"]))
        swing_stats = _compute_accuracy(snapshot, fec_winners, swing_states)

        rows.append(
            {
                "label": f"{label}\n({all_stats['n_states']} states)",
                "all_pct": all_stats["pct"],
                "swing_pct": swing_stats["pct"],
            }
        )

    return pd.DataFrame(rows)


def build_ev_comparison_metrics(pm, p538, fec):
    """Compute EV share bars from prediction snapshots."""
    pm_snap = _pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p538_snap = _p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)

    actual = fec[["state", "winner"]].rename(columns={"winner": "predicted_winner"})

    rows = [
        {"label": "Polymarket\nSept 12", **_compute_ev_share(pm_snap, fec, OVERLAP_STATES)},
        {"label": "538\nSept 12", **_compute_ev_share(p538_snap, fec, OVERLAP_STATES)},
        {"label": "Actual\nResult", **_compute_ev_share(actual, fec)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _style_axis(ax, y_grid_alpha=0.3):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=y_grid_alpha)
    ax.grid(axis="x", visible=False)


def _annotate_bars(ax, bars, color, y_offset=1.5, bold=False):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold" if bold else "normal",
        )


def _save_figure(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 1: Time-Series Crossover
# ---------------------------------------------------------------------------

def plot_timeseries_crossover(ts):
    """Single-axis line chart with period shading and event marker."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        ts["date"],
        ts["pm_pct"],
        color=CLR_PM,
        linewidth=2.5,
        label="Polymarket",
        zorder=3,
    )
    ax.plot(
        ts["date"],
        ts["p538_pct"],
        color=CLR_538,
        linewidth=2.5,
        linestyle="--",
        dashes=(6, 3),
        label="FiveThirtyEight",
        zorder=3,
    )

    shade_colors = ["#FFFFFF", "#F4F4F6", "#FFFFFF"]
    for i, (label, pstart, pend) in enumerate(PERIODS):
        pstart_ts = pd.Timestamp(pstart)
        pend_ts = pd.Timestamp(pend)
        ax.axvspan(pstart_ts, pend_ts, color=shade_colors[i], alpha=0.6, zorder=0)

        sub = ts[(ts["date"] >= pstart_ts) & (ts["date"] <= pend_ts)]
        if sub.empty:
            continue
        mid_x = pstart_ts + (pend_ts - pstart_ts) / 2
        ax.text(mid_x, 105, label, ha="center", va="bottom", fontsize=9, color="#888888")

    biden_date = pd.Timestamp("2024-07-21")
    ax.axvline(
        biden_date,
        color="#555555",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        zorder=2,
    )
    ax.text(
        pd.Timestamp("2024-07-24"),
        45,
        "Biden drops out",
        fontsize=9,
        color="#555555",
        fontstyle="italic",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        zorder=4,
    )

    _style_axis(ax, y_grid_alpha=0.4)
    ax.set_ylim(30, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_ylabel("States Predicted Correctly (%)")
    ax.set_xlabel("Date (2024)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(
        "Polymarket Overtook FiveThirtyEight in State-Level Accuracy\n"
        "13 overlap states, Mar-Sep 12 2024",
        pad=18,
        fontweight="bold",
    )
    ax.legend(loc="lower left", frameon=False, fontsize=11)

    fig.tight_layout()
    _save_figure(fig, "timeseries.png")


# ---------------------------------------------------------------------------
# Plot 2: Head-to-Head Snapshot (Sept 12)
# ---------------------------------------------------------------------------

def plot_head_to_head(metrics):
    """Grouped bar chart: Polymarket vs 538 on Sept 12."""
    groups = metrics["group"].tolist()
    pm_vals = metrics["pm_pct"].to_numpy()
    p538_vals = metrics["p538_pct"].to_numpy()

    x = np.arange(len(groups))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_pm = ax.bar(x - width / 2, pm_vals, width, label="Polymarket", color=CLR_PM)
    bars_538 = ax.bar(
        x + width / 2,
        p538_vals,
        width,
        label="FiveThirtyEight",
        color=CLR_538,
    )

    _annotate_bars(ax, bars_pm, color=CLR_PM, bold=True)
    _annotate_bars(ax, bars_538, color=CLR_538, bold=True)

    ax.set_ylim(0, 105)
    ax.set_ylabel("States Predicted Correctly (%)")
    ax.set_title(
        "State-by-State Prediction Accuracy",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(frameon=False)
    _style_axis(ax)

    fig.tight_layout()
    _save_figure(fig, "head-to-head.png")


# ---------------------------------------------------------------------------
# Plot 3: Polymarket Trajectory (Sept 12 / Oct 6 / Nov 4)
# ---------------------------------------------------------------------------

def plot_polymarket_trajectory(metrics):
    """Grouped bars: Polymarket accuracy across three snapshots."""
    labels = metrics["label"].tolist()
    all_vals = metrics["all_pct"].to_numpy()
    swing_vals = metrics["swing_pct"].to_numpy()

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_all = ax.bar(x - width / 2, all_vals, width, label="Sample States", color=CLR_PM)
    bars_sw = ax.bar(
        x + width / 2,
        swing_vals,
        width,
        label="Swing States",
        color=CLR_PM,
        alpha=0.55,
    )

    _annotate_bars(ax, bars_all, color=CLR_PM, bold=True)
    _annotate_bars(ax, bars_sw, color=CLR_PM)

    final_pct = float(all_vals[-1]) if len(all_vals) else 0.0
    ax.set_ylim(0, 115)
    ax.set_ylabel("States Predicted Correctly (%)")
    ax.set_title(
        f"Polymarket Reached {final_pct:.0f}% Accuracy by Election Eve",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    _style_axis(ax)

    fig.tight_layout()
    _save_figure(fig, "pm-trajectory.png")


# ---------------------------------------------------------------------------
# Plot 4: Electoral Vote Comparison (normalized %)
# ---------------------------------------------------------------------------

def plot_ev_comparison(metrics):
    """Stacked bar chart showing Trump/Harris EV share by scenario."""
    labels = metrics["label"].tolist()
    trump_pct = metrics["trump_pct"].to_numpy()
    harris_pct = metrics["harris_pct"].to_numpy()

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, trump_pct, width, color=CLR_TRUMP)
    ax.bar(x, harris_pct, width, bottom=trump_pct, color=CLR_HARRIS)

    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":", zorder=3)
    ax.text(len(labels) - 0.5, 51, "50%", fontsize=8, color="gray", va="bottom")

    actual_idx = len(labels) - 1
    ax.axvspan(actual_idx - 0.5, actual_idx + 0.5, color="#E8E8E8", alpha=0.5, zorder=0)

    for i in range(len(labels)):
        ax.text(
            x[i],
            trump_pct[i] / 2,
            f"Trump {trump_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x[i],
            trump_pct[i] + harris_pct[i] / 2,
            f"Harris {harris_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of Available Electoral Votes (%)")
    ax.set_title("Polymarket Came Closer to the Actual Electoral Vote Split", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _style_axis(ax)

    fig.tight_layout()
    _save_figure(fig, "ev-comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

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

    print("Done - 4 plots saved to figures/accuracy/")


if __name__ == "__main__":
    main()
