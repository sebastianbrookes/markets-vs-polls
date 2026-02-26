"""
Visualize accuracy analysis findings: Polymarket vs. FiveThirtyEight
prediction accuracy for the 2024 U.S. presidential election.

Generates four plots:
    1. Time-series crossover (daily accuracy, Mar–Sep 12)
    2. Head-to-head snapshot (Sept 12, grouped bars)
    3. Polymarket trajectory (Sept 12 / Oct 6 / Nov 4)
    4. Electoral vote comparison (normalized %)

Run from project root:
    python -m src.visualize.accuracy_plots
"""

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
FIGURES_DIR = Path(_project_root) / "figures"

OVERLAP_CUTOFF = "2024-09-12"
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"

OVERLAP_STATES = [
    "AZ", "CA", "FL", "GA", "MI", "MN", "NC", "NH", "NV", "OH", "PA",
    "TX", "WI",
]
SWING_OVERLAP = sorted(set(OVERLAP_STATES) & set(SWING_STATES))

# Periods for the time-series shading
PERIODS = [
    ("Mar–May", "2024-03-01", "2024-05-31"),
    ("Jun–Jul", "2024-06-01", "2024-07-31"),
    ("Aug–Sep 12", "2024-08-01", "2024-09-12"),
]

# Colors — colorblind-friendly palette (Okabe-Ito)
CLR_PM = "#0072B2"       # deep blue for Polymarket
CLR_538 = "#D55E00"      # vermillion for FiveThirtyEight
CLR_SHADE = ["#e2e8f0", "#cbd5e0", "#a0aec0"]  # light-to-medium grays
# Political colors (for EV chart)
CLR_TRUMP = "#c53030"
CLR_HARRIS = "#2b6cb0"

DPI = 200


# ---------------------------------------------------------------------------
# Data loading (mirrors accuracy.py)
# ---------------------------------------------------------------------------

def load_data():
    pm = pd.read_csv(PROCESSED_DIR / "polymarket_daily.csv",
                     parse_dates=["date"])
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv",
                       parse_dates=["date"])
    fec = pd.read_csv(PROCESSED_DIR / "fec_results.csv")
    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538, fec


def _latest_per_state(p538, as_of):
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


def _predict_winner_pm(df):
    df = df.copy()
    df["predicted_winner"] = np.where(
        df["trump_prob"] > 0.5, "Trump", "Harris"
    )
    return df


def _predict_winner_538(df):
    df = df.copy()
    df["predicted_winner"] = np.where(
        df["trump_pct"] > df["dem_pct"], "Trump", "Harris"
    )
    return df


# ---------------------------------------------------------------------------
# Reusable computation: daily accuracy time-series
# ---------------------------------------------------------------------------

def compute_daily_accuracy(pm, p538, fec):
    """Return a DataFrame with daily accuracy % for both sources."""
    pm_start = pm["date"].min()
    p538_start = p538["date"].min()
    start = max(pm_start, p538_start)
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")
    fec_winners = fec.set_index("state")["winner"]

    records = []
    for d in dates:
        pm_day = pm[pm["date"] == d]
        pm_day = pm_day[pm_day["state"].isin(OVERLAP_STATES)]
        if pm_day.empty:
            continue
        pm_day = _predict_winner_pm(pm_day)

        p5_day = _latest_per_state(p538, d)
        p5_day = p5_day[p5_day["state"].isin(OVERLAP_STATES)]
        if p5_day.empty:
            continue
        p5_day = _predict_winner_538(p5_day)

        common = sorted(set(pm_day["state"]) & set(p5_day["state"]))
        if not common:
            continue

        pm_correct = sum(
            pm_day.loc[pm_day["state"] == s, "predicted_winner"].iloc[0]
            == fec_winners.get(s, "")
            for s in common
        )
        p5_correct = sum(
            p5_day.loc[p5_day["state"] == s, "predicted_winner"].iloc[0]
            == fec_winners.get(s, "")
            for s in common
        )
        records.append({
            "date": d,
            "n_states": len(common),
            "pm_correct": pm_correct,
            "p538_correct": p5_correct,
            "pm_pct": pm_correct / len(common) * 100,
            "p538_pct": p5_correct / len(common) * 100,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot 1: Time-Series Crossover
# ---------------------------------------------------------------------------

def plot_timeseries_crossover(ts):
    """Single-axis chart with integrated period shading and annotations."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Data lines ---
    ax.plot(ts["date"], ts["pm_pct"], color=CLR_PM, linewidth=2.5,
            label="Polymarket", zorder=3)
    ax.plot(ts["date"], ts["p538_pct"], color=CLR_538, linewidth=2.5,
            linestyle="--", dashes=(6, 3), label="FiveThirtyEight", zorder=3)

    # --- Integrated period shading & annotations ---
    shade_colors = ["#ffffff", "#f4f4f6", "#ffffff"]
    for i, (label, pstart, pend) in enumerate(PERIODS):
        pstart_ts = pd.Timestamp(pstart)
        pend_ts = pd.Timestamp(pend)
        ax.axvspan(pstart_ts, pend_ts, color=shade_colors[i],
                   alpha=0.6, zorder=0)

        mask = (ts["date"] >= pstart_ts) & (ts["date"] <= pend_ts)
        sub = ts[mask]
        if sub.empty:
            continue
        pm_mean = sub["pm_pct"].mean()
        p5_mean = sub["p538_pct"].mean()
        mid_x = pstart_ts + (pend_ts - pstart_ts) / 2


    # --- Biden dropout marker (July 21, 2024) ---
    biden_date = pd.Timestamp("2024-07-21")
    ax.axvline(biden_date, color="#555555", linestyle=":", linewidth=1.5,
               alpha=0.8, zorder=2)
    ax.text(pd.Timestamp("2024-07-24"), 45, "Biden drops out",
            fontsize=9, color="#555555", fontstyle="italic",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            zorder=4)

    # --- Spines ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Grid ---
    ax.grid(axis="y", alpha=0.4, linewidth=0.6)
    ax.grid(axis="x", visible=False)

    # --- Axes ---
    ax.set_ylim(30, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}%"))
    ax.set_ylabel("States Predicted Correctly")
    ax.set_xlabel("Date (2024)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # --- Title ---
    ax.set_title(
        "Daily Prediction Accuracy: Polymarket vs. FiveThirtyEight\n"
        "(13 overlap states, Mar\u2013Sep 12 2024)",
        pad=18, fontsize=13,
    )

    # --- Legend (explicit handles to guarantee correct style) ---
    handle_pm = mlines.Line2D([], [], color=CLR_PM, linewidth=2.5,
                              linestyle="-", label="Polymarket")
    handle_538 = mlines.Line2D([], [], color=CLR_538, linewidth=2.5,
                               linestyle="--", dashes=(6, 3),
                               label="FiveThirtyEight")
    ax.legend(handles=[handle_pm, handle_538], loc="lower left",
              frameon=False, fontsize=11)

    fig.tight_layout()
    path = FIGURES_DIR / "accuracy_timeseries.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    plt.style.use("default")
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Head-to-Head Snapshot (Sept 12)
# ---------------------------------------------------------------------------

def plot_head_to_head():
    """Grouped bar chart: Polymarket vs 538, all states + swing states."""
    groups = ["All 13 States", "7 Swing States"]
    pm_vals = [84.6, 71.4]
    p538_vals = [61.5, 28.6]

    x = np.arange(len(groups))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_pm = ax.bar(x - width / 2, pm_vals, width, label="Polymarket",
                     color=CLR_PM)
    bars_538 = ax.bar(x + width / 2, p538_vals, width,
                      label="FiveThirtyEight", color=CLR_538)

    # Label bars
    for bar in bars_pm:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=CLR_PM)
    for bar in bars_538:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=CLR_538)

    ax.set_ylim(0, 105)
    ax.set_ylabel("States Predicted Correctly (%)")
    ax.set_title("Winner Prediction Accuracy (Sept 12, 2024)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "accuracy_head_to_head.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Polymarket Trajectory (Sept 12 / Oct 6 / Nov 4)
# ---------------------------------------------------------------------------

def plot_polymarket_trajectory():
    """Grouped bars: Polymarket accuracy at three snapshots."""
    snapshots = [
        "Sept 12\n(13 states)",
        "Oct 6\n(50 states)",
        "Nov 4\n(50 states)",
    ]
    all_vals = [84.6, 94.0, 96.0]
    swing_vals = [71.4, 57.1, 71.4]

    x = np.arange(len(snapshots))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_all = ax.bar(x - width / 2, all_vals, width,
                      label="All States", color=CLR_PM)
    bars_sw = ax.bar(x + width / 2, swing_vals, width,
                     label="Swing States", color=CLR_PM, alpha=0.55)

    for bar in bars_all:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for bar in bars_sw:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=10)

    ax.set_ylim(0, 115)
    ax.set_ylabel("States Predicted Correctly (%)")
    ax.set_title("Polymarket Prediction Accuracy Over Time")
    ax.set_xticks(x)
    ax.set_xticklabels(snapshots)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "accuracy_pm_trajectory.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Plot 4: Electoral Vote Comparison (normalized %)
# ---------------------------------------------------------------------------

def plot_ev_comparison():
    """Grouped bar chart showing Trump EV share (%) across scenarios."""
    labels = [
        "Polymarket\nSept 12",
        "538\nSept 12",
        "Actual\nResult",
    ]
    trump_ev = [155, 114, 312]
    total_ev = [248, 248, 538]
    trump_pct = [t / tot * 100 for t, tot in zip(trump_ev, total_ev)]
    harris_pct = [100 - p for p in trump_pct]

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))

    # Stacked bars: Trump (red) on bottom, Harris (blue) on top
    bars_t = ax.bar(x, trump_pct, width, color=CLR_TRUMP)
    bars_h = ax.bar(x, harris_pct, width, bottom=trump_pct,
                    color=CLR_HARRIS)

    # 50% reference line
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":", zorder=3)
    ax.text(len(labels) - 0.5, 51, "50%", fontsize=8, color="gray",
            va="bottom")

    # Neutral background highlight behind the "Actual Result" column
    actual_idx = len(labels) - 1
    ax.axvspan(actual_idx - 0.5, actual_idx + 0.5,
               color="#E8E8E8", alpha=0.5, zorder=0)

    # Annotate with EV percentages
    for i in range(len(labels)):
        ax.text(x[i], trump_pct[i] / 2, f"Trump {trump_pct[i]:.1f}%",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
        ax.text(x[i], trump_pct[i] + harris_pct[i] / 2,
                f"Harris {harris_pct[i]:.1f}%",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of Available Electoral Votes (%)")
    ax.set_title("Electoral Vote Predictions vs. Actual Result")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "accuracy_ev_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pm, p538, fec = load_data()

    print("Computing daily accuracy time-series...")
    ts = compute_daily_accuracy(pm, p538, fec)

    print("Generating plots...")
    plot_timeseries_crossover(ts)
    plot_head_to_head()
    plot_polymarket_trajectory()
    plot_ev_comparison()

    print("Done — 4 plots saved to figures/")


if __name__ == "__main__":
    main()
