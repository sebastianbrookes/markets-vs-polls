"""
Computation engine for the accuracy analysis: Polymarket vs. FiveThirtyEight
prediction accuracy for the 2024 U.S. presidential election.

All data loading, prediction logic, accuracy scoring, and metric building
lives here.  Visualization code imports from this module — not the other
way around.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAP_CUTOFF = "2024-09-12"
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"
ELECTION_EVE = (
    pd.Timestamp(ELECTION_DATE) - pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load Polymarket, FiveThirtyEight, and FEC results from processed CSVs."""
    pm = pd.read_csv(
        PROCESSED_DIR / "polymarket_daily.csv", parse_dates=["date"]
    )
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv", parse_dates=["date"])
    fec = pd.read_csv(PROCESSED_DIR / "fec_results.csv")
    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538, fec


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def latest_per_state(p538, as_of):
    """Return the most recent 538 row per state on or before *as_of*."""
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
        df["trump_pct"] > df["dem_pct"], "Trump", "Harris"
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


# ---------------------------------------------------------------------------
# Accuracy scoring
# ---------------------------------------------------------------------------

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
        (predicted.loc[common_states] == fec_winners.loc[common_states]).sum()
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
        fec[["state", "electoral_votes"]], on="state", how="inner"
    )
    if states is not None:
        merged = merged[merged["state"].isin(states)]

    total_ev = int(merged["electoral_votes"].sum())
    if total_ev == 0:
        return {
            "trump_pct": 0.0, "harris_pct": 0.0,
            "trump_ev": 0, "harris_ev": 0, "total_ev": 0,
        }

    trump_mask = merged["predicted_winner"] == "Trump"
    trump_ev = int(merged.loc[trump_mask, "electoral_votes"].sum())
    harris_ev = total_ev - trump_ev
    trump_pct = trump_ev / total_ev * 100
    return {
        "trump_pct": trump_pct, "harris_pct": 100 - trump_pct,
        "trump_ev": trump_ev, "harris_ev": harris_ev, "total_ev": total_ev,
    }


# ---------------------------------------------------------------------------
# Metric builders (used by both plots and console analysis)
# ---------------------------------------------------------------------------

def compute_daily_accuracy(pm, p538, fec):
    """Compute daily accuracy percentages for both sources
    over the overlap period."""
    start = max(pm["date"].min(), p538["date"].min())
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")
    fec_winners = fec.set_index("state")["winner"]

    records = []
    for as_of_date in dates:
        pm_day = pm[
            (pm["date"] == as_of_date) & (pm["state"].isin(OVERLAP_STATES))
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
            set(pm_day["state"]) & set(p5_day["state"]) & set(fec_winners.index)
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
    """Build per-group accuracy comparison between Polymarket and 538
    on Sept 12."""
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
        rows.append({
            "group": group_label,
            "pm_pct": pm_stats["pct"],
            "p538_pct": p538_stats["pct"],
        })
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
    """Build Electoral Vote share metrics for Polymarket, 538,
    and actual results."""
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
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(pm_preds, p538_preds, fec_winners, states):
    """Run McNemar's exact test on paired state-level winner calls.

    Returns a dict with counts for each cell of the 2x2 disagreement
    table and the two-sided p-value.
    """
    only_pm_right = 0
    only_538_right = 0
    both_right = 0
    both_wrong = 0

    for st in states:
        pm_row = pm_preds[pm_preds["state"] == st]
        p538_row = p538_preds[p538_preds["state"] == st]
        if pm_row.empty or p538_row.empty:
            continue
        actual = fec_winners.loc[st]
        pm_ok = pm_row["predicted_winner"].iloc[0] == actual
        p538_ok = p538_row["predicted_winner"].iloc[0] == actual

        if pm_ok and p538_ok:
            both_right += 1
        elif pm_ok and not p538_ok:
            only_pm_right += 1
        elif not pm_ok and p538_ok:
            only_538_right += 1
        else:
            both_wrong += 1

    n_disc = only_pm_right + only_538_right
    if n_disc == 0:
        p_val = 1.0
    else:
        p_val = binomtest(only_pm_right, n_disc, 0.5).pvalue

    return {
        "both_right": both_right,
        "both_wrong": both_wrong,
        "only_pm_right": only_pm_right,
        "only_538_right": only_538_right,
        "n_discordant": n_disc,
        "p_value": p_val,
    }
