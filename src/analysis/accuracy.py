"""
Compare Polymarket odds vs. FiveThirtyEight poll averages against FEC
ground truth for the 2024 U.S. presidential election.

Sections:
    1. Winner prediction accuracy (head-to-head + Polymarket standalone)
    2. Margin accuracy (MAE)
    3. Electoral vote predictions
    4. Time-series accuracy (daily, March–Sept 12)

Run from project root:
    python -m src.analysis.accuracy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as `python -m src.analysis.accuracy` from project root
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAP_CUTOFF = "2024-09-12"   # last date with 538 data
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"

# 13 states present in both sources on Sept 12 (538 minus "US")
OVERLAP_STATES = [
    "AZ", "CA", "FL", "GA", "MI", "MN", "NC", "NH", "NV", "OH", "PA",
    "TX", "WI",
]

SWING_OVERLAP = sorted(set(OVERLAP_STATES) & set(SWING_STATES))  # 7
NON_SWING_OVERLAP = sorted(set(OVERLAP_STATES) - set(SWING_STATES))  # 6


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data():
    """Load the three processed CSVs and do minimal prep."""
    pm = pd.read_csv(PROCESSED_DIR / "polymarket_daily.csv",
                     parse_dates=["date"])
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv",
                       parse_dates=["date"])
    fec = pd.read_csv(PROCESSED_DIR / "fec_results.csv")

    # Drop 538 national aggregate rows
    p538 = p538[p538["state"] != "US"].copy()

    return pm, p538, fec


def _latest_per_state(p538, as_of):
    """Forward-fill 538: latest row per state on or before *as_of*."""
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


# ---------------------------------------------------------------------------
# Winner prediction helpers
# ---------------------------------------------------------------------------

def _predict_winner_pm(df):
    """Add predicted_winner column from Polymarket probabilities."""
    df = df.copy()
    df["predicted_winner"] = np.where(
        df["trump_prob"] > 0.5, "Trump", "Harris"
    )
    return df


def _predict_winner_538(df):
    """Add predicted_winner column from 538 vote-share estimates."""
    df = df.copy()
    df["predicted_winner"] = np.where(
        df["trump_pct"] > df["dem_pct"], "Trump", "Harris"
    )
    return df


def _winner_accuracy(snapshot, fec, label, states=None):
    """Print winner-call accuracy for a snapshot vs FEC results."""
    merged = snapshot.merge(fec[["state", "winner"]], on="state")
    if states is not None:
        merged = merged[merged["state"].isin(states)]
    n = len(merged)
    correct = (merged["predicted_winner"] == merged["winner"]).sum()
    pct = correct / n * 100 if n else 0
    print(f"  {label}: {correct}/{n} correct ({pct:.1f}%)")

    # Show misses
    misses = merged[merged["predicted_winner"] != merged["winner"]]
    if not misses.empty:
        for _, row in misses.iterrows():
            print(f"    MISS: {row['state']} — predicted {row['predicted_winner']}, "
                  f"actual {row['winner']}")
    return correct, n


# ---------------------------------------------------------------------------
# Section 1: Winner Prediction Accuracy
# ---------------------------------------------------------------------------

def _section_winner(pm, p538, fec):
    """Head-to-head winner accuracy + Polymarket standalone snapshots."""
    print("=" * 60)
    print("1. WINNER PREDICTION ACCURACY")
    print("=" * 60)

    # --- Head-to-head on Sept 12, 13 overlap states ---
    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]
    pm_snap = _predict_winner_pm(pm_snap)

    p538_snap = _latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]
    p538_snap = _predict_winner_538(p538_snap)

    print(f"\nHead-to-head on {OVERLAP_CUTOFF} ({len(OVERLAP_STATES)} states):")
    _winner_accuracy(pm_snap, fec, "Polymarket", OVERLAP_STATES)
    _winner_accuracy(p538_snap, fec, "538", OVERLAP_STATES)

    print(f"\nSwing states only ({len(SWING_OVERLAP)} states):")
    _winner_accuracy(pm_snap, fec, "Polymarket", SWING_OVERLAP)
    _winner_accuracy(p538_snap, fec, "538", SWING_OVERLAP)

    # --- Polymarket standalone snapshots ---
    print(f"\nPolymarket standalone — Oct 6 (30 days out):")
    pm_oct = pm[pm["date"] == pd.Timestamp(THIRTY_DAYS_OUT)]
    pm_oct = _predict_winner_pm(pm_oct)
    _winner_accuracy(pm_oct, fec, "All states")
    _winner_accuracy(pm_oct, fec, "Swing states", SWING_STATES)

    print(f"\nPolymarket standalone — Nov 4 (election eve):")
    pm_eve = pm[pm["date"] == pd.Timestamp(ELECTION_DATE).normalize()
                - pd.Timedelta(days=1)]
    pm_eve = _predict_winner_pm(pm_eve)
    _winner_accuracy(pm_eve, fec, "All states")
    _winner_accuracy(pm_eve, fec, "Swing states", SWING_STATES)
    print()


# ---------------------------------------------------------------------------
# Section 2: Margin Accuracy (MAE)
# ---------------------------------------------------------------------------

def _margin_mae(pm_snap, p538_snap, fec, states, label):
    """Compute and print MAE of predicted vs actual margin."""
    fec_sub = fec[fec["state"].isin(states)].copy()
    fec_sub["actual_margin_pp"] = fec_sub["margin"] * 100  # to percentage pts

    # Polymarket margin: trump_lead is probability gap, scale to pp
    pm_m = pm_snap[pm_snap["state"].isin(states)].merge(
        fec_sub[["state", "actual_margin_pp"]], on="state"
    )
    pm_mae = (pm_m["trump_lead"] * 100 - pm_m["actual_margin_pp"]).abs().mean()

    # 538 margin: trump_lead is already in vote-share pp
    p5_m = p538_snap[p538_snap["state"].isin(states)].merge(
        fec_sub[["state", "actual_margin_pp"]], on="state"
    )
    p5_mae = (p5_m["trump_lead"] - p5_m["actual_margin_pp"]).abs().mean()

    print(f"  {label}:")
    print(f"    Polymarket MAE: {pm_mae:.2f} pp  (probability gap — not directly "
          f"comparable to vote-share)")
    print(f"    538 MAE:        {p5_mae:.2f} pp  (vote-share margin)")


def _section_margin(pm, p538, fec):
    """MAE comparison on Sept 12 overlap states."""
    print("=" * 60)
    print("2. MARGIN ACCURACY (MAE) — as of Sept 12")
    print("=" * 60)
    print("  NOTE: Polymarket margin is a probability gap, 538 margin is")
    print("  vote-share. These are NOT directly comparable units.\n")

    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]

    p538_snap = _latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]

    _margin_mae(pm_snap, p538_snap, fec, OVERLAP_STATES,
                f"All 13 overlap states")
    _margin_mae(pm_snap, p538_snap, fec, SWING_OVERLAP,
                f"Swing states ({len(SWING_OVERLAP)})")
    _margin_mae(pm_snap, p538_snap, fec, NON_SWING_OVERLAP,
                f"Non-swing states ({len(NON_SWING_OVERLAP)})")
    print()


# ---------------------------------------------------------------------------
# Section 3: Electoral Vote Predictions
# ---------------------------------------------------------------------------

def _ev_prediction(snapshot, fec, label):
    """Sum electoral votes by predicted winner."""
    merged = snapshot.merge(fec[["state", "electoral_votes"]], on="state")
    trump_ev = merged.loc[merged["predicted_winner"] == "Trump",
                          "electoral_votes"].sum()
    harris_ev = merged.loc[merged["predicted_winner"] == "Harris",
                           "electoral_votes"].sum()
    total = merged["electoral_votes"].sum()
    print(f"  {label}: Trump {trump_ev} — Harris {harris_ev}  "
          f"(of {total} EV in sample)")


def _section_ev(pm, p538, fec):
    """Electoral vote totals from each source's predictions."""
    print("=" * 60)
    print("3. ELECTORAL VOTE PREDICTIONS")
    print("=" * 60)
    print(f"  Actual 2024 result: Trump 312 — Harris 226\n")

    # Head-to-head on 13 overlap states, Sept 12
    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]
    pm_snap = _predict_winner_pm(pm_snap)

    p538_snap = _latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]
    p538_snap = _predict_winner_538(p538_snap)

    print(f"  Head-to-head ({OVERLAP_CUTOFF}, {len(OVERLAP_STATES)} states):")
    _ev_prediction(pm_snap, fec, "Polymarket")
    _ev_prediction(p538_snap, fec, "538")

    # Polymarket standalone on election eve (all 50 states)
    pm_eve = pm[pm["date"] == pd.Timestamp(ELECTION_DATE).normalize()
                - pd.Timedelta(days=1)]
    pm_eve = _predict_winner_pm(pm_eve)
    print(f"\n  Polymarket standalone (Nov 4, all states):")
    _ev_prediction(pm_eve, fec, "Polymarket")
    print()


# ---------------------------------------------------------------------------
# Section 4: Time-Series Accuracy (daily, March–Sept 12)
# ---------------------------------------------------------------------------

def _section_timeseries(pm, p538, fec):
    """Daily winner-accuracy for both sources over their overlap period."""
    print("=" * 60)
    print("4. TIME-SERIES ACCURACY (daily, March 2024 – Sept 12)")
    print("=" * 60)

    # Build date range: first date both sources have data → Sept 12
    pm_start = pm["date"].min()
    p538_start = p538["date"].min()
    start = max(pm_start, p538_start)
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")

    fec_winners = fec.set_index("state")["winner"]

    records = []
    for d in dates:
        # Polymarket snapshot
        pm_day = pm[pm["date"] == d]
        pm_day = pm_day[pm_day["state"].isin(OVERLAP_STATES)]
        if pm_day.empty:
            continue
        pm_day = _predict_winner_pm(pm_day)

        # 538 forward-fill snapshot
        p5_day = _latest_per_state(p538, d)
        p5_day = p5_day[p5_day["state"].isin(OVERLAP_STATES)]
        if p5_day.empty:
            continue
        p5_day = _predict_winner_538(p5_day)

        # Common states available on this day
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

    ts = pd.DataFrame(records)

    if ts.empty:
        print("  No overlapping data found.\n")
        return

    print(f"\n  Period: {ts['date'].min().date()} to {ts['date'].max().date()}")
    print(f"  Total days with data: {len(ts)}")
    print(f"\n  Overall accuracy (mean of daily %):")
    print(f"    Polymarket: {ts['pm_pct'].mean():.1f}%  "
          f"(median {ts['pm_pct'].median():.1f}%)")
    print(f"    538:        {ts['p538_pct'].mean():.1f}%  "
          f"(median {ts['p538_pct'].median():.1f}%)")

    # Period breakdown
    periods = [
        ("Mar–May", "2024-03-01", "2024-05-31"),
        ("Jun–Jul", "2024-06-01", "2024-07-31"),
        ("Aug–Sep 12", "2024-08-01", "2024-09-12"),
    ]
    print("\n  Period breakdown (mean daily %):")
    for label, pstart, pend in periods:
        mask = (ts["date"] >= pstart) & (ts["date"] <= pend)
        sub = ts[mask]
        if sub.empty:
            print(f"    {label}: no data")
            continue
        print(f"    {label}:  PM {sub['pm_pct'].mean():.1f}%  |  "
              f"538 {sub['p538_pct'].mean():.1f}%  ({len(sub)} days)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run all accuracy analyses and print results."""
    print("\n" + "=" * 60)
    print("  POLYMARKET vs. 538: PREDICTION ACCURACY ANALYSIS")
    print("=" * 60 + "\n")

    pm, p538, fec = load_data()

    _section_winner(pm, p538, fec)
    _section_margin(pm, p538, fec)
    _section_ev(pm, p538, fec)
    _section_timeseries(pm, p538, fec)

    print("=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
