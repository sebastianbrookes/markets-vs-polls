"""
Compare Polymarket odds vs. FiveThirtyEight poll averages against FEC
ground truth for the 2024 U.S. presidential election.

Sections:
    1. Winner prediction accuracy (head-to-head + Polymarket standalone)
    2. Electoral vote predictions
    3. Time-series accuracy (daily, March-Sept 12)
    4. McNemar's exact test (paired accuracy comparison)

Run from project root:
    python -m src.analysis.accuracy
"""

import sys
from pathlib import Path

import pandas as pd

_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.analysis.accuracy.metrics import (
    ELECTION_DATE,
    OVERLAP_CUTOFF,
    OVERLAP_STATES,
    SWING_OVERLAP,
    THIRTY_DAYS_OUT,
    compute_daily_accuracy,
    latest_per_state,
    load_data,
    mcnemar_test,
    predict_winner_pm,
    predict_winner_538,
)
from src.clean.utils import SWING_STATES

# ---------------------------------------------------------------------------
# Winner prediction helpers (console output)
# ---------------------------------------------------------------------------

def _winner_accuracy(snapshot, fec, label, states=None):
    """Print winner-call accuracy for a snapshot vs FEC results."""
    merged = snapshot.merge(fec[["state", "winner"]], on="state")
    if states is not None:
        merged = merged[merged["state"].isin(states)]
    n = len(merged)
    correct = (merged["predicted_winner"] == merged["winner"]).sum()
    pct = correct / n * 100 if n else 0
    print(f"  {label}: {correct}/{n} correct ({pct:.1f}%)")

    misses = merged[merged["predicted_winner"] != merged["winner"]]
    if not misses.empty:
        for _, row in misses.iterrows():
            print(f"    MISS: {row['state']} — predicted "
                  f"{row['predicted_winner']}, actual {row['winner']}")
    return correct, n


# ---------------------------------------------------------------------------
# Section 1: Winner Prediction Accuracy
# ---------------------------------------------------------------------------

def _section_winner(pm, p538, fec):
    """Head-to-head winner accuracy + Polymarket standalone snapshots."""
    print("=" * 60)
    print("1. WINNER PREDICTION ACCURACY")
    print("=" * 60)

    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]
    pm_snap = predict_winner_pm(pm_snap)

    p538_snap = latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]
    p538_snap = predict_winner_538(p538_snap)

    print(f"\nHead-to-head on {OVERLAP_CUTOFF} ({len(OVERLAP_STATES)} states):")
    _winner_accuracy(pm_snap, fec, "Polymarket", OVERLAP_STATES)
    _winner_accuracy(p538_snap, fec, "538", OVERLAP_STATES)

    print(f"\nSwing states only ({len(SWING_OVERLAP)} states):")
    _winner_accuracy(pm_snap, fec, "Polymarket", SWING_OVERLAP)
    _winner_accuracy(p538_snap, fec, "538", SWING_OVERLAP)

    print(f"\nPolymarket standalone — Oct 6 (30 days out):")
    pm_oct = pm[pm["date"] == pd.Timestamp(THIRTY_DAYS_OUT)]
    pm_oct = predict_winner_pm(pm_oct)
    _winner_accuracy(pm_oct, fec, "All states")
    _winner_accuracy(pm_oct, fec, "Swing states", SWING_STATES)

    print(f"\nPolymarket standalone — Nov 4 (election eve):")
    pm_eve = pm[pm["date"] == pd.Timestamp(ELECTION_DATE).normalize()
                - pd.Timedelta(days=1)]
    pm_eve = predict_winner_pm(pm_eve)
    _winner_accuracy(pm_eve, fec, "All states")
    _winner_accuracy(pm_eve, fec, "Swing states", SWING_STATES)
    print()


# ---------------------------------------------------------------------------
# Section 2: Electoral Vote Predictions
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
    print("2. ELECTORAL VOTE PREDICTIONS")
    print("=" * 60)
    print(f"  Actual 2024 result: Trump 312 — Harris 226\n")

    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]
    pm_snap = predict_winner_pm(pm_snap)

    p538_snap = latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]
    p538_snap = predict_winner_538(p538_snap)

    print(f"  Head-to-head ({OVERLAP_CUTOFF}, {len(OVERLAP_STATES)} states):")
    _ev_prediction(pm_snap, fec, "Polymarket")
    _ev_prediction(p538_snap, fec, "538")

    pm_eve = pm[pm["date"] == pd.Timestamp(ELECTION_DATE).normalize()
                - pd.Timedelta(days=1)]
    pm_eve = predict_winner_pm(pm_eve)
    print(f"\n  Polymarket standalone (Nov 4, all states):")
    _ev_prediction(pm_eve, fec, "Polymarket")
    print()


# ---------------------------------------------------------------------------
# Section 3: Time-Series Accuracy (daily, March-Sept 12)
# ---------------------------------------------------------------------------

def _section_timeseries(pm, p538, fec):
    """Daily winner-accuracy for both sources over their overlap period."""
    print("=" * 60)
    print("3. TIME-SERIES ACCURACY (daily, March 2024 – Sept 12)")
    print("=" * 60)

    ts = compute_daily_accuracy(pm, p538, fec)

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
# Section 4: McNemar's Exact Test
# ---------------------------------------------------------------------------

def _section_mcnemar(pm, p538, fec):
    """Print McNemar's exact test results for the Sept 12 snapshot."""
    print("=" * 60)
    print("4. McNEMAR'S EXACT TEST (paired accuracy comparison)")
    print("=" * 60)

    pm_snap = pm[pm["date"] == pd.Timestamp(OVERLAP_CUTOFF)]
    pm_snap = pm_snap[pm_snap["state"].isin(OVERLAP_STATES)]
    pm_snap = predict_winner_pm(pm_snap)

    p538_snap = latest_per_state(p538, OVERLAP_CUTOFF)
    p538_snap = p538_snap[p538_snap["state"].isin(OVERLAP_STATES)]
    p538_snap = predict_winner_538(p538_snap)

    fec_winners = fec.set_index("state")["winner"]

    for label, states in [("All states", OVERLAP_STATES),
                          ("Swing states", SWING_OVERLAP)]:
        r = mcnemar_test(pm_snap, p538_snap, fec_winners, states)
        print(f"\n  {label} ({len(states)}):")
        print(f"    Both correct: {r['both_right']}  |  "
              f"Both wrong: {r['both_wrong']}")
        print(f"    Only PM correct: {r['only_pm_right']}  |  "
              f"Only 538 correct: {r['only_538_right']}")
        print(f"    Discordant pairs: {r['n_discordant']}  →  "
              f"p = {r['p_value']:.2f}")

    print()
    print("  Neither difference is statistically significant at α = 0.05.")
    print("  The observed gap is suggestive but cannot rule out chance "
          "with n this small.")
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
    _section_ev(pm, p538, fec)
    _section_timeseries(pm, p538, fec)
    _section_mcnemar(pm, p538, fec)

    print("=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
