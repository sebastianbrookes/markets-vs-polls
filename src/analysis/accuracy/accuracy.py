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

from .metrics import (
    ELECTION_EVE,
    OVERLAP_CUTOFF,
    OVERLAP_STATES,
    SWING_OVERLAP,
    THIRTY_DAYS_OUT,
    compute_daily_accuracy,
    load_data,
    mcnemar_test,
    p538_snapshot,
    pm_snapshot,
)
from src.clean.utils import PROCESSED_DIR, SWING_STATES


def _merge_with_fec(snapshot, fec, states=None):
    merged = snapshot.merge(fec[["state", "winner"]], on="state")
    if states is not None:
        merged = merged[merged["state"].isin(states)]
    return merged


def _print_misses(merged):
    misses = merged[merged["predicted_winner"] != merged["winner"]]
    for _, row in misses.iterrows():
        print(
            f"    MISS: {row['state']} — predicted "
            f"{row['predicted_winner']}, "
            f"actual {row['winner']}"
        )


def winner_accuracy(snapshot, fec, label, states=None):
    """Print and return winner-call accuracy vs FEC.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Predictions with state and predicted_winner.
    fec : pd.DataFrame
        FEC results with state and winner.
    label : str
        Display label for the output line.
    states : list or None
        States to include. None means all.

    Returns
    -------
    tuple of (int, int)
        (number correct, total states).
    """
    # Join predictions with actual FEC winners by state
    merged = _merge_with_fec(snapshot, fec, states)
    n = len(merged)
    correct = (merged["predicted_winner"] == merged["winner"]).sum()
    pct = correct / n * 100 if n else 0
    print(f"  {label}: {correct}/{n} correct ({pct:.1f}%)")
    _print_misses(merged)
    return (correct, n)


def _sum_ev_by_candidate(merged):
    trump_ev = merged.loc[
        merged["predicted_winner"] == "Trump", "electoral_votes"
    ].sum()
    harris_ev = merged.loc[
        merged["predicted_winner"] == "Harris", "electoral_votes"
    ].sum()
    total = merged["electoral_votes"].sum()
    return trump_ev, harris_ev, total


def ev_prediction(snapshot, fec, label):
    """Print electoral vote totals by predicted winner.

    Parameters
    ----------
    snapshot : pd.DataFrame
        Predictions with state and predicted_winner.
    fec : pd.DataFrame
        FEC results with state and electoral_votes.
    label : str
        Display label for the output line.
    """
    # Join predictions with EV counts
    merged = snapshot.merge(fec[["state", "electoral_votes"]], on="state")
    trump_ev, harris_ev, total = _sum_ev_by_candidate(merged)
    print(
        f"  {label}: Trump {trump_ev} — Harris "
        f"{harris_ev}  (of {total} EV in sample)"
    )


def print_head_to_head(pm_snap, p5_snap, fec):
    """Print head-to-head accuracy for both sources.

    Parameters
    ----------
    pm_snap : pd.DataFrame
        Polymarket snapshot with predicted_winner.
    p5_snap : pd.DataFrame
        538 snapshot with predicted_winner.
    fec : pd.DataFrame
        FEC election results.
    """
    n_all = len(OVERLAP_STATES)
    print(f"\nHead-to-head on {OVERLAP_CUTOFF} " f"({n_all} states):")
    winner_accuracy(pm_snap, fec, "Polymarket", OVERLAP_STATES)
    winner_accuracy(p5_snap, fec, "538", OVERLAP_STATES)

    n_swing = len(SWING_OVERLAP)
    print(f"\nSwing states only ({n_swing} states):")
    winner_accuracy(pm_snap, fec, "Polymarket", SWING_OVERLAP)
    winner_accuracy(p5_snap, fec, "538", SWING_OVERLAP)


def print_pm_standalone(pm, fec):
    """Print Polymarket standalone accuracy at two dates.

    Parameters
    ----------
    pm : pd.DataFrame
        Polymarket daily predictions.
    fec : pd.DataFrame
        FEC election results.
    """
    # 30 days before election (Oct 6)
    print("\nPolymarket standalone — Oct 6 (30 days out):")
    pm_oct = pm_snapshot(pm, THIRTY_DAYS_OUT)
    winner_accuracy(pm_oct, fec, "All states")
    winner_accuracy(pm_oct, fec, "Swing states", SWING_STATES)

    # Election eve (Nov 4)
    print("\nPolymarket standalone — Nov 4 (election eve):")
    pm_eve = pm_snapshot(pm, ELECTION_EVE)
    winner_accuracy(pm_eve, fec, "All states")
    winner_accuracy(pm_eve, fec, "Swing states", SWING_STATES)


def print_section_winner(pm, p538, fec):
    """Print Section 1: winner prediction accuracy.

    Parameters
    ----------
    pm : pd.DataFrame
        Polymarket daily predictions.
    p538 : pd.DataFrame
        FiveThirtyEight poll averages.
    fec : pd.DataFrame
        FEC election results.
    """
    print("=" * 60)
    print("1. WINNER PREDICTION ACCURACY")
    print("=" * 60)

    pm_snap = pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p5_snap = p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)

    print_head_to_head(pm_snap, p5_snap, fec)
    print_pm_standalone(pm, fec)
    print()


def _print_ev_head_to_head(pm, p538, fec):
    pm_snap = pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p5_snap = p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)
    n = len(OVERLAP_STATES)
    print(f"  Head-to-head ({OVERLAP_CUTOFF}, {n} states):")
    ev_prediction(pm_snap, fec, "Polymarket")
    ev_prediction(p5_snap, fec, "538")


def print_section_ev(pm, p538, fec):
    """Print Section 2: electoral vote predictions.

    Parameters
    ----------
    pm : pd.DataFrame
        Polymarket daily predictions.
    p538 : pd.DataFrame
        FiveThirtyEight poll averages.
    fec : pd.DataFrame
        FEC election results.
    """
    print("=" * 60)
    print("2. ELECTORAL VOTE PREDICTIONS")
    print("=" * 60)
    print("  Actual 2024 result: Trump 312 — Harris 226\n")

    _print_ev_head_to_head(pm, p538, fec)

    pm_eve = pm_snapshot(pm, ELECTION_EVE)
    print("\n  Polymarket standalone (Nov 4, all states):")
    ev_prediction(pm_eve, fec, "Polymarket")
    print()


def print_timeseries_summary(ts):
    """Print overall time-series accuracy statistics.

    Parameters
    ----------
    ts : pd.DataFrame
        Daily accuracy with date, pm_pct, p538_pct.
    """
    start = ts["date"].min().date()
    end = ts["date"].max().date()
    print(f"\n  Period: {start} to {end}")
    print(f"  Total days with data: {len(ts)}")
    print("\n  Overall accuracy (mean of daily %):")
    pm_mean = ts["pm_pct"].mean()
    pm_med = ts["pm_pct"].median()
    print(f"    Polymarket: {pm_mean:.1f}%  " f"(median {pm_med:.1f}%)")
    p5_mean = ts["p538_pct"].mean()
    p5_med = ts["p538_pct"].median()
    print(f"    538:        {p5_mean:.1f}%  " f"(median {p5_med:.1f}%)")


def _print_single_period(ts, label, pstart, pend):
    # Filter to rows within this sub-period
    mask = (ts["date"] >= pstart) & (ts["date"] <= pend)
    sub = ts[mask]
    if sub.empty:
        print(f"    {label}: no data")
        return
    pm_mean = sub["pm_pct"].mean()
    p5_mean = sub["p538_pct"].mean()
    n_days = len(sub)
    print(
        f"    {label}:  PM {pm_mean:.1f}%  |  " f"538 {p5_mean:.1f}%  ({n_days} days)"
    )


def print_period_breakdown(ts):
    """Print accuracy broken down by time period.

    Parameters
    ----------
    ts : pd.DataFrame
        Daily accuracy with date, pm_pct, p538_pct.
    """
    # Split the timeline into sub-periods to show how
    # accuracy evolved across different campaign phases
    periods = [
        ("Mar\u2013May", "2024-03-01", "2024-05-31"),
        ("Jun\u2013Jul", "2024-06-01", "2024-07-31"),
        ("Aug\u2013Sep 12", "2024-08-01", "2024-09-12"),
    ]
    print("\n  Period breakdown (mean daily %):")
    for label, pstart, pend in periods:
        _print_single_period(ts, label, pstart, pend)


def print_section_timeseries(pm, p538, fec):
    """Print Section 3: daily time-series accuracy.

    Parameters
    ----------
    pm : pd.DataFrame
        Polymarket daily predictions.
    p538 : pd.DataFrame
        FiveThirtyEight poll averages.
    fec : pd.DataFrame
        FEC election results.
    """
    print("=" * 60)
    title = "3. TIME-SERIES ACCURACY "
    print(title + "(daily, March 2024 \u2013 Sept 12)")
    print("=" * 60)

    ts = compute_daily_accuracy(pm, p538, fec, OVERLAP_CUTOFF, OVERLAP_STATES)
    if ts.empty:
        print("  No overlapping data found.\n")
        return

    print_timeseries_summary(ts)
    print_period_breakdown(ts)
    print()


def print_mcnemar_result(label, n_states, result):
    """Print one McNemar test result block.

    Parameters
    ----------
    label : str
        Group label (e.g., "All states").
    n_states : int
        Number of states in the group.
    result : dict
        Output from mcnemar_test().
    """
    print(f"\n  {label} ({n_states}):")
    print(
        f"    Both correct: {result['both_right']}  |  "
        f"Both wrong: {result['both_wrong']}"
    )
    print(
        f"    Only PM correct: "
        f"{result['only_pm_right']}  |  "
        f"Only 538 correct: {result['only_538_right']}"
    )
    print(
        f"    Discordant pairs: "
        f"{result['n_discordant']}"
        f"  \u2192  p = {result['p_value']:.2f}"
    )


def _run_mcnemar_tests(pm, p538, fec):
    # Take same-date snapshots so McNemar compares
    # predictions made at the same point in time
    pm_snap = pm_snapshot(pm, OVERLAP_CUTOFF, OVERLAP_STATES)
    p5_snap = p538_snapshot(p538, OVERLAP_CUTOFF, OVERLAP_STATES)
    # Index FEC winners by state for fast lookup in the test
    fec_winners = fec.set_index("state")["winner"]

    # Run McNemar on all overlapping states, then swing only
    groups = [
        ("All states", OVERLAP_STATES),
        ("Swing states", SWING_OVERLAP),
    ]
    for label, states in groups:
        r = mcnemar_test(pm_snap, p5_snap, fec_winners, states)
        print_mcnemar_result(label, len(states), r)


def _print_mcnemar_conclusion():
    print()
    print("  Neither difference is statistically " "significant at \u03b1 = 0.05.")
    print(
        "  The observed gap is suggestive but cannot "
        "rule out chance with n this small."
    )
    print()


def print_section_mcnemar(pm, p538, fec):
    """Print Section 4: McNemar exact test results.

    Parameters
    ----------
    pm : pd.DataFrame
        Polymarket daily predictions.
    p538 : pd.DataFrame
        FiveThirtyEight poll averages.
    fec : pd.DataFrame
        FEC election results.
    """
    print("=" * 60)
    print("4. McNEMAR'S EXACT TEST " "(paired accuracy comparison)")
    print("=" * 60)

    _run_mcnemar_tests(pm, p538, fec)
    _print_mcnemar_conclusion()


def main():
    """Run all accuracy analyses and print results."""
    print("\n" + "=" * 60)
    print("  POLYMARKET vs. 538: " "PREDICTION ACCURACY ANALYSIS")
    print("=" * 60 + "\n")

    pm, p538, fec = load_data(PROCESSED_DIR)

    print_section_winner(pm, p538, fec)
    print_section_ev(pm, p538, fec)
    print_section_timeseries(pm, p538, fec)
    print_section_mcnemar(pm, p538, fec)

    print("=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
