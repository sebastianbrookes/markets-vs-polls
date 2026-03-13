"""
Computation engine for the accuracy analysis
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.clean.utils import PROCESSED_DIR, SWING_STATES

OVERLAP_CUTOFF = "2024-09-12"
ELECTION_DATE = "2024-11-05"
THIRTY_DAYS_OUT = "2024-10-06"
ELECTION_EVE = (pd.Timestamp(ELECTION_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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


def load_data(data_dir):
    """
    Load Polymarket, 538, and FEC data from CSVs.

    Parameters
    ----------
    data_dir : Path
        Directory containing processed CSV files.

    Returns
    -------
    tuple of DataFrame
        (polymarket, polls_538, fec_results).
    """
    pm = pd.read_csv(
        data_dir / "polymarket_daily.csv",
        parse_dates=["date"],
    )
    p538 = pd.read_csv(
        data_dir / "polls_538.csv",
        parse_dates=["date"],
    )
    fec = pd.read_csv(data_dir / "fec_results.csv")

    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538, fec


def latest_per_state(p538, as_of):
    """
    Return the most recent 538 row per state on or
    before a given date.

    Parameters
    ----------
    p538 : DataFrame
        FiveThirtyEight polling data.
    as_of : str
        Date cutoff in YYYY-MM-DD format.

    Returns
    -------
    DataFrame
        One row per state with the latest data.
    """
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


def predict_winner_pm(df):
    """
    Add predicted_winner column from Polymarket odds.

    Parameters
    ----------
    df : DataFrame
        Must contain a trump_prob column.

    Returns
    -------
    DataFrame
        Copy with predicted_winner column added.
    """
    df = df.copy()
    df["predicted_winner"] = df["trump_prob"].apply(
        lambda x: "Trump" if x > 0.5 else "Harris"
    )
    return df


def predict_winner_538(df):
    """
    Add predicted_winner column from 538 poll shares.

    Parameters
    ----------
    df : DataFrame
        Must contain trump_pct and dem_pct columns.

    Returns
    -------
    DataFrame
        Copy with predicted_winner column added.
    """
    df = df.copy()
    df["predicted_winner"] = df.apply(
        lambda row: "Trump" if row["trump_pct"] > row["dem_pct"] else "Harris",
        axis=1,
    )
    return df


def pm_snapshot(pm, as_of, states=None):
    """
    Return Polymarket predictions for a single date.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    as_of : str
        Date in YYYY-MM-DD format.
    states : list of str or None
        States to include. None means all.

    Returns
    -------
    DataFrame
        Filtered snapshot with predicted_winner.
    """
    snapshot = predict_winner_pm(pm[pm["date"] == pd.Timestamp(as_of)])
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def p538_snapshot(p538, as_of, states=None):
    """
    Return 538 predictions as of a given date.

    Parameters
    ----------
    p538 : DataFrame
        FiveThirtyEight polling data.
    as_of : str
        Date in YYYY-MM-DD format.
    states : list of str or None
        States to include. None means all.

    Returns
    -------
    DataFrame
        Filtered snapshot with predicted_winner.
    """
    snapshot = predict_winner_538(latest_per_state(p538, as_of))
    if states is not None:
        snapshot = snapshot[snapshot["state"].isin(states)]
    return snapshot


def compute_accuracy(snapshot, fec_winners, states=None):
    """
    Compare predicted winners to actual results.

    Parameters
    ----------
    snapshot : DataFrame
        Must have state and predicted_winner columns.
    fec_winners : Series
        Actual winners indexed by state.
    states : list of str or None
        States to evaluate. None means all.

    Returns
    -------
    dict
        Keys: correct, n_states, pct.
    """
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
    common = predicted.index.intersection(fec_winners.index)

    n_states = len(common)
    if n_states == 0:
        return {"correct": 0, "n_states": 0, "pct": 0.0}

    correct = int((predicted.loc[common] == fec_winners.loc[common]).sum())
    return {
        "correct": correct,
        "n_states": n_states,
        "pct": correct / n_states * 100,
    }


def compute_ev_share(snapshot, fec, states=None):
    """
    Compute Electoral Vote share from predicted winners.

    Parameters
    ----------
    snapshot : DataFrame
        Must have state and predicted_winner columns.
    fec : DataFrame
        Must have state and electoral_votes columns.
    states : list of str or None
        States to include. None means all.

    Returns
    -------
    dict
        Keys: trump_pct, harris_pct, trump_ev,
        harris_ev, total_ev.
    """
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

    trump = merged["predicted_winner"] == "Trump"
    trump_ev = int(merged.loc[trump, "electoral_votes"].sum())
    harris_ev = total_ev - trump_ev
    trump_pct = trump_ev / total_ev * 100

    return {
        "trump_pct": trump_pct,
        "harris_pct": 100 - trump_pct,
        "trump_ev": trump_ev,
        "harris_ev": harris_ev,
        "total_ev": total_ev,
    }


def _daily_record(pm, p538, as_of, states, fec_winners):
    """
    Compute accuracy for one day for both sources.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    p538 : DataFrame
        FiveThirtyEight polling data.
    as_of : Timestamp
        Date to evaluate.
    states : list of str
        States to include.
    fec_winners : Series
        Actual winners indexed by state.

    Returns
    -------
    dict or None
        Record with accuracy stats, or None if
        data is missing for that day.
    """
    pm_day = pm[(pm["date"] == as_of) & (pm["state"].isin(states))]
    if pm_day.empty:
        return None
    pm_day = predict_winner_pm(pm_day)

    p5_day = latest_per_state(p538, as_of)
    p5_day = p5_day[p5_day["state"].isin(states)]
    if p5_day.empty:
        return None
    p5_day = predict_winner_538(p5_day)

    common = sorted(
        set(pm_day["state"]) & set(p5_day["state"]) & set(fec_winners.index)
    )
    if not common:
        return None

    pm_acc = compute_accuracy(pm_day, fec_winners, common)
    p5_acc = compute_accuracy(p5_day, fec_winners, common)

    return {
        "date": as_of,
        "n_states": pm_acc["n_states"],
        "pm_correct": pm_acc["correct"],
        "p538_correct": p5_acc["correct"],
        "pm_pct": pm_acc["pct"],
        "p538_pct": p5_acc["pct"],
    }


def compute_daily_accuracy(pm, p538, fec, cutoff, states):
    """
    Compute daily accuracy for both sources over
    the overlap period.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    p538 : DataFrame
        FiveThirtyEight polling data.
    fec : DataFrame
        FEC results data.
    cutoff : str
        End date in YYYY-MM-DD format.
    states : list of str
        States to include.

    Returns
    -------
    DataFrame
        Daily accuracy records with columns: date,
        n_states, pm_correct, p538_correct, pm_pct,
        p538_pct.
    """
    start = max(pm["date"].min(), p538["date"].min())
    end = pd.Timestamp(cutoff)
    dates = pd.date_range(start, end, freq="D")
    fec_winners = fec.set_index("state")["winner"]

    records = []
    for as_of_date in dates:
        record = _daily_record(pm, p538, as_of_date, states, fec_winners)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records)


def build_head_to_head_metrics(pm, p538, fec, cutoff, groups):
    """
    Build per-group accuracy comparison between
    Polymarket and 538 on a given date.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    p538 : DataFrame
        FiveThirtyEight polling data.
    fec : DataFrame
        FEC results data.
    cutoff : str
        Date in YYYY-MM-DD format.
    groups : list of tuple
        Each tuple is (label, list_of_states).

    Returns
    -------
    DataFrame
        One row per group with pm_pct and p538_pct.
    """
    fec_winners = fec.set_index("state")["winner"]
    all_states = sorted(set().union(*[s for _, s in groups]))

    pm_snap = pm_snapshot(pm, cutoff, all_states)
    p5_snap = p538_snapshot(p538, cutoff, all_states)

    rows = []
    for label, states in groups:
        pm_acc = compute_accuracy(pm_snap, fec_winners, states)
        p5_acc = compute_accuracy(p5_snap, fec_winners, states)
        rows.append(
            {
                "group": label,
                "pm_pct": pm_acc["pct"],
                "p538_pct": p5_acc["pct"],
            }
        )
    return pd.DataFrame(rows)


def build_polymarket_trajectory_metrics(pm, fec, snapshots, swing_states):
    """
    Build Polymarket accuracy at multiple date snapshots.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    fec : DataFrame
        FEC results data.
    snapshots : list of tuple
        Each tuple is (label, date_str, states_or_none).
    swing_states : list of str
        Swing state abbreviations.

    Returns
    -------
    DataFrame
        One row per snapshot with all_pct and swing_pct.
    """
    fec_winners = fec.set_index("state")["winner"]

    rows = []
    for label, as_of, state_subset in snapshots:
        snap = pm_snapshot(pm, as_of)
        if state_subset is None:
            states = sorted(snap["state"].unique())
        else:
            states = state_subset

        all_acc = compute_accuracy(snap, fec_winners, states)
        swing = sorted(set(swing_states) & set(snap["state"]))
        swing_acc = compute_accuracy(snap, fec_winners, swing)

        n = all_acc["n_states"]
        rows.append(
            {
                "label": f"{label}\n({n} states)",
                "all_pct": all_acc["pct"],
                "swing_pct": swing_acc["pct"],
            }
        )
    return pd.DataFrame(rows)


def build_ev_comparison_metrics(pm, p538, fec, cutoff, states):
    """
    Build Electoral Vote share metrics for Polymarket,
    538, and actual results.

    Parameters
    ----------
    pm : DataFrame
        Polymarket daily data.
    p538 : DataFrame
        FiveThirtyEight polling data.
    fec : DataFrame
        FEC results data.
    cutoff : str
        Date in YYYY-MM-DD format.
    states : list of str
        States to include.

    Returns
    -------
    DataFrame
        One row per source with EV share metrics.
    """
    pm_snap = pm_snapshot(pm, cutoff, states)
    p5_snap = p538_snapshot(p538, cutoff, states)
    actual = fec[["state", "winner"]].rename(columns={"winner": "predicted_winner"})

    dt = pd.Timestamp(cutoff)
    date_lbl = f"{dt.strftime('%b')} {dt.day}"

    rows = [
        {
            "label": f"Polymarket\n({date_lbl})",
            **compute_ev_share(pm_snap, fec, states),
        },
        {
            "label": f"538\n({date_lbl})",
            **compute_ev_share(p5_snap, fec, states),
        },
        {
            "label": "Actual\nResult",
            **compute_ev_share(actual, fec),
        },
    ]
    return pd.DataFrame(rows)


def _classify_pair(pm_preds, p538_preds, fec_winners, st):
    """
    Classify one state as both-right, both-wrong,
    only-pm-right, or only-538-right.

    Parameters
    ----------
    pm_preds : DataFrame
        Polymarket predictions with predicted_winner.
    p538_preds : DataFrame
        538 predictions with predicted_winner.
    fec_winners : Series
        Actual winners indexed by state.
    st : str
        State abbreviation.

    Returns
    -------
    str or None
        Classification label, or None if data missing.
    """
    pm_row = pm_preds[pm_preds["state"] == st]
    p538_row = p538_preds[p538_preds["state"] == st]
    if pm_row.empty or p538_row.empty:
        return None

    actual = fec_winners.loc[st]
    pm_ok = pm_row["predicted_winner"].iloc[0] == actual
    p538_ok = p538_row["predicted_winner"].iloc[0] == actual

    if pm_ok and p538_ok:
        return "both_right"
    elif pm_ok:
        return "only_pm_right"
    elif p538_ok:
        return "only_538_right"
    return "both_wrong"


def mcnemar_test(pm_preds, p538_preds, fec_winners, states):
    """
    Run McNemar's exact test on paired state-level
    winner calls.

    Parameters
    ----------
    pm_preds : DataFrame
        Polymarket predictions with predicted_winner.
    p538_preds : DataFrame
        538 predictions with predicted_winner.
    fec_winners : Series
        Actual winners indexed by state.
    states : list of str
        States to evaluate.

    Returns
    -------
    dict
        2x2 table counts, n_discordant, and p_value.
    """
    counts = {
        "both_right": 0,
        "both_wrong": 0,
        "only_pm_right": 0,
        "only_538_right": 0,
    }
    for st in states:
        label = _classify_pair(pm_preds, p538_preds, fec_winners, st)
        if label is not None:
            counts[label] += 1

    n_disc = counts["only_pm_right"] + counts["only_538_right"]
    if n_disc == 0:
        p_val = 1.0
    else:
        p_val = binomtest(counts["only_pm_right"], n_disc, 0.5).pvalue

    return {
        **counts,
        "n_discordant": n_disc,
        "p_value": p_val,
    }


def main():
    """
    Run accuracy analysis and print summary metrics.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pm, p538, fec = load_data(PROCESSED_DIR)

    # Head-to-head comparison on overlap cutoff
    groups = [
        ("All 13 States", OVERLAP_STATES),
        (
            f"{len(SWING_OVERLAP)} Swing States",
            SWING_OVERLAP,
        ),
    ]
    h2h = build_head_to_head_metrics(pm, p538, fec, OVERLAP_CUTOFF, groups)
    print("Head-to-head accuracy:")
    print(h2h)

    # Polymarket trajectory
    snapshots = [
        ("Sept 12", OVERLAP_CUTOFF, OVERLAP_STATES),
        ("Oct 6", THIRTY_DAYS_OUT, None),
        ("Nov 4", ELECTION_EVE, None),
    ]
    traj = build_polymarket_trajectory_metrics(pm, fec, snapshots, SWING_STATES)
    print("\nPolymarket trajectory:")
    print(traj)


if __name__ == "__main__":
    main()
