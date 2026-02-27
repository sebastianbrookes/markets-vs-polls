"""
Analyze how Polymarket and FiveThirtyEight responded to major 2024
campaign events — shift size, direction, and reaction speed.

Five events analyzed (all within the March–Sept 12 overlap period):
    1. Biden-Trump Debate (June 27)
    2. Trump Assassination Attempt (July 13)
    3. Biden Drops Out (July 21)
    4. Walz VP Pick (Aug 6)
    5. Harris-Trump Debate (Sept 10)

Run from project root:
    python -m src.analysis.events.event_response
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.clean.utils import PROCESSED_DIR, SWING_STATES

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
EVENTS = [
    {
        "name": "Biden-Trump Debate",
        "date": "2024-06-27",
        "expected_direction": "pro-Trump",
        "reason": "Biden performed terribly",
    },
    {
        "name": "Assassination Attempt",
        "date": "2024-07-13",
        "expected_direction": "pro-Trump",
        "reason": "Sympathy rally, show of resilience",
    },
    {
        "name": "Biden Drops Out",
        "date": "2024-07-21",
        "expected_direction": "pro-Harris",
        "reason": "Replaced weak candidate with stronger one",
    },
    {
        "name": "Walz VP Pick",
        "date": "2024-08-06",
        "expected_direction": "neutral",
        "reason": "Safe pick, no big electoral impact",
    },
    {
        "name": "Harris-Trump Debate",
        "date": "2024-09-10",
        "expected_direction": "pro-Harris",
        "reason": "Harris widely seen as winning",
    },
]

OVERLAP_CUTOFF = "2024-09-12"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load processed daily CSVs for Polymarket and 538."""
    pm = pd.read_csv(PROCESSED_DIR / "polymarket_daily.csv",
                     parse_dates=["date"])
    p538 = pd.read_csv(PROCESSED_DIR / "polls_538.csv",
                       parse_dates=["date"])
    p538 = p538[p538["state"] != "US"].copy()
    return pm, p538


def load_hourly():
    """Load Polymarket hourly data with proper datetime parsing."""
    hourly = pd.read_csv(PROCESSED_DIR / "polymarket_hourly.csv")
    hourly["date_utc"] = pd.to_datetime(hourly["date_utc"])
    return hourly


# ---------------------------------------------------------------------------
# Swing-state averaging
# ---------------------------------------------------------------------------
def compute_swing_average(df, date_col="date", value_col="trump_lead"):
    """Average trump_lead across 7 swing states per date.

    Returns a Series indexed by date with the mean trump_lead.
    """
    swing = df[df["state"].isin(SWING_STATES)].copy()
    avg = (swing
           .groupby(date_col)[value_col]
           .mean()
           .sort_index())
    return avg


def compute_hourly_swing_average(hourly):
    """Average trump_lead across swing states per hour (date_utc)."""
    swing = hourly[hourly["state"].isin(SWING_STATES)].copy()
    avg = (swing
           .groupby("date_utc")["trump_lead"]
           .mean()
           .sort_index())
    return avg


def _latest_per_state(p538, as_of):
    """Forward-fill 538: latest row per state on or before *as_of*."""
    mask = p538["date"] <= pd.Timestamp(as_of)
    subset = p538[mask].sort_values(["state", "date"])
    return subset.drop_duplicates(subset="state", keep="last")


def compute_538_swing_timeseries(p538):
    """Build a daily swing-state average for 538, forward-filling gaps.

    For each date in the range, take the latest available value per
    swing state, then average across states.
    """
    start = p538[p538["state"].isin(SWING_STATES)]["date"].min()
    end = pd.Timestamp(OVERLAP_CUTOFF)
    dates = pd.date_range(start, end, freq="D")

    records = []
    for d in dates:
        snap = _latest_per_state(p538, d)
        snap = snap[snap["state"].isin(SWING_STATES)]
        if snap.empty:
            continue
        records.append({"date": d, "trump_lead": snap["trump_lead"].mean()})

    return pd.Series(
        [r["trump_lead"] for r in records],
        index=pd.DatetimeIndex([r["date"] for r in records]),
    )


# ---------------------------------------------------------------------------
# Event reaction computation
# ---------------------------------------------------------------------------
def compute_event_reaction(swing_avg, event_date, pre_days=3, post_days=7):
    """Compute baseline, immediate, and settled reaction around an event.

    Parameters
    ----------
    swing_avg : pd.Series
        Daily swing-state average trump_lead, indexed by date.
    event_date : str or pd.Timestamp
        Date of the event.
    pre_days : int
        Number of days before event for baseline window.
    post_days : int
        Number of days after event for settled window.

    Returns
    -------
    dict with keys: baseline, immediate, settled, shift, has_data,
                    n_pre, n_post, data_gap_days
    """
    event_date = pd.Timestamp(event_date)
    pre_start = event_date - pd.Timedelta(days=pre_days)
    pre_end = event_date - pd.Timedelta(days=1)
    post_start = event_date + pd.Timedelta(days=2)
    post_end = event_date + pd.Timedelta(days=post_days)

    pre_mask = (swing_avg.index >= pre_start) & (swing_avg.index <= pre_end)
    imm_mask = (swing_avg.index >= event_date) & \
               (swing_avg.index <= event_date + pd.Timedelta(days=1))
    post_mask = (swing_avg.index >= post_start) & \
                (swing_avg.index <= post_end)

    pre_vals = swing_avg[pre_mask]
    imm_vals = swing_avg[imm_mask]
    post_vals = swing_avg[post_mask]

    baseline = pre_vals.mean() if len(pre_vals) > 0 else np.nan
    immediate = imm_vals.mean() if len(imm_vals) > 0 else np.nan
    settled = post_vals.mean() if len(post_vals) > 0 else np.nan
    shift = settled - baseline if not (np.isnan(settled) or
                                       np.isnan(baseline)) else np.nan

    # Detect data gaps: days in post window with no data
    if len(post_vals) > 0:
        expected_days = (post_end - post_start).days + 1
        data_gap_days = expected_days - len(post_vals)
    else:
        data_gap_days = post_days

    return {
        "baseline": baseline,
        "immediate": immediate,
        "settled": settled,
        "shift": shift,
        "has_data": len(pre_vals) > 0 and len(post_vals) > 0,
        "n_pre": len(pre_vals),
        "n_post": len(post_vals),
        "data_gap_days": data_gap_days,
    }


def classify_direction(shift, expected):
    """Classify whether the shift matched the expected direction."""
    if np.isnan(shift):
        return "no data"
    if expected == "neutral":
        return "correct" if abs(shift) < 0.01 else "overreaction"
    if expected == "pro-Trump":
        return "correct" if shift > 0 else "wrong"
    if expected == "pro-Harris":
        return "correct" if shift < 0 else "wrong"
    return "unknown"


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------
def build_reaction_summary(pm, p538):
    """Build a DataFrame summarizing reactions for all events × both sources."""
    pm_swing = compute_swing_average(pm)
    p538_swing = compute_538_swing_timeseries(p538)

    rows = []
    for event in EVENTS:
        for source, swing in [("Polymarket", pm_swing),
                              ("538", p538_swing)]:
            reaction = compute_event_reaction(swing, event["date"])
            direction = classify_direction(
                reaction["shift"], event["expected_direction"]
            )
            rows.append({
                "event": event["name"],
                "event_date": event["date"],
                "source": source,
                "expected": event["expected_direction"],
                "baseline": reaction["baseline"],
                "immediate": reaction["immediate"],
                "settled": reaction["settled"],
                "shift": reaction["shift"],
                "direction_match": direction,
                "has_data": reaction["has_data"],
                "n_pre": reaction["n_pre"],
                "n_post": reaction["n_post"],
                "data_gap_days": reaction["data_gap_days"],
            })

    df = pd.DataFrame(rows)

    # Background jitter: std of day-over-day changes for each source
    bg_std = {
        "Polymarket": pm_swing.diff().dropna().std(),
        "538": p538_swing.diff().dropna().std(),
    }

    df["shift_z"] = df.apply(
        lambda r: r["shift"] / bg_std[r["source"]]
        if bg_std.get(r["source"], 0) and not np.isnan(r["shift"])
        else np.nan,
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 65)
    print("  EVENT RESPONSE ANALYSIS: POLYMARKET vs. FIVETHIRTYEIGHT")
    print("=" * 65 + "\n")

    pm, p538 = load_data()
    summary = build_reaction_summary(pm, p538)

    # Print per-event comparison
    for event in EVENTS:
        name = event["name"]
        print(f"{'─' * 65}")
        print(f"  {name}  ({event['date']})  "
              f"— expected: {event['expected_direction']}")
        print(f"  {event['reason']}")
        print(f"{'─' * 65}")

        for source in ["Polymarket", "538"]:
            row = summary[(summary["event"] == name) &
                          (summary["source"] == source)].iloc[0]

            if not row["has_data"]:
                print(f"  {source:12s}: INSUFFICIENT DATA "
                      f"(pre={row['n_pre']}, post={row['n_post']})")
                continue

            shift_str = f"{row['shift']:+.4f}" if not np.isnan(
                row['shift']) else "N/A"
            match_str = row["direction_match"].upper()
            gap_str = (f"  ({row['data_gap_days']} gap days)"
                       if row["data_gap_days"] > 0 else "")

            print(f"  {source:12s}: baseline={row['baseline']:.4f}  "
                  f"settled={row['settled']:.4f}  "
                  f"shift={shift_str}  [{match_str}]{gap_str}")

        print()

    # Summary scorecard
    print("=" * 65)
    print("  SCORECARD")
    print("=" * 65)
    for source in ["Polymarket", "538"]:
        src_rows = summary[summary["source"] == source]
        n_correct = (src_rows["direction_match"] == "correct").sum()
        n_with_data = src_rows["has_data"].sum()
        total_gap = src_rows["data_gap_days"].sum()
        print(f"  {source:12s}: {n_correct}/{len(src_rows)} correct direction  "
              f"| {int(n_with_data)}/{len(src_rows)} events with data  "
              f"| {int(total_gap)} total gap days")

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
