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
        "reason": "Display of resilience",
    },
    {
        "name": "Biden Drops Out",
        "date": "2024-07-21",
        "expected_direction": "pro-Harris",
        "reason": "Replaced struggling candidate with stronger one",
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
MIN_POST_COVERAGE = 1.0
RULE_WIDTH = 65

_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.clean.utils import PROCESSED_DIR, SWING_STATES


def load_daily_data(processed_dir):
    """Load processed daily CSVs for Polymarket and 538.

    Returns (polymarket_df, state_polls_df) with the national
    538 row removed.
    """
    polymarket_df = pd.read_csv(
        processed_dir / "polymarket_daily.csv",
        parse_dates=["date"],
    )
    polls_df = pd.read_csv(
        processed_dir / "polls_538.csv",
        parse_dates=["date"],
    )
    state_polls_df = polls_df[polls_df["state"] != "US"].copy()
    return polymarket_df, state_polls_df


def load_hourly_data(processed_dir):
    """Load Polymarket hourly data with proper datetime parsing."""
    hourly_df = pd.read_csv(processed_dir / "polymarket_hourly.csv")
    hourly_df["date_utc"] = pd.to_datetime(hourly_df["date_utc"])
    return hourly_df


def compute_swing_average(
    dataframe,
    swing_states,
    date_column="date",
    value_column="trump_lead",
):
    """Average trump_lead across swing states per date.

    Returns a Series indexed by date.
    """
    swing_df = dataframe[dataframe["state"].isin(swing_states)].copy()
    return swing_df.groupby(date_column)[value_column].mean().sort_index()


def compute_hourly_swing_average(hourly_df, swing_states):
    """Average trump_lead across swing states per hour (date_utc)."""
    swing_df = hourly_df[hourly_df["state"].isin(swing_states)].copy()
    return swing_df.groupby("date_utc")["trump_lead"].mean().sort_index()


def get_latest_state_snapshot(polls_df, as_of_date):
    """Forward-fill 538: latest row per state on or before as_of_date."""
    eligible_rows = polls_df[polls_df["date"] <= pd.Timestamp(as_of_date)]
    sorted_rows = eligible_rows.sort_values(["state", "date"])
    return sorted_rows.drop_duplicates(subset="state", keep="last")


def build_fivethirtyeight_swing_series(polls_df, swing_states, overlap_cutoff):
    """Build a daily swing-state average for 538, forward-filling gaps.

    For each date in the range, take the latest available value per
    swing state, then average across states. Returns a Series indexed
    by date.
    """
    swing_polls_df = polls_df[polls_df["state"].isin(swing_states)]
    start_date = swing_polls_df["date"].min()
    end_date = pd.Timestamp(overlap_cutoff)
    all_dates = pd.date_range(start_date, end_date, freq="D")

    records = []
    for current_date in all_dates:
        snapshot_df = get_latest_state_snapshot(polls_df, current_date)
        swing_snapshot_df = snapshot_df[snapshot_df["state"].isin(swing_states)]
        if not swing_snapshot_df.empty:
            records.append({
                "date": current_date,
                "trump_lead": swing_snapshot_df["trump_lead"].mean(),
            })

    return pd.Series(
        [r["trump_lead"] for r in records],
        index=pd.DatetimeIndex([r["date"] for r in records]),
    )


# ---------------------------------------------------------------------------
# Event reaction computation
# ---------------------------------------------------------------------------

def compute_event_reaction(
    swing_series,
    event_date,
    pre_days=3,
    post_days=7,
    min_post_coverage=MIN_POST_COVERAGE,
):
    """Compute baseline, immediate, and settled reaction around an event.

    Parameters
    ----------
    swing_series : pd.Series
        Daily swing-state average trump_lead, indexed by date.
    event_date : str or pd.Timestamp
        Date of the event.
    pre_days : int
        Number of days before event for baseline window.
    post_days : int
        Number of days after event for settled window.
    min_post_coverage : float
        Minimum fraction of post-event days required.

    Returns
    -------
    dict with keys: baseline, immediate, settled, shift, has_data,
                    n_pre, n_post, data_gap_days, post_coverage
    """
    ev = pd.Timestamp(event_date)
    pre_start = ev - pd.Timedelta(days=pre_days)
    pre_end = ev - pd.Timedelta(days=1)
    post_start = ev + pd.Timedelta(days=2)
    post_end = ev + pd.Timedelta(days=post_days)

    pre_vals = swing_series[
        (swing_series.index >= pre_start) & (swing_series.index <= pre_end)
    ]
    imm_vals = swing_series[
        (swing_series.index >= ev)
        & (swing_series.index <= ev + pd.Timedelta(days=1))
    ]
    post_vals = swing_series[
        (swing_series.index >= post_start) & (swing_series.index <= post_end)
    ]

    baseline = pre_vals.mean() if len(pre_vals) > 0 else np.nan
    immediate = imm_vals.mean() if len(imm_vals) > 0 else np.nan
    settled = post_vals.mean() if len(post_vals) > 0 else np.nan
    shift = (
        settled - baseline
        if not (np.isnan(settled) or np.isnan(baseline))
        else np.nan
    )

    expected_days = (post_end - post_start).days + 1
    data_gap_days = expected_days - len(post_vals)
    post_coverage = len(post_vals) / expected_days
    has_data = len(pre_vals) > 0 and post_coverage >= min_post_coverage

    return {
        "baseline": baseline,
        "immediate": immediate,
        "settled": settled,
        "shift": shift,
        "has_data": has_data,
        "n_pre": len(pre_vals),
        "n_post": len(post_vals),
        "data_gap_days": data_gap_days,
        "post_coverage": post_coverage,
    }


def _zeroed_event_window(swing_series, event_date, pre_days, post_days):
    """Extract an event window, relabel as day offsets, zero from Day -1.

    Returns a Series indexed by integer day offset with Day -1 = 0,
    or an empty Series if Day -1 has no data.
    """
    ev = pd.Timestamp(event_date)
    start = ev - pd.Timedelta(days=pre_days)
    end = ev + pd.Timedelta(days=post_days)

    window = swing_series[
        (swing_series.index >= start) & (swing_series.index <= end)
    ].copy()
    window.index = (window.index - ev).days

    if -1 not in window.index:
        return pd.Series(dtype=float)

    return window - window.loc[-1]


def compute_indexed_window(
    swing_series,
    event_date,
    pre_days=2,
    post_days=10,
    normalize=True,
):
    """Zero-indexed window around an event for small-multiple plotting.

    Returns a Series indexed by integer day offset, values = change
    from Day -1. If normalize=True, divides by daily volatility to
    produce z-scored changes.
    """
    window = _zeroed_event_window(swing_series, event_date, pre_days, post_days)
    if window.empty or not normalize:
        return window

    daily_vol = swing_series.diff().dropna().std()
    if daily_vol > 0:
        window = window / daily_vol
    return window


def compute_raw_indexed_window(
    swing_series,
    event_date,
    pre_days=2,
    post_days=10,
    scale=1,
):
    """Zero-indexed window in raw units (not z-scored).

    Parameters
    ----------
    scale : float
        Multiply values by this factor (e.g. 100 to convert
        probability to percentage points).
    """
    window = _zeroed_event_window(swing_series, event_date, pre_days, post_days)
    return window * scale


def detect_price_in_day(window_series, threshold_pct=0.90):
    """Find the first day >= 0 where the source reaches threshold_pct
    of its total move (Day 10 value). Returns the integer day offset,
    or None if never crossed.
    """
    if window_series.empty or 10 not in window_series.index:
        return None

    total_move = window_series.loc[10]
    if total_move == 0:
        return None

    threshold = threshold_pct * total_move
    post_event_values = window_series[window_series.index >= 0].sort_index()

    for day_offset, value in post_event_values.items():
        if total_move > 0 and value >= threshold:
            return int(day_offset)
        if total_move < 0 and value <= threshold:
            return int(day_offset)

    return None


def classify_direction(shift, expected_direction):
    """Classify whether the shift matched the expected direction."""
    if np.isnan(shift):
        return "no data"
    if expected_direction == "neutral":
        return "correct" if abs(shift) < 0.01 else "overreaction"
    if expected_direction == "pro-Trump":
        return "correct" if shift > 0 else "wrong"
    if expected_direction == "pro-Harris":
        return "correct" if shift < 0 else "wrong"
    return "unknown"


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_reaction_summary(
    events,
    polymarket_df,
    polls_df,
    swing_states,
    overlap_cutoff,
    min_post_coverage=1.0,
):
    """Build a DataFrame summarizing reactions for all events x both sources."""
    source_series = {
        "Polymarket": compute_swing_average(polymarket_df, swing_states),
        "538": build_fivethirtyeight_swing_series(
            polls_df, swing_states, overlap_cutoff,
        ),
    }

    rows = []
    for event in events:
        for source_name, swing in source_series.items():
            reaction = compute_event_reaction(
                swing, event["date"], min_post_coverage=min_post_coverage,
            )
            rows.append({
                "event": event["name"],
                "event_date": event["date"],
                "source": source_name,
                "expected": event["expected_direction"],
                "baseline": reaction["baseline"],
                "immediate": reaction["immediate"],
                "settled": reaction["settled"],
                "shift": reaction["shift"],
                "direction_match": classify_direction(
                    reaction["shift"], event["expected_direction"],
                ),
                "has_data": reaction["has_data"],
                "n_pre": reaction["n_pre"],
                "n_post": reaction["n_post"],
                "data_gap_days": reaction["data_gap_days"],
                "post_coverage": reaction["post_coverage"],
            })

    df = pd.DataFrame(rows)

    # z-score shifts by each source's background daily volatility
    bg_std = {
        name: s.diff().dropna().std()
        for name, s in source_series.items()
    }
    df["shift_z"] = df.apply(
        lambda r: r["shift"] / bg_std[r["source"]]
        if bg_std.get(r["source"], 0) and not np.isnan(r["shift"])
        else np.nan,
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def _print_source_result(row):
    """Print one formatted result line for a source."""
    if not row["has_data"]:
        print(
            f"  {row['source']:12s}: INSUFFICIENT DATA "
            f"(pre={row['n_pre']}, post={row['n_post']}, "
            f"post_coverage={row['post_coverage']:.0%})"
        )
        return

    shift_str = f"{row['shift']:+.4f}" if not np.isnan(row["shift"]) else "N/A"
    match_str = row["direction_match"].upper()
    gap_str = (
        f"  ({row['data_gap_days']} gap days)" if row["data_gap_days"] > 0 else ""
    )

    print(
        f"  {row['source']:12s}: baseline={row['baseline']:.4f}  "
        f"settled={row['settled']:.4f}  "
        f"shift={shift_str}  [{match_str}]{gap_str}"
    )


def _print_event_summary(event, summary_df):
    """Print the formatted comparison for one event."""
    print("─" * RULE_WIDTH)
    print(
        f"  {event['name']}  ({event['date']})  "
        f"— expected: {event['expected_direction']}"
    )
    print(f"  {event['reason']}")
    print("─" * RULE_WIDTH)

    for source_name in ["Polymarket", "538"]:
        row = summary_df[
            (summary_df["event"] == event["name"])
            & (summary_df["source"] == source_name)
        ].iloc[0]
        _print_source_result(row)

    print()


def _print_scorecard(summary_df):
    """Print the overall direction-match scorecard for each source."""
    print("\n" + "=" * RULE_WIDTH)
    print("  SCORECARD")
    print("=" * RULE_WIDTH)
    for source_name in ["Polymarket", "538"]:
        src_rows = summary_df[summary_df["source"] == source_name]
        eligible = src_rows[src_rows["has_data"]]
        n_correct = (eligible["direction_match"] == "correct").sum()
        n_eligible = len(eligible)
        n_with_data = int(src_rows["has_data"].sum())
        total_gap = int(src_rows["data_gap_days"].sum())
        print(
            f"  {source_name:12s}: {n_correct}/{n_eligible} correct direction  "
            f"| {n_with_data}/{len(src_rows)} events with data  "
            f"| {total_gap} total gap days"
        )


def main():
    """Load data, compute event responses, and print the analysis report."""
    print("\n" + "=" * RULE_WIDTH)
    print("  EVENT RESPONSE ANALYSIS: POLYMARKET vs. 538")
    print("=" * RULE_WIDTH)

    polymarket_df, polls_df = load_daily_data(PROCESSED_DIR)
    summary_df = build_reaction_summary(
        EVENTS, polymarket_df, polls_df,
        SWING_STATES, OVERLAP_CUTOFF, MIN_POST_COVERAGE,
    )

    for event in EVENTS:
        _print_event_summary(event, summary_df)

    _print_scorecard(summary_df)

    print("\n" + "=" * RULE_WIDTH)
    print("  Analysis complete.")
    print("=" * RULE_WIDTH + "\n")


if __name__ == "__main__":
    main()
