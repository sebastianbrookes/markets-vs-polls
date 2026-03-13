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


def add_project_root_to_path():
    """
    Adds the project root to sys.path so src imports work correctly.
    """
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


add_project_root_to_path()

from src.clean.utils import PROCESSED_DIR, SWING_STATES


def load_daily_data(processed_dir):
    """
    Given a processed data directory, loads the daily analysis data.

    Returns a tuple with Polymarket daily data and FiveThirtyEight
    state polling data with the national row removed.
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
    """
    Given a processed data directory, loads hourly Polymarket data.

    Returns a DataFrame with a parsed UTC datetime column.
    """
    hourly_df = pd.read_csv(processed_dir / "polymarket_hourly.csv")
    hourly_df["date_utc"] = pd.to_datetime(hourly_df["date_utc"])
    return hourly_df


def compute_swing_average(
    dataframe,
    swing_states,
    date_column="date",
    value_column="trump_lead",
):
    """
    Given a state-level DataFrame, computes the daily swing-state mean.

    Returns a Series indexed by date with the average Trump lead
    across the requested swing states.
    """
    swing_df = dataframe[dataframe["state"].isin(swing_states)].copy()
    return swing_df.groupby(date_column)[value_column].mean().sort_index()


def compute_hourly_swing_average(hourly_df, swing_states):
    """
    Given hourly state-level Polymarket data, computes the hourly
    swing-state mean Trump lead.

    Returns a Series indexed by date_utc.
    """
    swing_df = hourly_df[hourly_df["state"].isin(swing_states)].copy()
    return swing_df.groupby("date_utc")["trump_lead"].mean().sort_index()


def get_latest_state_snapshot(polls_df, as_of_date):
    """
    Given FiveThirtyEight polling data and a cutoff date, finds the
    latest available row for each state on or before that date.

    Returns a DataFrame with one row per state.

    This is the key step for manual forward-fill: for each state,
    keep the most recent poll at or before the target date.
    """
    eligible_rows = polls_df[polls_df["date"] <= pd.Timestamp(as_of_date)]
    sorted_rows = eligible_rows.sort_values(["state", "date"])
    return sorted_rows.drop_duplicates(subset="state", keep="last")


def collect_fivethirtyeight_records(polls_df, swing_states, all_dates):
    """
    Given FiveThirtyEight polling data and a date range, builds one
    forward-filled swing-state average record per day.

    Returns a list of dictionaries.

    Because 538 polls are irregular, we rebuild a daily series by
    taking a daily snapshot of each state's latest available value.
    """
    records = []
    for current_date in all_dates:
        snapshot_df = get_latest_state_snapshot(polls_df, current_date)
        swing_snapshot_df = snapshot_df[snapshot_df["state"].isin(swing_states)]
        if not swing_snapshot_df.empty:
            records.append(
                {
                    "date": current_date,
                    "trump_lead": swing_snapshot_df["trump_lead"].mean(),
                }
            )
    return records


def build_fivethirtyeight_swing_series(
    polls_df,
    swing_states,
    overlap_cutoff,
):
    """
    Builds a daily swing-state time series for FiveThirtyEight.

    For each day in the overlap window, this function forward-fills by
    taking the latest available poll value for each swing state.

    Returns a Series indexed by date.
    """
    swing_polls_df = polls_df[polls_df["state"].isin(swing_states)]
    start_date = swing_polls_df["date"].min()
    end_date = pd.Timestamp(overlap_cutoff)
    all_dates = pd.date_range(start_date, end_date, freq="D")
    records = collect_fivethirtyeight_records(
        polls_df,
        swing_states,
        all_dates,
    )

    return pd.Series(
        [record["trump_lead"] for record in records],
        index=pd.DatetimeIndex([record["date"] for record in records]),
    )


def get_event_windows(event_date, pre_days, post_days):
    """
    Given an event date and window lengths, returns the date ranges
    used for the baseline, immediate reaction, and settled reaction.
    """
    event_timestamp = pd.Timestamp(event_date)
    baseline_window = (
        event_timestamp - pd.Timedelta(days=pre_days),
        event_timestamp - pd.Timedelta(days=1),
    )
    immediate_window = (
        event_timestamp,
        event_timestamp + pd.Timedelta(days=1),
    )
    settled_window = (
        event_timestamp + pd.Timedelta(days=2),
        event_timestamp + pd.Timedelta(days=post_days),
    )
    return baseline_window, immediate_window, settled_window


def select_series_window(series, start_date, end_date):
    """
    Given a dated Series and a date range, returns the values inside
    that inclusive range.
    """
    mask = (series.index >= start_date) & (series.index <= end_date)
    return series[mask]


def calculate_mean_or_nan(series):
    """
    Given a Series, returns its mean or NaN when it is empty.
    """
    if len(series) == 0:
        return np.nan
    return series.mean()


def compute_shift(baseline, settled):
    """
    Given a baseline value and a settled value, returns the post-event
    shift or NaN if either value is missing.
    """
    if np.isnan(baseline) or np.isnan(settled):
        return np.nan
    return settled - baseline


def compute_post_coverage_stats(settled_values, settled_window):
    """
    Given settled-window values, computes post-event coverage stats.

    Returns a tuple with the number of expected days, gap days, and
    coverage share.
    """
    settled_start, settled_end = settled_window
    expected_days = (settled_end - settled_start).days + 1
    data_gap_days = expected_days - len(settled_values)
    post_coverage = len(settled_values) / expected_days
    return expected_days, data_gap_days, post_coverage


def summarize_reaction_windows(
    baseline_values,
    immediate_values,
    settled_values,
    settled_window,
    min_post_coverage,
):
    """Computes summary metrics for one event window.

    An event is marked usable only if it has pre-event data and enough
    settled-window coverage after the event.
    """
    baseline = calculate_mean_or_nan(baseline_values)
    immediate = calculate_mean_or_nan(immediate_values)
    settled = calculate_mean_or_nan(settled_values)
    shift = compute_shift(baseline, settled)
    _, data_gap_days, post_coverage = compute_post_coverage_stats(
        settled_values,
        settled_window,
    )
    has_data = len(baseline_values) > 0 and post_coverage >= min_post_coverage

    return {
        "baseline": baseline,
        "immediate": immediate,
        "settled": settled,
        "shift": shift,
        "has_data": has_data,
        "n_pre": len(baseline_values),
        "n_post": len(settled_values),
        "data_gap_days": data_gap_days,
        "post_coverage": post_coverage,
    }


def compute_event_reaction(
    swing_series,
    event_date,
    pre_days=3,
    post_days=7,
    min_post_coverage=1.0,
):
    """Computes baseline and post-event reaction statistics."""
    baseline_window, immediate_window, settled_window = get_event_windows(
        event_date,
        pre_days,
        post_days,
    )
    baseline_values = select_series_window(swing_series, *baseline_window)
    immediate_values = select_series_window(swing_series, *immediate_window)
    settled_values = select_series_window(swing_series, *settled_window)

    return summarize_reaction_windows(
        baseline_values,
        immediate_values,
        settled_values,
        settled_window,
        min_post_coverage,
    )


def build_relative_window(swing_series, event_date, pre_days, post_days):
    """
    Given a swing-state Series and an event date, extracts the event
    window and relabels the index as integer day offsets.

    Returns a Series indexed by days relative to the event.
    """
    event_timestamp = pd.Timestamp(event_date)
    start_date = event_timestamp - pd.Timedelta(days=pre_days)
    end_date = event_timestamp + pd.Timedelta(days=post_days)
    window_series = select_series_window(swing_series, start_date, end_date)
    day_offsets = (window_series.index - event_timestamp).days
    relative_window = window_series.copy()
    relative_window.index = day_offsets
    return relative_window


def zero_from_day_before(relative_window):
    """
    Given a relative event window, subtracts the Day -1 value from the
    whole window.

    Returns an empty Series if Day -1 is missing.

    Day -1 is our event baseline so every event window is compared
    from a common starting point.
    """
    if -1 not in relative_window.index:
        return pd.Series(dtype=float)

    baseline = relative_window.loc[-1]
    return relative_window - baseline


def normalize_indexed_window(indexed_window, swing_series):
    """
    Given an indexed event window, divides it by the standard
    deviation of daily changes when possible.

    Returns a Series in normalized units.
    """
    daily_volatility = swing_series.diff().dropna().std()
    if daily_volatility > 0:
        return indexed_window / daily_volatility
    return indexed_window


def compute_indexed_window(
    swing_series,
    event_date,
    pre_days=2,
    post_days=10,
    normalize=True,
):
    """
    Builds an event-centered window of day offsets and changes from
    Day -1.

    Returns a Series indexed by integer day offset.
    """
    relative_window = build_relative_window(
        swing_series,
        event_date,
        pre_days,
        post_days,
    )
    indexed_window = zero_from_day_before(relative_window)
    if indexed_window.empty or not normalize:
        return indexed_window
    return normalize_indexed_window(indexed_window, swing_series)


def compute_raw_indexed_window(
    swing_series,
    event_date,
    pre_days=2,
    post_days=10,
    scale=1,
):
    """
    Builds an event-centered window in raw units instead of z-scores.

    Returns a Series indexed by integer day offset.
    """
    relative_window = build_relative_window(
        swing_series,
        event_date,
        pre_days,
        post_days,
    )
    indexed_window = zero_from_day_before(relative_window)
    return indexed_window * scale


def detect_price_in_day(window_series, threshold_pct=0.90):
    """
    Given a zeroed event window, finds the first post-event day that
    reaches the requested share of the total Day 10 move.

    Returns an integer day offset or None.

    We define reaction speed as the first day that reaches 90% of the
    final Day 10 move, with sign-aware logic for up vs. down moves.
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
    """
    Given an observed shift and an expected direction, labels whether
    the result matched the expectation.

    For "neutral" events, shifts within +/-0.01 are treated as no
    meaningful movement.
    """
    if np.isnan(shift):
        return "no data"

    if expected_direction == "neutral":
        if abs(shift) < 0.01:
            return "correct"
        return "overreaction"

    if expected_direction == "pro-Trump":
        if shift > 0:
            return "correct"
        return "wrong"

    if expected_direction == "pro-Harris":
        if shift < 0:
            return "correct"
        return "wrong"

    return "unknown"


def build_source_series(polymarket_df, polls_df, swing_states, overlap_cutoff):
    """
    Given the two source DataFrames, builds one swing-state Series for
    each source.

    Returns a dictionary mapping source names to Series objects.
    """
    return {
        "Polymarket": compute_swing_average(polymarket_df, swing_states),
        "538": build_fivethirtyeight_swing_series(
            polls_df,
            swing_states,
            overlap_cutoff,
        ),
    }


def build_summary_row(event, source_name, swing_series, min_post_coverage):
    """Builds one summary row for an event-source pair."""
    reaction = compute_event_reaction(
        swing_series,
        event["date"],
        min_post_coverage=min_post_coverage,
    )
    return {
        "event": event["name"],
        "event_date": event["date"],
        "source": source_name,
        "expected": event["expected_direction"],
        "baseline": reaction["baseline"],
        "immediate": reaction["immediate"],
        "settled": reaction["settled"],
        "shift": reaction["shift"],
        "direction_match": classify_direction(
            reaction["shift"],
            event["expected_direction"],
        ),
        "has_data": reaction["has_data"],
        "n_pre": reaction["n_pre"],
        "n_post": reaction["n_post"],
        "data_gap_days": reaction["data_gap_days"],
        "post_coverage": reaction["post_coverage"],
    }


def build_summary_rows(events, source_series, min_post_coverage):
    """
    Given the event list and source time series, builds summary rows
    for every event-source pair.

    Returns a list of dictionaries.
    """
    rows = []
    for event in events:
        for source_name, swing_series in source_series.items():
            row = build_summary_row(
                event,
                source_name,
                swing_series,
                min_post_coverage,
            )
            rows.append(row)
    return rows


def add_shift_z_scores(summary_df, source_series):
    """
    Given the event summary table, adds a z-scored shift column based
    on each source's background day-to-day volatility.

    Returns the updated DataFrame.
    """
    background_std = {}
    for source_name, swing_series in source_series.items():
        background_std[source_name] = swing_series.diff().dropna().std()

    summary_df["shift_z"] = summary_df.apply(
        lambda row: compute_shift_z_score(row, background_std),
        axis=1,
    )
    return summary_df


def compute_shift_z_score(row, background_std):
    """
    Given one summary row and a dictionary of source volatilities,
    computes the z-scored shift for that row.

    Returns a float or NaN.
    """
    source_name = row["source"]
    shift = row["shift"]
    source_std = background_std.get(source_name, 0)

    if np.isnan(shift) or source_std == 0:
        return np.nan

    return shift / source_std


def build_reaction_summary(
    events,
    polymarket_df,
    polls_df,
    swing_states,
    overlap_cutoff,
    min_post_coverage=1.0,
):
    """
    Given the source DataFrames, builds a complete event response
    summary table.

    Returns a DataFrame with one row per event-source pair.
    """
    source_series = build_source_series(
        polymarket_df,
        polls_df,
        swing_states,
        overlap_cutoff,
    )
    summary_rows = build_summary_rows(
        events,
        source_series,
        min_post_coverage,
    )
    summary_df = pd.DataFrame(summary_rows)
    return add_shift_z_scores(summary_df, source_series)


def print_rule(rule_width):
    """
    Prints a horizontal separator line.
    """
    print("─" * rule_width)


def print_box_header(title, rule_width):
    """
    Prints a boxed section header.
    """
    print("\n" + "=" * rule_width)
    print(f"  {title}")
    print("=" * rule_width)


def format_shift(row):
    """
    Given a summary row, returns a formatted shift string.
    """
    if np.isnan(row["shift"]):
        return "N/A"
    return f"{row['shift']:+.4f}"


def format_gap_text(row):
    """
    Given a summary row, returns formatted gap-day text.
    """
    if row["data_gap_days"] > 0:
        return f"  ({row['data_gap_days']} gap days)"
    return ""


def print_source_result(row):
    """
    Prints one formatted result line for a source.
    """
    if not row["has_data"]:
        print(
            f"  {row['source']:12s}: INSUFFICIENT DATA "
            f"(pre={row['n_pre']}, post={row['n_post']}, "
            f"post_coverage={row['post_coverage']:.0%})"
        )
        return

    shift_text = format_shift(row)
    match_text = row["direction_match"].upper()
    gap_text = format_gap_text(row)

    print(
        f"  {row['source']:12s}: baseline={row['baseline']:.4f}  "
        f"settled={row['settled']:.4f}  "
        f"shift={shift_text}  [{match_text}]{gap_text}"
    )


def get_event_row(summary_df, event_name, source_name):
    """
    Given the summary table, returns the row for one event-source pair.
    """
    row_df = summary_df[
        (summary_df["event"] == event_name) & (summary_df["source"] == source_name)
    ]
    return row_df.iloc[0]


def print_event_summary(event, summary_df, rule_width):
    """
    Prints the formatted comparison for one event.
    """
    print_rule(rule_width)
    print(
        f"  {event['name']}  ({event['date']})  "
        f"— expected: {event['expected_direction']}"
    )
    print(f"  {event['reason']}")
    print_rule(rule_width)

    for source_name in ["Polymarket", "538"]:
        event_row = get_event_row(summary_df, event["name"], source_name)
        print_source_result(event_row)

    print()


def print_scorecard(summary_df, rule_width):
    """
    Prints the overall direction-match scorecard for each source.
    """
    print_box_header("SCORECARD", rule_width)
    for source_name in ["Polymarket", "538"]:
        source_rows = summary_df[summary_df["source"] == source_name]
        eligible_rows = source_rows[source_rows["has_data"]]
        correct_count = (eligible_rows["direction_match"] == "correct").sum()
        eligible_count = len(eligible_rows)
        available_count = int(source_rows["has_data"].sum())
        total_gap_days = int(source_rows["data_gap_days"].sum())

        print(
            f"  {source_name:12s}: {correct_count}/{eligible_count} "
            f"correct direction  | {available_count}/{len(source_rows)} "
            f"events with data  | {total_gap_days} total gap days"
        )


def main():
    """
    Loads the processed data, computes event responses, and prints the
    analysis report.
    """
    print_box_header("EVENT RESPONSE ANALYSIS: POLYMARKET vs. 538", RULE_WIDTH)

    polymarket_df, polls_df = load_daily_data(PROCESSED_DIR)
    summary_df = build_reaction_summary(
        EVENTS,
        polymarket_df,
        polls_df,
        SWING_STATES,
        OVERLAP_CUTOFF,
        MIN_POST_COVERAGE,
    )

    for event in EVENTS:
        print_event_summary(event, summary_df, RULE_WIDTH)

    print_scorecard(summary_df, RULE_WIDTH)
    print_box_header("Analysis complete.", RULE_WIDTH)


if __name__ == "__main__":
    main()
