"""
Clean Polymarket prediction-market data.

Reads ~50 per-state CSVs at daily, hourly, and minutely granularity from
data/raw/polymarket/archive/ and outputs three consolidated CSVs to
data/processed/.

Outputs:
    polymarket_daily.csv
    polymarket_hourly.csv
    polymarket_minutely.csv
"""

import pandas as pd
from pathlib import Path

from src.clean.utils import (
    POLYMARKET_RAW, STATE_ABBREV_TO_NAME, SWING_STATES, save_csv,
)

# Map subdirectory names to output filenames and frequency labels
GRANULARITIES = {
    "csv_day":    ("polymarket_daily.csv",    "daily"),
    "csv_hour":   ("polymarket_hourly.csv",   "hourly"),
    "csv_minute": ("polymarket_minutely.csv", "minutely"),
}


def _extract_state(filename):
    """Extract state abbreviation from filename like 'PA_daily.csv'."""
    return filename.split("_")[0].upper()


def _clean_granularity(subdir, label):
    """Load and clean all state CSVs from one granularity subdirectory."""
    folder = POLYMARKET_RAW / subdir
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        print(f"  WARNING: No CSV files found in {folder}")
        return pd.DataFrame()

    print(f"  Loading {len(csv_files)} {label} files...")

    frames = []
    for csv_path in csv_files:
        state = _extract_state(csv_path.name)
        df = pd.read_csv(csv_path)
        df["state"] = state
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Rename columns
    combined = combined.rename(columns={
        "Date (UTC)": "date_utc",
        "Timestamp (UTC)": "timestamp_utc",
        "Donald Trump": "trump_prob",
        "Kamala Harris": "harris_prob",
        "Other": "other_prob",
    })

    # Parse date_utc and extract date-only column
    combined["date_utc"] = pd.to_datetime(combined["date_utc"],
                                          format="%m-%d-%Y %H:%M")
    combined["date"] = combined["date_utc"].dt.date

    # Cast types
    combined["trump_prob"] = combined["trump_prob"].astype(float)
    combined["harris_prob"] = combined["harris_prob"].astype(float)
    combined["other_prob"] = combined["other_prob"].astype(float)
    combined["timestamp_utc"] = combined["timestamp_utc"].astype(int)

    # Derived columns
    combined["trump_lead"] = combined["trump_prob"] - combined["harris_prob"]
    combined["state_name"] = combined["state"].map(STATE_ABBREV_TO_NAME)
    combined["is_swing"] = combined["state"].isin(SWING_STATES)

    # Sanity check: probabilities should sum to ~1.0
    prob_sum = (combined["trump_prob"]
                + combined["harris_prob"]
                + combined["other_prob"])
    bad_rows = (prob_sum - 1.0).abs() > 0.05
    if bad_rows.any():
        print(f"  WARNING: {bad_rows.sum()} rows have probabilities that "
              f"deviate from 1.0 by more than 0.05")

    # Sort and deduplicate
    combined = combined.sort_values(["state", "date_utc"])
    combined = combined.drop_duplicates(subset=["state", "date_utc"])

    # For daily granularity, multiple timestamps can map to the same date.
    # Keep the latest snapshot per (state, date).
    if label == "daily":
        before = len(combined)
        combined = combined.sort_values(["state", "date", "timestamp_utc"])
        combined = combined.drop_duplicates(subset=["state", "date"], keep="last")
        dropped = before - len(combined)
        if dropped:
            print(f"  Dropped {dropped} same-date duplicate rows (kept latest)")

    # Final column order
    combined = combined[[
        "date", "date_utc", "timestamp_utc", "state", "state_name",
        "is_swing", "trump_prob", "harris_prob", "other_prob", "trump_lead",
    ]]

    return combined


def main():
    """Clean all three Polymarket granularities."""
    print("Cleaning Polymarket data...")

    for subdir, (output_file, label) in GRANULARITIES.items():
        df = _clean_granularity(subdir, label)
        if not df.empty:
            save_csv(df, output_file, f"Polymarket {label}")

    print()


if __name__ == "__main__":
    main()
