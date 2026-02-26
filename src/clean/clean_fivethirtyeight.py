"""
Clean FiveThirtyEight polling average data.

Reads the raw 27k-row CSV covering 2020 and 2024 cycles, filters to 2024,
and pivots to wide format (one row per date x state).

Output:
    polls_538.csv
"""

import pandas as pd

from src.clean.utils import (
    FIVETHIRTYEIGHT_RAW, STATE_NAME_TO_ABBREV, STATE_ABBREV_TO_NAME,
    SWING_STATES, save_csv,
)

INPUT_FILE = (FIVETHIRTYEIGHT_RAW
              / "presidential_general_averages_2024-09-12_uncorrected.csv")


def _load_and_filter(path):
    """Load the full CSV and filter to 2024 cycle only."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} total rows")

    df = df[df["cycle"] == 2024].copy()
    print(f"  Filtered to 2024: {len(df)} rows")

    # Drop columns that are all-null or constant for 2024
    df = df.drop(columns=["pct_trend_adjusted", "cycle"])

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    return df


def _map_states(df):
    """Map full state names to abbreviations. National -> US."""
    mapping = STATE_NAME_TO_ABBREV.copy()
    mapping["National"] = "US"
    df["state"] = df["state"].map(mapping)
    return df


def _pivot_to_wide(df):
    """Pivot long-format data to one row per (date, state) with prefixed columns."""

    # Separate candidates into groups
    trump = df[df["candidate"] == "Trump"].copy()
    biden = df[df["candidate"] == "Biden"].copy()
    harris = df[df["candidate"] == "Harris"].copy()
    kennedy = df[df["candidate"] == "Kennedy"].copy()

    # Combine Biden + Harris as the Democratic candidate
    dem = pd.concat([biden, harris], ignore_index=True)
    dem["dem_candidate"] = dem["candidate"].map({
        "Biden": "Biden",
        "Harris": "Harris",
    })

    # Rename estimate columns with prefixes
    trump_wide = trump.rename(columns={
        "pct_estimate": "trump_pct",
        "hi": "trump_hi",
        "lo": "trump_lo",
    })[["date", "state", "trump_pct", "trump_hi", "trump_lo"]]

    dem_wide = dem.rename(columns={
        "pct_estimate": "dem_pct",
        "hi": "dem_hi",
        "lo": "dem_lo",
    })[["date", "state", "dem_pct", "dem_hi", "dem_lo", "dem_candidate"]]

    kennedy_wide = kennedy.rename(columns={
        "pct_estimate": "kennedy_pct",
        "hi": "kennedy_hi",
        "lo": "kennedy_lo",
    })[["date", "state", "kennedy_pct", "kennedy_hi", "kennedy_lo"]]

    # Merge on (date, state)
    merged = trump_wide.merge(dem_wide, on=["date", "state"], how="outer")
    merged = merged.merge(kennedy_wide, on=["date", "state"], how="left")

    return merged


def main():
    """Clean FiveThirtyEight polling data."""
    print("Cleaning FiveThirtyEight data...")

    df = _load_and_filter(INPUT_FILE)
    df = _map_states(df)

    wide = _pivot_to_wide(df)

    # Derived columns
    wide["trump_lead"] = wide["trump_pct"] - wide["dem_pct"]
    wide["state_name"] = wide["state"].map(
        {**STATE_ABBREV_TO_NAME, "US": "National"}
    )
    wide["is_swing"] = wide["state"].isin(SWING_STATES)

    # Sort by state and date
    wide = wide.sort_values(["state", "date"])

    # Final column order
    wide = wide[[
        "date", "state", "state_name", "is_swing",
        "trump_pct", "trump_hi", "trump_lo",
        "dem_pct", "dem_hi", "dem_lo", "dem_candidate",
        "kennedy_pct", "kennedy_hi", "kennedy_lo",
        "trump_lead",
    ]]

    save_csv(wide, "polls_538.csv", "FiveThirtyEight polls")
    print(f"  Unique states: {sorted(wide['state'].unique())}")
    print(f"  Date range: {wide['date'].min()} to {wide['date'].max()}")
    print()


if __name__ == "__main__":
    main()
