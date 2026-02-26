"""
Clean campaign events timeline data.

Reads the manually curated events CSV and adds derived columns for
days-before-election and affected candidate flags.

Output:
    events.csv
"""

import pandas as pd

from src.clean.utils import EVENTS_RAW, save_csv

INPUT_FILE = EVENTS_RAW / "events.csv"
ELECTION_DAY = pd.Timestamp("2024-11-05")


def main():
    """Clean events timeline data."""
    print("Cleaning events data...")

    df = pd.read_csv(INPUT_FILE)

    # Drop empty trailing rows
    df = df.dropna(subset=["date"]).copy()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Derived columns
    df["days_before_election"] = (ELECTION_DAY - df["date"]).dt.days
    df["affects_trump"] = df["affects"].isin(["Trump", "both"])
    df["affects_harris"] = df["affects"].isin(["Harris", "both"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Final column order
    df = df[[
        "date", "event", "category", "affects",
        "days_before_election", "affects_trump", "affects_harris",
    ]]

    save_csv(df, "events.csv", "Events timeline")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Events affecting Trump: {df['affects_trump'].sum()}")
    print(f"  Events affecting Harris: {df['affects_harris'].sum()}")
    print()


if __name__ == "__main__":
    main()
