"""
Clean FEC official 2024 presidential election results.

Reads the Excel file with the official results sheet and outputs a clean
CSV with vote shares, margins, and winner for each state.

Output:
    fec_results.csv
"""

import pandas as pd

from src.clean.utils import (
    FEC_RAW, STATE_ABBREV_TO_NAME, SWING_STATES, save_csv,
)

INPUT_FILE = FEC_RAW / "fec-results.xlsx"
SHEET_NAME = "OFFICIAL 2024 PRES GE RESULTS"

# Two-letter state/territory abbreviations we want to keep (50 states + DC)
VALID_STATES = set(STATE_ABBREV_TO_NAME.keys())


def main():
    """Clean FEC election results."""
    print("Cleaning FEC results...")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    print(f"  Loaded {len(df)} raw rows")

    # Keep only rows where STATE is a valid 2-letter abbreviation
    df = df[df["STATE"].isin(VALID_STATES)].copy()
    print(f"  Filtered to valid states: {len(df)} rows")

    # Rename columns to snake_case
    df = df.rename(columns={
        "STATE": "state",
        "ELECTORAL VOTES": "electoral_votes",
        "ELECTORAL VOTE: TRUMP (R)": "electoral_votes_trump",
        "ELECTORAL VOTE: HARRIS (D)": "electoral_votes_harris",
        "TRUMP": "trump_votes",
        "HARRIS": "harris_votes",
        "KENNEDY": "kennedy_votes",
        "TOTAL VOTES": "total_votes",
    })

    # Select and cast vote columns to int (fill NaN with 0 for minor candidates)
    vote_cols = ["trump_votes", "harris_votes", "kennedy_votes", "total_votes"]
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    ev_cols = ["electoral_votes", "electoral_votes_trump", "electoral_votes_harris"]
    for col in ev_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derive vote shares
    df["trump_vote_share"] = df["trump_votes"] / df["total_votes"]
    df["harris_vote_share"] = df["harris_votes"] / df["total_votes"]
    df["other_vote_share"] = (
        1.0 - df["trump_vote_share"] - df["harris_vote_share"]
    )

    # Margin (positive = Trump lead)
    df["margin"] = df["trump_vote_share"] - df["harris_vote_share"]

    # Winner
    df["winner"] = df.apply(
        lambda r: "Trump" if r["trump_votes"] > r["harris_votes"] else "Harris",
        axis=1,
    )

    # State name and swing flag
    df["state_name"] = df["state"].map(STATE_ABBREV_TO_NAME)
    df["is_swing"] = df["state"].isin(SWING_STATES)

    # Final column order
    df = df[[
        "state", "state_name", "is_swing", "electoral_votes",
        "trump_votes", "harris_votes", "kennedy_votes", "total_votes",
        "trump_vote_share", "harris_vote_share", "other_vote_share",
        "margin", "winner",
        "electoral_votes_trump", "electoral_votes_harris",
    ]]

    # Sort by state
    df = df.sort_values("state").reset_index(drop=True)

    save_csv(df, "fec_results.csv", "FEC results")
    print(f"  States with Trump win: "
          f"{(df['winner'] == 'Trump').sum()}")
    print(f"  States with Harris win: "
          f"{(df['winner'] == 'Harris').sum()}")
    print()


if __name__ == "__main__":
    main()
