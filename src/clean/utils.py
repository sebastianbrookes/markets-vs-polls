"""
Shared constants and helper functions for data cleaning scripts.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

POLYMARKET_RAW = RAW_DIR / "polymarket" / "archive"
FIVETHIRTYEIGHT_RAW = RAW_DIR / "fivethirtyeight"
FEC_RAW = RAW_DIR / "fec"
EVENTS_RAW = RAW_DIR / "events"

# ---------------------------------------------------------------------------
# State mappings
# ---------------------------------------------------------------------------
STATE_ABBREV_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida",
    "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

STATE_NAME_TO_ABBREV = {v: k for k, v in STATE_ABBREV_TO_NAME.items()}

SWING_STATES = ["AZ", "GA", "MI", "NV", "NC", "PA", "WI"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_output_dir():
    """Create data/processed/ directory if it doesn't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df, filename, description=""):
    """Save a DataFrame to data/processed/ and print summary info."""
    ensure_output_dir()
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved {description}: {filename} ({len(df)} rows, {len(df.columns)} cols)")
