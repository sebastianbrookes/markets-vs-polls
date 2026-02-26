"""
Run all data cleaning scripts in sequence.

Usage:
    python -m src.clean.run_all
"""

from src.clean.clean_polymarket import main as clean_polymarket
from src.clean.clean_fivethirtyeight import main as clean_fivethirtyeight
from src.clean.clean_fec import main as clean_fec
from src.clean.clean_events import main as clean_events


def main():
    print("=" * 60)
    print("Running all data cleaning scripts")
    print("=" * 60)
    print()

    clean_polymarket()
    clean_fivethirtyeight()
    clean_fec()
    clean_events()

    print("=" * 60)
    print("All cleaning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
