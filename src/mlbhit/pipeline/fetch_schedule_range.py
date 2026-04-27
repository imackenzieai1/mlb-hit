"""Fetch schedule parquet for each day in a range (needed for historical joins in build_features)."""

from __future__ import annotations

import argparse
from datetime import date

from tqdm import tqdm

from ..utils.dates import daterange
from .fetch_schedule import fetch_schedule


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch MLB schedule to data/raw/schedule per day.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    for d in tqdm(list(daterange(start, end))):
        fetch_schedule(d)


if __name__ == "__main__":
    main()
