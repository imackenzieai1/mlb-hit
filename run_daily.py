#!/usr/bin/env python
"""Daily pipeline: schedule -> lineups -> model -> odds -> EV-ranked picks.

Odds source is controlled by --odds-source. Default is the-odds-api if
ODDS_API_KEY is set, otherwise we look for a manual CSV, otherwise we
produce model-only picks with no edge column.
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from mlbhit.config import env
from mlbhit.io import output_path, raw_path
from mlbhit.pipeline.fetch_lineups import fetch_lineups
from mlbhit.pipeline.fetch_prop_odds import load_props
from mlbhit.pipeline.fetch_schedule import fetch_schedule
from mlbhit.pipeline.recommend import recommend
from mlbhit.pipeline.score_today import score_for_date

# Prior-season stats blend cutover: before this date (inclusive) we blend the
# current season with the prior season to damp small-sample xBA swings. After
# this day the current season has enough PA per player to stand on its own.
# Roughly July 1 = ~40% of the season = ~250 PA per everyday hitter, which is
# enough signal to drop the prior-season regularizer.
_BLEND_CUTOVER_MONTH = 7
_BLEND_CUTOVER_DAY = 1


def _should_blend(target_date: date) -> bool:
    return (target_date.month, target_date.day) < (_BLEND_CUTOVER_MONTH, _BLEND_CUTOVER_DAY)


def _resolve_odds_source(explicit: str | None) -> str | None:
    """Decide where to get odds from today.

    Priority: --odds-source flag > ODDS_API_KEY env > manual CSV on disk > none.
    """
    if explicit:
        return explicit
    if env("ODDS_API_KEY"):
        return "theodds"
    return None


def _load_odds_or_none(target: date, source: str | None) -> pd.DataFrame | None:
    if source is None:
        print("No odds source configured — producing model-only picks.")
        return None
    if source == "csv":
        csv_path = raw_path("props", f"{target.isoformat()}_props.csv")
        if not csv_path.exists():
            print(f"WARN: --odds-source csv but {csv_path.name} missing. Skipping EV.")
            return None
    try:
        return load_props(target, source=source)
    except Exception as e:
        print(f"WARN: odds fetch failed ({e}). Falling back to model-only picks.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--odds-source",
        choices=["csv", "theodds"],
        default=None,
        help="Where to pull today's prop prices from. Defaults to theodds if "
             "ODDS_API_KEY is set, otherwise model-only.",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    fetch_schedule(target)
    fetch_lineups(target)

    prior_season = (target.year - 1) if _should_blend(target) else None
    if prior_season:
        print(f"Early-season blend active: current season ({target.year}) "
              f"weighted against prior season ({prior_season}).")
    preds = score_for_date(target, season=target.year, prior_season=prior_season)
    if preds.empty:
        print("No predictions produced — maybe lineups not out yet.")
        return

    source = _resolve_odds_source(args.odds_source)
    prop_prices = _load_odds_or_none(target, source)

    recs = recommend(preds, prop_prices=prop_prices)
    out_csv = output_path("recommendations", f"{target.isoformat()}.csv")
    recs.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(recs)} rows")
    print(recs.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
