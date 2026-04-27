"""Diagnose why opp_sp_id is NaN for today's slate.

Checks four layers, top-down:
  1. Raw statsapi.schedule() output — does the API itself return probable pitcher ids?
  2. fetch_schedule() output — does our row-builder preserve them?
  3. Cached parquet — is there a stale cache being read instead of fresh data?
  4. Lineup join — does game_pk align between schedule and lineups?

Usage:
    python scripts/debug_probable_pitchers.py --date 2026-04-24
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd
import statsapi

from mlbhit.io import raw_path
from mlbhit.pipeline.fetch_lineups import fetch_lineups
from mlbhit.pipeline.fetch_schedule import fetch_schedule


def check(d: date) -> None:
    print("=" * 70)
    print(f"DEBUG  probable pitchers  {d}")
    print("=" * 70)

    # 1. Raw API
    print("\n[1] Raw statsapi.schedule() — first 3 games:")
    raw = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    print(f"    total games: {len(raw)}")
    if not raw:
        print("    !! API returned 0 games. Date wrong, off-day, or API outage.")
        return
    for g in raw[:3]:
        keys_of_interest = {k: g.get(k) for k in [
            "game_id", "status",
            "home_name", "away_name",
            "home_probable_pitcher", "away_probable_pitcher",
            "home_probable_pitcher_id", "away_probable_pitcher_id",
        ]}
        print("   ", json.dumps(keys_of_interest, default=str, indent=4)[:600])

    # API-wide hit rate
    n = len(raw)
    n_home_id = sum(1 for g in raw if g.get("home_probable_pitcher_id"))
    n_away_id = sum(1 for g in raw if g.get("away_probable_pitcher_id"))
    print(f"    home_probable_pitcher_id populated: {n_home_id}/{n}")
    print(f"    away_probable_pitcher_id populated: {n_away_id}/{n}")
    if n_home_id == 0 and n_away_id == 0:
        print("    !! statsapi returned ZERO probable pitchers. Either MLB hasn't")
        print("       posted them yet (run again in a few hours) or the API's")
        print("       field name has shifted. Print one full game above to inspect.")
        print("       Full keys for game[0]:")
        print("       ", sorted(raw[0].keys()))

    # 2. fetch_schedule output (also overwrites the cache)
    print("\n[2] fetch_schedule(d) parquet contents:")
    sched = fetch_schedule(d)
    print(f"    rows: {len(sched)}")
    cols = ["home_probable_pitcher_id", "away_probable_pitcher_id"]
    for c in cols:
        if c in sched.columns:
            print(f"    {c}: {sched[c].notna().sum()}/{len(sched)} populated, "
                  f"dtype={sched[c].dtype}")

    # 3. Stale cache?
    cache = raw_path("schedule", f"{d.isoformat()}.parquet")
    print(f"\n[3] Cache path: {cache}")
    print(f"    exists: {cache.exists()}")
    if cache.exists():
        cached = pd.read_parquet(cache)
        for c in cols:
            if c in cached.columns:
                print(f"    cached {c}: {cached[c].notna().sum()}/{len(cached)}")

    # 4. Lineups join
    print("\n[4] fetch_lineups(d) and join check:")
    lineups = fetch_lineups(d)
    print(f"    lineup rows: {len(lineups)}")
    if lineups.empty:
        print("    !! No lineups yet. Probable pitcher merge would also be empty.")
        return
    sched_lite = sched[["game_pk", "home_probable_pitcher_id", "away_probable_pitcher_id"]]
    j = lineups.merge(sched_lite, on="game_pk", how="left")
    matched = j["home_probable_pitcher_id"].notna() | j["away_probable_pitcher_id"].notna()
    print(f"    lineup rows with at least one probable pitcher: {matched.sum()}/{len(j)}")
    if matched.sum() == 0:
        print("    !! Even after join, every row is missing both probables.")
        print("    Common causes: schedule game_pk dtype mismatch, schedule fetched")
        print("    before MLB posted probables, or different game_pk source.")
        print("    Sample row:")
        print("    ", j.iloc[0].to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default today)")
    args = parser.parse_args()
    d = date.fromisoformat(args.date) if args.date else date.today()
    check(d)
