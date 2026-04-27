"""Backfill historical hit-prop odds from the-odds-api for a date range.

Pricing (as of writing): historical endpoints cost 10x the regular rate.
 - /historical/.../events?date=T            -> 1 credit per call
 - /historical/.../events/{id}/odds?date=T  -> 10 credits per call per market per region

Per MLB game day (~13 games, 1 market, 1 region):
    1 + 13 * 10 = ~131 credits

Full 2026 YTD backfill (March 20 -> today, ~35 game days):
    ~35 * 131 = ~4,585 credits — within budget for a 20K credit tier.

Usage:
    # Test first (ONE day only, ~131 credits). Always run this before full backfill.
    python -m mlbhit.pipeline.fetch_historical_odds --date 2026-04-23

    # Full 2026 YTD backfill. Resumable: skips dates already cached.
    python -m mlbhit.pipeline.fetch_historical_odds --start 2026-03-20 --end 2026-04-23

Output: data/raw/historical_props/YYYY-MM-DD_props.parquet (one file per date)
Schema matches the live fetch_prop_odds output so downstream reconciliation
treats both sources identically.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import env
from ..io import clean_path, raw_path
from .fetch_prop_odds import DEFAULT_BOOKS, _load_player_map, _match_player_ids

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_MLB = "baseball_mlb"

# Morning snapshot — closest to when Ian would actually run run_daily.py and
# place paper bets. 14:00 UTC = 10am ET = 7am PT.
SNAPSHOT_HOUR_UTC = 14


def _historical_dir() -> Path:
    p = raw_path("historical_props", "")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.parent


def _date_path(d: date) -> Path:
    return _historical_dir() / f"{d.isoformat()}_props.parquet"


def _snapshot_timestamp(d: date) -> str:
    """ISO8601 Z-suffixed timestamp for the morning snapshot on date d."""
    return f"{d.isoformat()}T{SNAPSHOT_HOUR_UTC:02d}:00:00Z"


def _fetch_events_at(api_key: str, timestamp: str) -> list[dict]:
    """List games active at the given historical timestamp. Costs 1 credit."""
    r = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT_MLB}/events",
        params={"apiKey": api_key, "date": timestamp},
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    # Historical endpoints return {"timestamp": ..., "previous_timestamp": ..., "next_timestamp": ..., "data": [...]}
    return payload.get("data", []) if isinstance(payload, dict) else payload


def _fetch_event_odds_at(api_key: str, event_id: str, timestamp: str) -> dict:
    """Fetch odds for a single event at the given historical timestamp. Costs 10 credits."""
    r = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{ODDS_SPORT_MLB}/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "date": timestamp,
            "bookmakers": ",".join(sorted(DEFAULT_BOOKS)),
            "markets": "batter_hits",
            "oddsFormat": "american",
        },
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    # Historical single-event endpoint wraps the event under "data".
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def _parse_outcomes(event: dict, fetched_at: str, target_date: str) -> list[dict]:
    """Flatten bookmakers -> markets -> outcomes into one row per (player, book, side)."""
    rows = []
    for book in event.get("bookmakers", []):
        if book.get("key") not in DEFAULT_BOOKS:
            continue
        for market in book.get("markets", []):
            if market.get("key") != "batter_hits":
                continue
            for out in market.get("outcomes", []):
                rows.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": book["key"],
                    "side": out.get("name"),
                    "player_name": out.get("description") or out.get("name"),
                    "price": out.get("price"),
                    "point": out.get("point", 0.5),
                    "fetched_at": fetched_at,
                    "date": target_date,
                })
    return rows


def fetch_historical_day(
    target: date,
    api_key: str,
    sleep_between_games: float = 0.1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Pull one day of historical hit props. Writes parquet and returns DataFrame.

    Credit cost: ~131 for a typical full-slate day.
    """
    out_path = _date_path(target)
    if out_path.exists() and not overwrite:
        print(f"  [{target}] already cached at {out_path.name} (use --overwrite to re-pull)")
        return pd.read_parquet(out_path)

    timestamp = _snapshot_timestamp(target)
    print(f"  [{target}] snapshot @ {timestamp}")
    events = _fetch_events_at(api_key, timestamp)
    print(f"    events returned: {len(events)}  (cost so far: 1 credit)")
    if not events:
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, ev in enumerate(events, 1):
        try:
            event_data = _fetch_event_odds_at(api_key, ev["id"], timestamp)
            rows.extend(_parse_outcomes(event_data, fetched_at, target.isoformat()))
        except requests.HTTPError as e:
            # 422 = no data at this timestamp (game already final, or too old)
            print(f"    skip event {ev.get('id')}: {e}")
            continue
        time.sleep(sleep_between_games)
    print(f"    odds calls: {len(events)}  (cost this day: ~{1 + len(events) * 10} credits)")

    if not rows:
        print(f"  [{target}] no rows parsed — market may not have been offered at snapshot time")
        return pd.DataFrame()

    raw_df = pd.DataFrame(rows)
    raw_df = raw_df[raw_df["point"] == 0.5].copy()

    pivot = (
        raw_df.pivot_table(
            index=["player_name", "book", "event_id", "commence_time",
                   "home_team", "away_team", "fetched_at", "date"],
            columns="side",
            values="price",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"Over": "over_price", "Under": "under_price"})
    )

    pivot = _match_player_ids(pivot, _load_player_map())
    pivot["player_id"] = pivot["mlbam_id"].astype("Int64")

    out_cols = [
        "date", "player_id", "player_name", "book",
        "over_price", "under_price", "fetched_at",
    ]
    out = pivot[out_cols].dropna(subset=["player_id"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)
    print(f"  [{target}] wrote {len(out)} rows -> {out_path.name}")
    return out


def backfill_range(
    start: date,
    end: date,
    confirm_cost: bool = True,
    sleep_between_days: float = 0.5,
) -> pd.DataFrame:
    """Pull historical odds for every date in [start, end]. Resumable."""
    api_key = env("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing in .env")

    # Dates already cached — skip to save credits.
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d)
        d += timedelta(days=1)

    todo = [d for d in all_dates if not _date_path(d).exists()]
    skipped = len(all_dates) - len(todo)

    est_per_day = 131  # worst-case; light days cost less
    est_total = len(todo) * est_per_day

    print(f"Backfill plan: {len(all_dates)} days in range, {skipped} already cached, {len(todo)} to pull.")
    print(f"Estimated credit cost: ~{est_total:,} credits (upper bound, ~{est_per_day}/day x {len(todo)} days)")

    if confirm_cost and len(todo) > 1:
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return pd.DataFrame()

    frames = []
    for i, d in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {d}")
        try:
            day_df = fetch_historical_day(d, api_key)
        except requests.HTTPError as e:
            print(f"  FATAL for {d}: {e}")
            if e.response is not None and e.response.status_code == 401:
                print("  401 unauthorized — API key issue. Stopping.")
                break
            if e.response is not None and e.response.status_code == 429:
                print("  429 rate-limited — pausing 30s and retrying once.")
                time.sleep(30)
                try:
                    day_df = fetch_historical_day(d, api_key)
                except Exception as e2:
                    print(f"  retry failed: {e2}. Moving on.")
                    continue
            else:
                continue
        if not day_df.empty:
            frames.append(day_df)
        time.sleep(sleep_between_days)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Single date YYYY-MM-DD (test mode).")
    parser.add_argument("--start", type=str, help="Range start YYYY-MM-DD.")
    parser.add_argument("--end", type=str, help="Range end YYYY-MM-DD (default: yesterday).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-pull dates that are already cached.")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip interactive cost confirmation (for cron use).")
    args = parser.parse_args()

    api_key = env("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing in .env")

    if args.date:
        target = date.fromisoformat(args.date)
        df = fetch_historical_day(target, api_key, overwrite=args.overwrite)
        if not df.empty:
            print(df.head(10).to_string(index=False))
    elif args.start:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else (date.today() - timedelta(days=1))
        backfill_range(start, end, confirm_cost=not args.no_confirm)
    else:
        parser.print_help()
