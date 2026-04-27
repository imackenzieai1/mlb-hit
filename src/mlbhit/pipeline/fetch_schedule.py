"""Pull MLB schedule + probable pitchers.

Single-day mode is used by daily_runner (today's games).
Range/season mode is used for historical backfill so training data can join
on actual starting pitchers. For completed games, `statsapi.schedule(d)`
returns the actual starter in the probable_pitcher_id fields.
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import pandas as pd
import statsapi
from tqdm import tqdm

from ..io import clean_path, raw_path
from ..utils.dates import daterange


def _row_from_game(d: date, g: dict) -> dict:
    return {
        "date": d.isoformat(),
        "game_pk": g["game_id"],
        "status": g["status"],
        "home_team": g["home_name"],
        "away_team": g["away_name"],
        "home_abbr": g.get("home_id"),
        "away_abbr": g.get("away_id"),
        "venue": g.get("venue_name"),
        "game_datetime": g.get("game_datetime"),
        "home_probable_pitcher": g.get("home_probable_pitcher"),
        "away_probable_pitcher": g.get("away_probable_pitcher"),
        "home_probable_pitcher_id": g.get("home_probable_pitcher_id"),
        "away_probable_pitcher_id": g.get("away_probable_pitcher_id"),
    }


# Process-wide cache for name -> mlbam_id so a single backtest doesn't repeat
# expensive `statsapi.lookup_player` calls. Reset between processes is fine.
_PITCHER_NAME_CACHE: dict[str, int | None] = {}


def _load_players_name_index() -> tuple[pd.DataFrame | None, str | None]:
    """Load players.parquet and pick the most likely name column.

    Returns (players_df, name_col) or (None, None) if not available. Cached
    once per process via the closure on `clean_path`.
    """
    try:
        players = pd.read_parquet(clean_path("players.parquet"))
    except Exception:
        return None, None
    candidates = [
        "name", "player_name", "pitcher_name", "full_name",
        "name_fangraphs", "Name", "name_first_last",
    ]
    for c in candidates:
        if c in players.columns:
            return players, c
    return players, None


def _resolve_pitcher_id(name: str, players: pd.DataFrame | None, name_col: str | None) -> int | None:
    """Best-effort name -> mlbam_id resolution.

    Strategy: local players.parquet (no network) first, then
    `statsapi.lookup_player` as a fallback for rookies/call-ups not yet in the
    parquet. Negative results are cached too so we don't retry every call.
    """
    if not name or pd.isna(name):
        return None
    if name in _PITCHER_NAME_CACHE:
        return _PITCHER_NAME_CACHE[name]

    pid: int | None = None
    if players is not None and name_col is not None and "mlbam_id" in players.columns:
        m = players[players[name_col].astype(str).str.lower() == str(name).lower()]
        if not m.empty:
            v = m.iloc[0]["mlbam_id"]
            if pd.notna(v):
                try:
                    pid = int(v)
                except (TypeError, ValueError):
                    pid = None

    if pid is None:
        try:
            results = statsapi.lookup_player(name)
            if results:
                # Prefer pitchers if statsapi returns multiple matches.
                pitchers = [r for r in results
                            if str(r.get("primaryPosition", {}).get("abbreviation", "")).upper() == "P"]
                pick = (pitchers or results)[0]
                pid = int(pick["id"])
        except Exception:
            pid = None

    _PITCHER_NAME_CACHE[name] = pid
    return pid


def _backfill_pitcher_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill *_probable_pitcher_id from *_probable_pitcher when statsapi
    omits the IDs (newer python-mlb-statsapi versions stopped including them
    for in-progress games — they only return the names)."""
    pairs = [
        ("home_probable_pitcher", "home_probable_pitcher_id"),
        ("away_probable_pitcher", "away_probable_pitcher_id"),
    ]
    needs = set()
    for nc, ic in pairs:
        if nc in df.columns and ic in df.columns:
            mask = df[ic].isna() & df[nc].notna()
            needs.update(df.loc[mask, nc].astype(str).tolist())
    if not needs:
        return df

    players, name_col = _load_players_name_index()
    resolved: dict[str, int] = {}
    for name in needs:
        pid = _resolve_pitcher_id(name, players, name_col)
        if pid is not None:
            resolved[name] = pid

    for nc, ic in pairs:
        if nc in df.columns and ic in df.columns:
            mask = df[ic].isna() & df[nc].notna()
            df.loc[mask, ic] = df.loc[mask, nc].astype(str).map(resolved)

    print(f"  backfilled probable pitcher IDs: {len(resolved)}/{len(needs)} resolved by name")
    return df


def fetch_schedule(d: date) -> pd.DataFrame:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    df = pd.DataFrame([_row_from_game(d, g) for g in games])
    df = _backfill_pitcher_ids(df)
    out = raw_path("schedule", f"{d.isoformat()}.parquet")
    df.to_parquet(out, index=False)
    return df


def fetch_schedule_range(
    start: date,
    end: date,
    overwrite: bool = False,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Pull schedules day-by-day across [start, end]. One parquet per day.

    Skips days that already have a cached parquet unless overwrite=True.
    Sleeps briefly between calls to be polite to the API.
    """
    frames = []
    for d in tqdm(list(daterange(start, end))):
        out = raw_path("schedule", f"{d.isoformat()}.parquet")
        if out.exists() and not overwrite:
            frames.append(pd.read_parquet(out))
            continue
        try:
            games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
        except Exception as e:
            print(f"  skip {d}: {e}")
            continue
        df = pd.DataFrame([_row_from_game(d, g) for g in games])
        df = _backfill_pitcher_ids(df)
        df.to_parquet(out, index=False)
        frames.append(df)
        time.sleep(sleep_s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_season_schedules(season: int, overwrite: bool = False) -> pd.DataFrame:
    """Backfill a full season's schedules. ~2600 API calls per season."""
    return fetch_schedule_range(
        date(season, 3, 20), date(season, 11, 5), overwrite=overwrite
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, help="Backfill an entire season.")
    parser.add_argument("--date", type=str, help="Pull a specific date (YYYY-MM-DD).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch even if a cached parquet exists.")
    args = parser.parse_args()

    if args.season:
        df = fetch_season_schedules(args.season, overwrite=args.overwrite)
        print(f"Season {args.season}: {len(df)} schedule rows")
        if "home_probable_pitcher_id" in df.columns:
            hit_rate = df["home_probable_pitcher_id"].notna().mean()
            print(f"  rows with home_probable_pitcher_id: {hit_rate:.1%}")
    elif args.date:
        d = date.fromisoformat(args.date)
        df = fetch_schedule(d)
        print(df.head())
        print(f"{len(df)} games")
    else:
        df = fetch_schedule(date.today())
        print(df.head())
        print(f"{len(df)} games")
