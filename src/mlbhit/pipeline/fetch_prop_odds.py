"""Ingest player hit-prop odds from a manual CSV (V1) or DraftKings JSON (V2 stub).

Why manual CSV first: it's zero-cost, zero-dependency, and gets you to "first real
bet evaluation" in minutes. Every morning you paste the hit props from your book
of choice into a CSV in data/raw/props/. This module normalizes them into a
parquet that `recommend.py` consumes.

The CSV you paste in should live at:

    data/raw/props/YYYY-MM-DD_props.csv

Required columns (case-insensitive, extras are preserved):

    player_name    - "Jose Altuve"  (we fuzzy-match to mlbam_id via players.parquet)
    over_price     - -135            (American odds for the "1+ hits" OVER)
    under_price    - +110            (American odds for the UNDER; optional)
    book           - "DraftKings"    (optional, defaults to "manual")

Output: data/raw/props/YYYY-MM-DD_props.parquet with one row per player-book
combination. `recommend.py` expects this schema:

    date, player_id, player_name, book, over_price, under_price, fetched_at
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import env
from ..io import clean_path, raw_path

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_MLB = "baseball_mlb"

# the-odds-api bookmaker keys. Keep this short — every extra book we include is
# more rows to scan. DK + FD are the two books Ian actually places at.
DEFAULT_BOOKS = {"draftkings", "fanduel"}

# On-disk cache of statsapi lookups so we don't re-hit MLB for the same rookie
# name every day. Shape: {normalized_name: mlbam_id}
_LOOKUP_CACHE_PATH = lambda: clean_path("player_name_lookup_cache.parquet")


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching across data sources.

    Handles:
      - accented characters (García -> garcia, Páges -> pages)
      - suffix noise (Jr., Sr., II, III)
      - period punctuation in initials (J.T. -> jt)
      - whitespace collapsing
    """
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    # Strip diacritics: decompose to base char + combining marks, drop the marks.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Strip common suffixes. Books format these inconsistently (Jr., jr, JR).
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)\s*$", "", s)
    # Remove periods (J.T. Realmuto -> jt realmuto).
    s = s.replace(".", "")
    # Collapse double spaces.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_lookup_cache() -> dict[str, int]:
    p = _LOOKUP_CACHE_PATH()
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["player_name_norm"], df["mlbam_id"]))


def _save_lookup_cache(cache: dict[str, int]) -> None:
    if not cache:
        return
    df = pd.DataFrame(
        [{"player_name_norm": k, "mlbam_id": int(v)} for k, v in cache.items()]
    )
    df.to_parquet(_LOOKUP_CACHE_PATH(), index=False)


def _statsapi_lookup(name: str) -> int | None:
    """Fallback to MLB Stats API for names missing from Chadwick.

    MLB's /people/search endpoint has everyone active including yesterday's
    call-ups. Zero cost (no the-odds-api credits consumed). Import is local
    so the module doesn't require statsapi for CSV-only use.
    """
    try:
        import statsapi
    except ImportError:
        return None
    try:
        matches = statsapi.lookup_player(name)
    except Exception:
        return None
    if not matches:
        return None
    # statsapi returns list-of-dicts sorted by most-recent activity.
    # Prefer someone who actually played in MLB (id is present and numeric).
    for m in matches:
        mid = m.get("id")
        if isinstance(mid, int):
            return mid
    return None


def _load_player_map() -> pd.DataFrame:
    """Return a normalized lookup: normalized player_name -> mlbam_id."""
    players = pd.read_parquet(clean_path("players.parquet"))
    keep = ["mlbam_id", "player_name", "name_first", "name_last"]
    players = players[[c for c in keep if c in players.columns]].copy()
    players["player_name_norm"] = players["player_name"].apply(_normalize_name)
    return players


def _match_player_ids(props: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach mlbam_id to each prop row, with statsapi fallback for rookies.

    Strategy:
      1. Normalize both sides (accents, suffixes, punctuation).
      2. Exact match on normalized name against the Chadwick-derived player map.
      3. For anything still unmatched, fall back to MLB /people/search.
      4. Cache successful statsapi hits to disk so future days skip the lookup.
    """
    props["player_name_norm"] = props["player_name"].apply(_normalize_name)
    merged = props.merge(
        players[["mlbam_id", "player_name_norm"]],
        on="player_name_norm",
        how="left",
    )

    missing_mask = merged["mlbam_id"].isna()
    if missing_mask.any():
        lookup_cache = _load_lookup_cache()
        unmatched_names = merged.loc[missing_mask, "player_name_norm"].unique()
        resolved = 0
        for nm in unmatched_names:
            if nm in lookup_cache:
                mid = lookup_cache[nm]
            else:
                # Reconstruct a reasonable lookup string — statsapi handles
                # accent-free inputs fine.
                mid = _statsapi_lookup(nm)
                if mid is not None:
                    lookup_cache[nm] = mid
            if mid is not None:
                merged.loc[merged["player_name_norm"] == nm, "mlbam_id"] = mid
                resolved += 1
        _save_lookup_cache(lookup_cache)
        if resolved:
            print(f"  statsapi fallback resolved {resolved}/{len(unmatched_names)} unmatched names")

    still_missing = merged[merged["mlbam_id"].isna()]["player_name"].unique()
    if len(still_missing) > 0:
        print(f"WARN: {len(still_missing)} player name(s) unmatched even after statsapi fallback:")
        for m in still_missing[:20]:
            print(f"   - {m}")

    merged = merged.drop(columns=["player_name_norm"])
    return merged


def load_manual_csv(target_date: date) -> pd.DataFrame:
    """Read the day's manual prop CSV, normalize, and resolve player IDs."""
    csv_path = raw_path("props", f"{target_date.isoformat()}_props.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No CSV at {csv_path}. Create it with columns: "
            "player_name, over_price [, under_price, book]"
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"player_name", "over_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "under_price" not in df.columns:
        df["under_price"] = pd.NA
    if "book" not in df.columns:
        df["book"] = "manual"

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["over_price"] = pd.to_numeric(df["over_price"], errors="coerce").astype("Int64")
    df["under_price"] = pd.to_numeric(df["under_price"], errors="coerce").astype("Int64")

    df = _match_player_ids(df, _load_player_map())
    df["date"] = target_date.isoformat()
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    df["player_id"] = df["mlbam_id"].astype("Int64")

    out_cols = [
        "date",
        "player_id",
        "player_name",
        "book",
        "over_price",
        "under_price",
        "fetched_at",
    ]
    out = df[out_cols].dropna(subset=["player_id"]).reset_index(drop=True)
    out_path = raw_path("props", f"{target_date.isoformat()}_props.parquet")
    out.to_parquet(out_path, index=False)
    print(f"{len(out)} props loaded from {csv_path.name} -> {out_path.name}")
    return out


def _list_mlb_events(api_key: str, target_date: date) -> list[dict]:
    """Return the-odds-api event objects for games commencing on target_date (UTC).

    Each event costs 0 credits to list; only the odds-per-event call consumes quota.
    """
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{ODDS_SPORT_MLB}/events",
        params={"apiKey": api_key},
        timeout=10,
    )
    r.raise_for_status()
    events = r.json()
    iso_prefix = target_date.isoformat()
    # commence_time is ISO8601 UTC; filter by date prefix. Games after ~7pm Central
    # will have a UTC date of the *next* day — include both to be safe, then filter
    # further downstream if needed.
    return [
        e for e in events
        if e.get("commence_time", "").startswith(iso_prefix)
    ]


def _parse_batter_hits_outcomes(event: dict, fetched_at: str) -> list[dict]:
    """Flatten the-odds-api nested bookmakers -> markets -> outcomes into rows."""
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "batter_hits":
                continue
            for out in market.get("outcomes", []):
                # Schema (as of the-odds-api v4): name = "Over" | "Under",
                # description = player name, price = American odds, point = line.
                rows.append({
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": book["key"],
                    "side": out.get("name"),
                    "player_name": out.get("description") or out.get("name"),
                    "price": out.get("price"),
                    "point": out.get("point"),  # no default — caught by strict filter
                    "fetched_at": fetched_at,
                })
    return rows


def fetch_theodds_hit_props(target_date: date) -> pd.DataFrame:
    """Fetch MLB batter_hits props from the-odds-api.com for target_date.

    Cost estimate (as of writing): 1 credit for the events list + 1 credit per
    event per market per region. Typical day: ~16 credits. Budget-friendly even
    on the 20K-credit $30/mo tier.

    Requires env var ODDS_API_KEY (set in .env in repo root).
    """
    api_key = env("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY not found. Add to .env:\n    ODDS_API_KEY=your_key_here"
        )

    events = _list_mlb_events(api_key, target_date)
    print(f"Found {len(events)} MLB events for {target_date.isoformat()}")
    if not events:
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in events:
        url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_MLB}/events/{ev['id']}/odds"
        params = {
            "apiKey": api_key,
            # Specifying bookmakers= narrows the response payload to just DK+FD.
            # Credit cost is still markets x regions, but the response is ~10x smaller.
            "bookmakers": ",".join(sorted(DEFAULT_BOOKS)),
            "markets": "batter_hits",
            "oddsFormat": "american",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"  skip event {ev['id']} ({ev.get('home_team')} vs {ev.get('away_team')}): {e}")
            continue
        rows.extend(_parse_batter_hits_outcomes(r.json(), fetched_at))

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        print("No batter_hits markets returned. Check your API tier includes player props.")
        return raw_df

    # Filter to just the books Ian actually plays at (DK + FD).
    before = len(raw_df)
    raw_df = raw_df[raw_df["book"].isin(DEFAULT_BOOKS)].copy()
    print(f"  book filter ({sorted(DEFAULT_BOOKS)}): {len(raw_df):,}/{before:,} rows kept")

    # Strict point filter: drop any row whose `point` isn't exactly 0.5.
    # This includes rows with NaN/None point (which the previous default-to-0.5
    # silently passed through). The-odds-api sometimes returns alternate-line
    # rows (1.5, 2.5 hits) without an explicit point field — those would
    # otherwise show up as "1+ hits at +220" which is structurally wrong and
    # produces phantom edges. Be ruthless here.
    before = len(raw_df)
    raw_df = raw_df[raw_df["point"].notna() & (raw_df["point"] == 0.5)].copy()
    if len(raw_df) < before:
        print(f"  line filter (point==0.5): {len(raw_df):,}/{before:,} rows kept "
              f"(dropped {before - len(raw_df)} non-0.5 / null-point rows)")

    # Sanity check: 1+ hit prices for MLB regulars are typically -100 to -300.
    # If we see a meaningful chunk of rows above +200 (or a price spread that
    # implies the market wasn't actually 1+ hits), warn loudly. This catches
    # API regressions and alternate-line leakage that the strict filter missed.
    if not raw_df.empty:
        price_num = pd.to_numeric(raw_df["price"], errors="coerce")
        n_extreme_plus = int((price_num > 200).sum())
        share_extreme = n_extreme_plus / len(raw_df)
        if share_extreme > 0.05:
            print(f"  WARN: {n_extreme_plus}/{len(raw_df)} rows ({share_extreme:.0%}) "
                  f"show price > +200. Unusual for 1+ hits markets — possible "
                  f"alternate-lines leakage. Spot-check the output before betting.")
        # Also flag if median price is unreasonably high
        median_price = float(price_num.median())
        if median_price > -80:
            print(f"  WARN: median price is {median_price:+.0f}, expected near -150. "
                  f"Possible market-key mismatch from theodds-api.")

    # Pivot Over / Under into two price columns on a single row per (player, book).
    pivot = (
        raw_df.pivot_table(
            index=["player_name", "book", "event_id", "commence_time",
                   "home_team", "away_team", "fetched_at"],
            columns="side",
            values="price",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"Over": "over_price", "Under": "under_price"})
    )

    # Match to mlbam_id via players.parquet
    pivot["date"] = target_date.isoformat()
    pivot = _match_player_ids(pivot, _load_player_map())
    pivot["player_id"] = pivot["mlbam_id"].astype("Int64")

    out_cols = [
        "date", "player_id", "player_name", "book",
        "over_price", "under_price", "fetched_at",
    ]
    out = pivot[out_cols].dropna(subset=["player_id"]).reset_index(drop=True)
    out_path = raw_path("props", f"{target_date.isoformat()}_props.parquet")
    out.to_parquet(out_path, index=False)
    print(f"{len(out)} props written -> {out_path.name}")
    return out


def fetch_draftkings_hit_props(target_date: date) -> pd.DataFrame:
    """DraftKings JSON scraping path. STUB: not implemented.

    Use --source theodds for programmatic ingestion. Only revisit this if you
    want a second data source for line-shopping.
    """
    raise NotImplementedError("Use --source theodds instead.")


# ---------------------------------------------------------------------------
# SharpAPI source
# ---------------------------------------------------------------------------
# SharpAPI exposes a unified MLB odds feed at /api/v1/odds with player props
# included as `market_type=player_hits`. The schema differs from the-odds-api:
#   * One row per (sportsbook, player, line, side) — we pivot Over/Under into
#     a single row per (player, book) at line=0.5.
#   * Player name lives in `player_name` (not buried in `description`).
#   * `is_live=False` is reliable for filtering pre-game lines.
#   * Combo props ("Trout & Soler 2+ Combined Hits") have selection != "Over"/
#     "Under" — we exclude those by requiring the selection_type sentinel.
#
# Switched on 2026-05-02 after the-odds-api's FanDuel feed was found to
# disagree with the live FD app by 10–25 cents — the source of much grief.
# Auth: X-API-Key header. Key in .env as SHARPAPI_KEY=sk_live_...
SHARPAPI_BASE = "https://api.sharpapi.io/api/v1"
SHARPAPI_PAGE_LIMIT = 100   # API caps somewhere around 100; tune up if allowed.


def fetch_sharpapi_hit_props(target_date: date) -> pd.DataFrame:
    """Fetch MLB 1+ hits props from SharpAPI for `target_date`.

    Filters applied:
      * league=MLB, market=player_hits
      * line=0.5 (the 1+ hits equivalent)
      * is_live=False (skip in-progress games)
      * single-player props only (drops "Combined Hits" combos)

    Output columns match the-odds-api fetcher: date, player_id, player_name,
    book, over_price, under_price, fetched_at. Downstream pipeline doesn't
    care which source produced the parquet.

    Requires env var SHARPAPI_KEY (set in .env).
    """
    api_key = env("SHARPAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "SHARPAPI_KEY not found. Add to .env:\n    SHARPAPI_KEY=sk_live_xxx"
        )

    fetched_at = datetime.now(timezone.utc).isoformat()

    # Paginate. SharpAPI returns has_more + next_offset in the pagination block,
    # but in practice has_more=True on the last page and the next request 400s.
    # Defensive: stop when a page returns fewer rows than the limit, OR when a
    # request returns a 4xx (which we treat as end-of-data, not a fatal error).
    rows: list[dict] = []
    offset = 0
    pages = 0
    while True:
        r = requests.get(
            f"{SHARPAPI_BASE}/odds",
            headers={"X-API-Key": api_key},
            params={
                "league": "MLB",
                "market": "player_hits",
                "limit": SHARPAPI_PAGE_LIMIT,
                "offset": offset,
            },
            timeout=20,
        )
        if r.status_code == 400:
            # Past the end of the result set — SharpAPI returns 400 when the
            # offset is out of range. Treat as end-of-data and stop cleanly.
            break
        if r.status_code >= 500:
            # Real server error — surface it.
            r.raise_for_status()
        if r.status_code >= 400:
            # Other 4xx (auth, rate limit, etc.) — surface so we don't
            # silently swallow real misconfigurations.
            r.raise_for_status()
        body = r.json()
        page_rows = body.get("data", [])
        rows.extend(page_rows)
        pages += 1

        # Stop conditions, in order:
        #   1. Page came back smaller than the requested limit → last page.
        #   2. Pagination block says no more.
        #   3. Defensive cap: prevent infinite loop on a buggy response.
        if len(page_rows) < SHARPAPI_PAGE_LIMIT:
            break
        pag = body.get("pagination", {}) or {}
        if not pag.get("has_more"):
            break
        next_off = pag.get("next_offset")
        if next_off is None:
            offset += len(page_rows)
        else:
            offset = next_off
        if pages > 100 or len(rows) > 50_000:
            print(f"  WARN: paginate cutoff at pages={pages} rows={len(rows)}")
            break

    print(f"  SharpAPI returned {len(rows)} player_hits rows across {pages} page(s).")
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    # 1. Pre-game only. is_live=True means the game has already started; the
    #    book has re-priced to "remaining PAs" rather than full game, so the
    #    line means something different than what the model assumes.
    before = len(raw)
    raw = raw[raw["is_live"].fillna(False) == False].copy()
    print(f"  is_live filter: {len(raw):,}/{before:,} rows kept (dropped in-progress games)")

    # 2. 1+ hits line only (line=0.5). The API also returns 1.5/2.5/3.5/4.5 — we
    #    don't model those today.
    before = len(raw)
    raw = raw[raw["line"] == 0.5].copy()
    print(f"  line=0.5 filter: {len(raw):,}/{before:,} rows kept (dropped alt lines)")

    # 3. Single-player props only. Combo rows look like:
    #       selection: "Trout & Soler 4+ Combined Hits"
    #       selection_type: "over" (or unique)
    #    Standard player rows have selection ∈ {"Over", "Under"} and a clean
    #    player_name. Drop anything where selection isn't Over/Under.
    before = len(raw)
    raw = raw[raw["selection"].isin(["Over", "Under"])].copy()
    print(f"  Over/Under filter: {len(raw):,}/{before:,} rows kept (dropped combo props)")

    # 4. Drop "Extra Base" markets. SharpAPI lumps FanDuel's "Extra Base Hit"
    #    market (doubles+triples+HRs only — singles don't count) under the
    #    same player_hits market_type, disambiguated by appending " Extra Base"
    #    to the player_name. Those prices are for a different bet than what
    #    our model predicts (P(1+ hit including singles)) — using them would
    #    silently corrupt the edge calculation.
    before = len(raw)
    raw = raw[~raw["player_name"].astype(str).str.contains(
        "Extra Base|Combined|\\+|/|&", case=False, na=False, regex=True,
    )].copy()
    print(f"  single-player filter: {len(raw):,}/{before:,} rows kept "
          f"(dropped Extra Base / combos / multi-player)")

    # 5. Sportsbooks the user has subscribed to. (FD and DK in practice today;
    #    HardRock/Fanatics/Caesars don't appear in MLB player_hits coverage.)
    raw = raw[raw["sportsbook"].notna()].copy()
    if not raw.empty:
        print(f"  per-book counts at line=0.5: "
              f"{dict(raw['sportsbook'].value_counts())}")

    # 5. Pivot Over/Under into a single row per (player, book). Each row in raw
    #    is one side of one player's prop at one book; we want both sides on
    #    one row so the existing parquet schema works downstream.
    if raw.empty:
        return pd.DataFrame()

    raw["player_name"] = raw["player_name"].astype(str).str.strip()
    pivot = (
        raw.pivot_table(
            index=["player_name", "sportsbook", "event_id",
                   "home_team", "away_team", "event_start_time"],
            columns="selection",
            values="odds_american",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"Over": "over_price", "Under": "under_price",
                         "sportsbook": "book"})
    )

    # Stamp the target date (UTC). event_start_time can cross midnight UTC for
    # late-night West Coast games, but the user's slate is always defined by
    # the calendar date they ran the workflow with.
    pivot["date"] = target_date.isoformat()
    pivot["fetched_at"] = fetched_at

    # 6. Match player_name to mlbam_id via players.parquet — same path the
    #    the-odds-api fetcher uses. Names should already be clean from SharpAPI.
    pivot = _match_player_ids(pivot, _load_player_map())
    pivot["player_id"] = pivot["mlbam_id"].astype("Int64")

    # 7. Final schema match. Keep the columns recommend.py + downstream expect.
    out_cols = [
        "date", "player_id", "player_name", "book",
        "over_price", "under_price", "fetched_at",
    ]
    out = pivot[out_cols].dropna(subset=["player_id"]).reset_index(drop=True)
    out_path = raw_path("props", f"{target_date.isoformat()}_props.parquet")
    out.to_parquet(out_path, index=False)
    print(f"{len(out)} props written -> {out_path.name}")

    if not out.empty:
        # Show a sample so the user can spot-check against FD/DK app prices.
        print(out.head(15).to_string(index=False))

    return out


def load_props(target_date: date, source: str = "csv") -> pd.DataFrame:
    if source == "csv":
        return load_manual_csv(target_date)
    if source == "theodds":
        return fetch_theodds_hit_props(target_date)
    if source == "sharpapi":
        return fetch_sharpapi_hit_props(target_date)
    if source == "draftkings":
        return fetch_draftkings_hit_props(target_date)
    raise ValueError(f"Unknown source: {source}")


def make_template_csv(target_date: date) -> Path:
    """Drop a blank template CSV with the right header for easy pasting."""
    path = raw_path("props", f"{target_date.isoformat()}_props.csv")
    if path.exists():
        print(f"Template already exists at {path}. Not overwriting.")
        return path
    pd.DataFrame(
        columns=["player_name", "over_price", "under_price", "book"]
    ).to_csv(path, index=False)
    print(f"Wrote empty template to {path}")
    print("Paste rows like:  Jose Altuve,-135,+110,DraftKings")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--source",
        choices=["csv", "theodds", "sharpapi", "draftkings"],
        default="sharpapi",
    )
    parser.add_argument(
        "--make-template", action="store_true",
        help="Create an empty CSV template for the date and exit.",
    )
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.make_template:
        make_template_csv(target)
    else:
        df = load_props(target, source=args.source)
        print(df.head(15).to_string(index=False))
