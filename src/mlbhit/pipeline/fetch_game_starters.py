"""Pull actual starting pitchers per game from the MLB boxscore endpoint.

Why not the schedule: MLB's /schedule endpoint returns probablePitcher only for
FUTURE games. For completed games, that field is empty. The starter is always
available on the boxscore (`home.pitchers[0]` / `away.pitchers[0]` = first
pitcher to throw for that side = the starter).

Output: data/clean/game_starters_{season}.parquet
Schema: game_pk, date, home_starter_id, away_starter_id

build_features.py will join this on game_pk to populate opp_sp_id, which
currently goes NaN for 100% of historical training rows (that's why
sp_xba_allowed etc. all had 0 importance).

Usage:
    python -m mlbhit.pipeline.fetch_game_starters --season 2023
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import statsapi
from tqdm import tqdm

from ..io import clean_path


def _starters_from_box(box: dict) -> tuple[int | None, int | None]:
    """Return (home_starter_id, away_starter_id) — first pitcher listed per side.

    Box shape: box["home"]["pitchers"] is a list of int person-ids in the order
    they appeared. The first element is the starter.
    """
    def _first(side: str) -> int | None:
        ps = box.get(side, {}).get("pitchers", [])
        return int(ps[0]) if ps else None

    return _first("home"), _first("away")


def fetch_starters_for_season(
    season: int,
    sleep_between_games: float = 0.2,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Iterate game_pks from the season's boxscores parquet and fetch starters.

    Caches per-game in data/clean/game_starters_cache/{game_pk}.json so a
    second run picks up where it left off if interrupted.
    """
    box_path = clean_path(f"boxscores_{season}.parquet")
    if not box_path.exists():
        raise FileNotFoundError(
            f"{box_path} missing — can't infer which games to pull. "
            "Run fetch_boxscores for this season first."
        )
    box = pd.read_parquet(box_path)
    games = box[["game_pk", "date"]].drop_duplicates().reset_index(drop=True)
    print(f"[{season}] {len(games)} unique games to pull starters for")

    cache_dir = clean_path("game_starters_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    out_path = clean_path(f"game_starters_{season}.parquet")
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        already_done = set(existing["game_pk"])
        remaining = games[~games["game_pk"].isin(already_done)]
        print(f"  found {len(already_done)} already cached; {len(remaining)} remaining")
    else:
        existing = pd.DataFrame()
        remaining = games

    rows = []
    for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
        game_pk = int(row["game_pk"])
        try:
            b = statsapi.boxscore_data(game_pk)
            home_id, away_id = _starters_from_box(b)
        except Exception as e:
            print(f"  skip game {game_pk}: {e}")
            continue
        rows.append({
            "game_pk": game_pk,
            "date": row["date"],
            "home_starter_id": home_id,
            "away_starter_id": away_id,
        })
        time.sleep(sleep_between_games)

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    combined.to_parquet(out_path, index=False)

    coverage = combined["home_starter_id"].notna().mean() if len(combined) else 0.0
    print(f"[{season}] wrote {len(combined)} rows -> {out_path}  "
          f"(home_starter_id coverage: {coverage:.1%})")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch every game, ignore cached parquet.")
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds between API calls (default 0.2).")
    args = parser.parse_args()

    fetch_starters_for_season(
        args.season, sleep_between_games=args.sleep, overwrite=args.overwrite
    )
