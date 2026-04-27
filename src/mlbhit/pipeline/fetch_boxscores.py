from __future__ import annotations

import argparse
import time
from datetime import date

import pandas as pd
import statsapi
from tqdm import tqdm

from ..io import clean_path
from ..utils.dates import daterange


def _day_cache_path(d: date):
    p = clean_path("daily_boxscores")
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{d.isoformat()}.parquet"


def _is_final(status: str) -> bool:
    if not status:
        return False
    return (
        status.startswith("Final")
        or status.startswith("Completed")
        or status == "Game Over"
    )


def fetch_day(d: date, sleep_between_games: float = 0.3, verbose: bool = False) -> list[dict]:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    if verbose:
        seen = sorted(set(g.get("status", "?") for g in games))
        print(f"  [{d}] {len(games)} games on schedule, statuses={seen}")

    rows = []
    finals = 0
    for g in games:
        if not _is_final(g.get("status", "")):
            continue
        finals += 1
        try:
            box = statsapi.boxscore_data(g["game_id"])
        except Exception as e:
            if verbose:
                print(f"    boxscore_data FAILED for game {g.get('game_id')}: {e}")
            continue

        for side in ("home", "away"):
            try:
                team = box["teamInfo"][side]["abbreviation"]
                opp_side = "away" if side == "home" else "home"
                opp = box["teamInfo"][opp_side]["abbreviation"]
            except KeyError as e:
                if verbose:
                    print(f"    teamInfo missing key for game {g.get('game_id')}: {e}")
                continue

            players = box.get(side, {}).get("players", {})
            venue = g.get("venue_name")
            for _pid, p in players.items():
                stats = p.get("stats", {}).get("batting", {})
                if not stats:
                    continue
                ab = int(stats.get("atBats", 0) or 0)
                pa = int(stats.get("plateAppearances", 0) or 0)
                hits = int(stats.get("hits", 0) or 0)
                bb_stat = int(stats.get("baseOnBalls", 0) or 0)
                hbp = int(stats.get("hitByPitch", 0) or 0)
                sf = int(stats.get("sacFlies", 0) or 0)
                sh = int(stats.get("sacBunts", 0) or 0)
                # Fallback if API stops sending plateAppearances directly
                if pa == 0:
                    pa = ab + bb_stat + hbp + sf + sh
                if pa == 0:
                    continue

                bo = p.get("battingOrder")
                batting_order = None
                if bo is not None and str(bo).isdigit():
                    v = int(bo) // 100
                    batting_order = v if v > 0 else None

                rows.append({
                    "date": d.isoformat(),
                    "game_pk": g["game_id"],
                    "player_id": int(p["person"]["id"]),
                    "player_name": p["person"]["fullName"],
                    "team": team,
                    "opponent": opp,
                    "home_away": "H" if side == "home" else "A",
                    "venue": venue,
                    "batting_order": batting_order,
                    "ab": ab,
                    "pa": pa,
                    "hits": hits,
                    "got_hit": int(hits > 0),
                })
        time.sleep(sleep_between_games)

    if verbose:
        print(f"  [{d}] finals={finals}  player_rows={len(rows)}")
    return rows


def fetch_range(start: date, end: date, use_cache: bool = True, verbose_first_n: int = 3) -> pd.DataFrame:
    frames = []
    diagnosed = 0
    for d in tqdm(list(daterange(start, end))):
        cache = _day_cache_path(d)
        if use_cache and cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        verbose = diagnosed < verbose_first_n
        rows = fetch_day(d, verbose=verbose)
        diagnosed += 1
        day_df = pd.DataFrame(rows)
        if not day_df.empty:
            day_df.to_parquet(cache, index=False)
            frames.append(day_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_season(season: int) -> pd.DataFrame:
    start = date(season, 3, 20)
    end = date(season, 11, 5)
    df = fetch_range(start, end)
    out = clean_path(f"boxscores_{season}.parquet")
    df.to_parquet(out, index=False)
    if df.empty:
        print(f"[{season}] EMPTY dataframe -> {out}")
    else:
        print(f"[{season}] rows={df.shape[0]}  hit_rate={df['got_hit'].mean():.3f}  -> {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.no_cache:
        df = fetch_range(date(args.season, 3, 20), date(args.season, 11, 5), use_cache=False)
        df.to_parquet(clean_path(f"boxscores_{args.season}.parquet"), index=False)
        print(df.shape, df["got_hit"].mean() if not df.empty else "empty")
    else:
        run_season(args.season)
