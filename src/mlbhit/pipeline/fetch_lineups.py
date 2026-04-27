from __future__ import annotations

from datetime import date

import pandas as pd
import statsapi

from ..io import raw_path


# Schema written to disk even when MLB has posted zero lineups. score_today
# reads this parquet expecting these columns; falling back to projections is
# the right behavior when the parquet is empty, but it crashes if the column
# names are missing entirely. Keeping the schema stable lets the early-morning
# cron run survive the no-lineups-yet state.
LINEUP_COLS = [
    "date",
    "game_pk",
    "team",
    "opponent",
    "home_away",
    "player_id",
    "player_name",
    "lineup_spot",
    "lineup_confirmed",
]


def fetch_lineups(d: date) -> pd.DataFrame:
    games = statsapi.schedule(date=d.strftime("%m/%d/%Y"))
    rows = []
    for g in games:
        pk = g["game_id"]
        try:
            box = statsapi.boxscore_data(pk)
        except Exception:
            continue
        for side in ("home", "away"):
            team = box["teamInfo"][side]["abbreviation"]
            opp_side = "away" if side == "home" else "home"
            opp = box["teamInfo"][opp_side]["abbreviation"]
            order = box[side].get("battingOrder", [])
            for spot, pid in enumerate(order, start=1):
                pid = int(pid)
                player = box[side]["players"].get(f"ID{pid}", {})
                name = player.get("person", {}).get("fullName", "")
                rows.append(
                    {
                        "date": d.isoformat(),
                        "game_pk": pk,
                        "team": team,
                        "opponent": opp,
                        "home_away": "H" if side == "home" else "A",
                        "player_id": pid,
                        "player_name": name,
                        "lineup_spot": spot,
                        "lineup_confirmed": bool(order),
                    }
                )
    if rows:
        df = pd.DataFrame(rows)
    else:
        # Empty slate (typical for the early-morning cron before any team has
        # posted lineups). Write a column-stable empty parquet so score_today
        # can read it, see zero rows, and fall back to project_lineups cleanly.
        df = pd.DataFrame(columns=LINEUP_COLS)
    out = raw_path("lineups", f"{d.isoformat()}.parquet")
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (default: today). Used for cloud-runner backfills.",
    )
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    df = fetch_lineups(target)
    if df.empty:
        # Zero confirmed lineups isn't an error — it's the normal state of
        # the world before teams post lineup cards (~3-4h before first pitch).
        # score_today will fall back to project_lineups for the bet generator.
        print(f"  no confirmed lineups yet for {target.isoformat()} "
              f"(0 slots) — score_today will use projections.")
    else:
        print(df.head())
        print(f"{int(df['lineup_confirmed'].sum())} confirmed lineup slots")
