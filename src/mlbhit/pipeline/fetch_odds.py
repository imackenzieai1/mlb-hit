from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from ..config import env
from ..io import raw_path

ODDS = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def fetch_game_odds() -> pd.DataFrame:
    key = env("ODDS_API_KEY")
    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
    }
    r = requests.get(ODDS, params=params, timeout=10)
    r.raise_for_status()
    rows = []
    now = datetime.now(timezone.utc)
    for game in r.json():
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                for o in market.get("outcomes", []):
                    rows.append(
                        {
                            "fetched_at": now.isoformat(),
                            "commence_time": game["commence_time"],
                            "home_team": game["home_team"],
                            "away_team": game["away_team"],
                            "book": book["key"],
                            "market": market["key"],
                            "name": o["name"],
                            "price": o["price"],
                            "point": o.get("point"),
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        out = raw_path(
            "odds",
            f"{now.date().isoformat()}_{now.strftime('%H%M')}.parquet",
        )
        df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    df = fetch_game_odds()
    print(df.head())
