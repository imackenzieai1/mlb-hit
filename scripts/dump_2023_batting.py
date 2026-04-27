#!/usr/bin/env python
"""Dump the per-game 2023 batting parquet to a CSV for browsing in Numbers/Excel.

Source : data/clean/boxscores_2023.parquet  (one row per batter per game)
Output : data/output/batting_2023.csv

Columns shipped (matches the parquet schema):
    date, game_pk, player_id, player_name, team, opponent, home_away,
    venue, batting_order, ab, pa, hits, got_hit

`got_hit` is the binary 1/0 the model is actually trained to predict
(1 = batter recorded at least one hit in this game). Useful for filtering.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "clean" / "boxscores_2023.parquet"
OUT = REPO_ROOT / "data" / "output" / "batting_2023.csv"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source parquet: {SRC}")

    df = pd.read_parquet(SRC)

    # Chronological sort makes the file easiest to scan top-to-bottom; sorting
    # in Numbers/Excel afterwards is one click if you want a different cut.
    df = df.sort_values(
        ["date", "game_pk", "home_away", "batting_order"],
        ascending=[True, True, True, True],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    n_games = df["game_pk"].nunique()
    n_players = df["player_id"].nunique()
    hit_rate = df["got_hit"].mean() * 100
    print(f"  wrote {len(df):,} rows -> {OUT}")
    print(f"  span: {df['date'].min()} -> {df['date'].max()}")
    print(f"  {n_games:,} unique games, {n_players:,} unique batters")
    print(f"  base hit-rate (1+ hit per game): {hit_rate:.1f}%")


if __name__ == "__main__":
    main()
