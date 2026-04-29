#!/usr/bin/env python
"""Augment the modeling parquet with v4-only features that aren't covered
by ``rolling.py``.

The Statcast-grade recent-form features (ba_3g/10g, xba_3g/10g,
hard_hit_pct_3g/10g, sweet_spot_pct_3g/10g, PA_3g/10g) live in
``data/clean/batter_rolling.parquet`` once you've run rolling.py with
``--game-windows 3 10``. ``build_features.py`` joins them onto the
modeling parquet automatically.

This script ONLY adds the two boxscore-derived sizing-context features that
rolling.py doesn't compute:

    hot_streak_avg     — last-6-game BA (continuous version of the binary
                          hot_streak flag used for 2x sizing)
    opp_consec_games   — opp team's calendar-day consecutive games streak
                          (fatigue/grind proxy)

Inputs:
    data/modeling/player_game_features.parquet   (existing v3 features +
                                                  3g/10g if rolling.py rerun)
    data/clean/boxscores_{year}.parquet          (per-game per-batter)

Output:
    data/modeling/player_game_features.parquet   (in-place, with 2 new cols)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlbhit.features.recent_form import (  # noqa: E402
    attach_hot_streak,
    attach_opp_grind,
)
from mlbhit.io import clean_path, modeling_path  # noqa: E402

NEW_COLS = ["hot_streak_avg", "opp_consec_games"]


def _load_boxscores(seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for yr in sorted(set(seasons)):
        p = clean_path(f"boxscores_{yr}.parquet")
        if not p.exists():
            print(f"  WARN: missing {p}; skipping season {yr}.")
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    in_path = modeling_path("player_game_features.parquet")
    print(f"loading modeling parquet: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"  rows: {len(df):,}")
    print(f"  date span: {df['date'].astype(str).min()} -> {df['date'].astype(str).max()}")

    # Sanity check: confirm rolling.py game-windows have already been merged
    # in. If not, warn so the user knows v4 will be missing the xba_3g/10g
    # set entirely (still trainable, just weaker).
    rolling_cols = ["ba_3g", "xba_3g", "hard_hit_pct_3g",
                    "ba_10g", "xba_10g", "hard_hit_pct_10g"]
    missing_rolling = [c for c in rolling_cols if c not in df.columns]
    if missing_rolling:
        print()
        print(f"  WARN: modeling parquet is missing rolling game-windows:")
        for c in missing_rolling:
            print(f"    - {c}")
        print(f"  To fix: run")
        print(f"    python -m mlbhit.features.rolling --seasons 2023 2024 2025 2026 \\")
        print(f"        --windows 14 30 --game-windows 3 10")
        print(f"    python -m mlbhit.pipeline.build_features  # rebuild modeling parquet")
        print(f"  Then re-run THIS script. Continuing to add boxscore features anyway...")
    else:
        present = sum(df[c].notna().sum() for c in rolling_cols)
        print(f"  rolling 3g/10g present (non-null cells across 6 cols: {present:,})")

    # Date strings → years for boxscore lookup.
    df["date"] = df["date"].astype(str)
    seasons = sorted({int(d[:4]) for d in df["date"]})
    print(f"  seasons in scope: {seasons}")

    box = _load_boxscores(seasons)
    if box.empty:
        print("  ERROR: no boxscores on disk; cannot compute v4 features.")
        sys.exit(1)
    box["date"] = box["date"].astype(str)
    print(f"  boxscores rows: {len(box):,}  ({box['date'].min()} -> {box['date'].max()})")

    # 1. hot_streak_avg (last-6 BA from boxscores). attach_hot_streak returns
    #    several columns; we only persist hot_streak_avg as a model feature.
    print("\n[1/2] computing hot_streak_avg (last-6-game BA)...")
    targets_b = df[["player_id", "date"]].copy()
    hot = attach_hot_streak(targets_b, box)
    df["hot_streak_avg"] = hot["hot_streak_avg"].values

    # 2. opp_consec_games. Modeling parquet uses `opponent`; recent_form
    #    expects `opp_team`.
    print("[2/2] computing opp_consec_games...")
    if "opponent" in df.columns:
        targets_g = df[["opponent", "date"]].rename(columns={"opponent": "opp_team"})
        grind = attach_opp_grind(targets_g, box)
        df["opp_consec_games"] = grind["opp_consec_games"].values
    else:
        print("  WARN: no `opponent` column in modeling parquet; opp_consec_games stays NaN.")
        df["opp_consec_games"] = pd.NA

    # Sanity print: distribution of new columns.
    print("\nNew column summary:")
    for c in NEW_COLS:
        s = df[c]
        non_null = s.notna().sum()
        if non_null > 0:
            try:
                print(f"  {c:20s} non_null={non_null:>7d}  mean={s.mean():.4f}  "
                      f"min={s.min():.3f}  max={s.max():.3f}")
            except (TypeError, ValueError):
                print(f"  {c:20s} non_null={non_null:>7d}  (non-numeric summary skipped)")
        else:
            print(f"  {c:20s} all NaN — check upstream.")

    out_path = in_path
    df.to_parquet(out_path, index=False)
    print(f"\nwrote: {out_path}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
