"""Leakage-safe 14/30-day rolling batter stats, derived from Statcast pitches.

Why: season-level stats don't capture recent form. A batter hitting .200 overall
but .340 over his last 30 days is a very different expected-value profile than
a batter doing the reverse. XGBoost can learn to weight both, but only if we
feed it the rolling values.

Critical leakage detail: for each (batter, game_date), the rolling window
includes *only games strictly before that date*. We use `closed="left"` on a
datetime-indexed rolling, which excludes the current row from its own window.
Per-season rolling (we don't blend across off-seasons) — opening day of a new
season has blank rolling features by design.

Output: data/clean/batter_rolling.parquet
Schema: mlbam_id, game_date, <col>_14d, <col>_30d for:
    PA, AB, H, ba, xba, hard_hit_pct, sweet_spot_pct

Join key: (mlbam_id=player_id, game_date=date) in build_features.
"""
from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from ..io import clean_path, raw_path


HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {
    "walk", "intent_walk", "hit_by_pitch",
    "sac_bunt", "sac_fly", "sac_fly_double_play",
    "sac_bunt_double_play", "catcher_interf",
}

# Aggregation-friendly numeric daily columns we'll sum in the rolling window.
_DAILY_NUMERIC = [
    "PA", "AB", "H",
    "xba_sum", "xba_cnt",
    "hard_hit_cnt", "sweet_spot_cnt",
    "bb_total",
]


def _daily_batter_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level rows to one row per (batter, game_date).

    We keep numerator+denominator pairs (e.g. xba_sum / xba_cnt) so rolling
    sums aggregate correctly — averaging the daily means would weight a
    1-PA day the same as a 5-PA day.
    """
    p = pitches[["batter", "game_date", "events", "type",
                 "estimated_ba_using_speedangle", "launch_angle", "launch_speed"]].copy()
    p["game_date"] = pd.to_datetime(p["game_date"]).dt.normalize()

    pa = p[p["events"].notna()].copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_ab"] = (~pa["events"].isin(NON_AB_EVENTS)).astype(int)

    pa_daily = (
        pa.groupby(["batter", "game_date"])
        .agg(
            PA=("events", "size"),
            AB=("is_ab", "sum"),
            H=("is_hit", "sum"),
            xba_sum=("estimated_ba_using_speedangle", "sum"),
            xba_cnt=("estimated_ba_using_speedangle", "count"),
        )
        .reset_index()
    )

    # Drop NaN launch metrics BEFORE boolean coercion — Statcast sometimes logs
    # a batted ball (type=="X") without launch_angle/speed (bunt, dropped reads,
    # spring-training gaps). .between() on NA returns NA, and astype(int) on an
    # NA-bearing nullable bool blows up with "cannot convert NA to integer".
    bb = p[(p["type"] == "X") & p["launch_angle"].notna() & p["launch_speed"].notna()].copy()
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32).astype(int)
    bb["is_hard_hit"] = (bb["launch_speed"] >= 95).astype(int)

    bb_daily = (
        bb.groupby(["batter", "game_date"])
        .agg(
            hard_hit_cnt=("is_hard_hit", "sum"),
            sweet_spot_cnt=("is_sweet_spot", "sum"),
            bb_total=("is_hard_hit", "size"),
        )
        .reset_index()
    )

    daily = pa_daily.merge(bb_daily, on=["batter", "game_date"], how="outer")
    for c in _DAILY_NUMERIC:
        if c not in daily.columns:
            daily[c] = 0
    daily[_DAILY_NUMERIC] = daily[_DAILY_NUMERIC].fillna(0)
    return daily


def _roll_sum(daily: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Per-batter rolling SUM over the last `window_days`, strictly prior.

    `closed="left"` on the offset window means the current row's own contribution
    is excluded — which is what we want for a feature fed into a model that
    predicts that same row's outcome.
    """
    d = daily.sort_values(["batter", "game_date"]).copy()
    d = d.set_index("game_date")

    rolled = (
        d.groupby("batter", group_keys=False)[_DAILY_NUMERIC]
        .rolling(f"{window_days}D", closed="left")
        .sum()
    )
    rolled = rolled.reset_index()
    # Rename sums to window-tagged column names and derive rates.
    suffix = f"_{window_days}d"
    rolled = rolled.rename(columns={c: f"{c}{suffix}" for c in _DAILY_NUMERIC})

    rolled[f"ba{suffix}"] = rolled[f"H{suffix}"] / rolled[f"AB{suffix}"].replace(0, pd.NA)
    rolled[f"xba{suffix}"] = rolled[f"xba_sum{suffix}"] / rolled[f"xba_cnt{suffix}"].replace(0, pd.NA)
    rolled[f"hard_hit_pct{suffix}"] = rolled[f"hard_hit_cnt{suffix}"] / rolled[f"bb_total{suffix}"].replace(0, pd.NA)
    rolled[f"sweet_spot_pct{suffix}"] = rolled[f"sweet_spot_cnt{suffix}"] / rolled[f"bb_total{suffix}"].replace(0, pd.NA)

    keep = [
        "batter", "game_date",
        f"PA{suffix}", f"AB{suffix}", f"H{suffix}",
        f"ba{suffix}", f"xba{suffix}",
        f"hard_hit_pct{suffix}", f"sweet_spot_pct{suffix}",
    ]
    return rolled[keep]


def _roll_sum_games(daily: pd.DataFrame, n_games: int) -> pd.DataFrame:
    """Per-batter rolling SUM over the last `n_games` PA-counted days,
    strictly prior to the current row.

    Companion to `_roll_sum` but uses a GAME COUNT window instead of a CALENDAR
    DAY window. Every row in `daily` is one PA-counted game (PA >= 1 by virtue
    of having an aggregated PA count > 0), so a count-N rolling on the per-batter
    series gives the last N actual at-bat opportunities — robust to off-days,
    IL stints, and doubleheaders that the day-window can't differentiate.

    Output columns mirror `_roll_sum` but with `_{n}g` suffix:
        PA_{n}g, AB_{n}g, H_{n}g,
        ba_{n}g, xba_{n}g, hard_hit_pct_{n}g, sweet_spot_pct_{n}g

    Notes
    -----
    Implementation uses an explicit per-batter loop rather than
    ``groupby().rolling()`` because pandas' chained groupby+rolling produces
    a MultiIndex output that reorders rows by group key — re-attaching keys
    by positional alignment (``d["batter"].values``) silently misaligns. The
    loop is O(batters * games_per_batter) ≈ 5s/season at our scale, which is
    fine for a once-per-season feature build.
    """
    d = daily.sort_values(["batter", "game_date"]).copy()

    # Drop daily rows where the batter never came up (PA == 0). For game-count
    # windowing those rows would dilute "his last N at-bat opportunities."
    d = d[d["PA"].fillna(0) > 0].copy()

    suffix = f"_{n_games}g"

    chunks = []
    for batter, g in d.groupby("batter", sort=False):
        # rolling().sum() with default closed="right" includes the current row.
        # shift(1) bumps each row's value to the NEXT row, so each row gets the
        # sum of its prior n_games rows — equivalent to closed="left".
        rolled = g[_DAILY_NUMERIC].rolling(window=n_games, min_periods=1).sum().shift(1)
        rolled = rolled.rename(columns={c: f"{c}{suffix}" for c in _DAILY_NUMERIC})
        rolled["batter"] = batter
        rolled["game_date"] = g["game_date"].values
        chunks.append(rolled)

    if not chunks:
        return pd.DataFrame(columns=["batter", "game_date"])
    rolled = pd.concat(chunks, ignore_index=True)

    rolled[f"ba{suffix}"] = rolled[f"H{suffix}"] / rolled[f"AB{suffix}"].replace(0, pd.NA)
    rolled[f"xba{suffix}"] = rolled[f"xba_sum{suffix}"] / rolled[f"xba_cnt{suffix}"].replace(0, pd.NA)
    rolled[f"hard_hit_pct{suffix}"] = rolled[f"hard_hit_cnt{suffix}"] / rolled[f"bb_total{suffix}"].replace(0, pd.NA)
    rolled[f"sweet_spot_pct{suffix}"] = rolled[f"sweet_spot_cnt{suffix}"] / rolled[f"bb_total{suffix}"].replace(0, pd.NA)

    keep = [
        "batter", "game_date",
        f"PA{suffix}", f"AB{suffix}", f"H{suffix}",
        f"ba{suffix}", f"xba{suffix}",
        f"hard_hit_pct{suffix}", f"sweet_spot_pct{suffix}",
    ]
    return rolled[keep]


def build_batter_rolling(
    seasons: Iterable[int],
    windows: Iterable[int] = (14, 30),
    game_windows: Iterable[int] = (3, 10),
) -> pd.DataFrame:
    """Produce per-(mlbam_id, game_date) rolling features across all given seasons.

    Each season rolls independently (no off-season carryover). ``windows``
    are calendar-day windows (compatible with the v3 set); ``game_windows``
    are game-count windows added in v4 (last-N PA-counted games per batter).
    Pass an empty tuple for either to disable that side.

    Writes to data/clean/batter_rolling.parquet and returns the combined DataFrame.
    """
    per_season = []
    for season in seasons:
        src = raw_path("statcast", f"pitches_{season}.parquet")
        if not src.exists():
            print(f"[{season}] pitches parquet not found at {src} — skipping")
            continue
        print(f"[{season}] loading pitches...")
        pitches = pd.read_parquet(src)
        print(f"[{season}] aggregating to daily stats ({len(pitches):,} pitches)...")
        daily = _daily_batter_stats(pitches)
        print(f"[{season}] {len(daily):,} (batter, date) daily rows — rolling...")

        out = None
        for w in windows:
            r = _roll_sum(daily, w)
            out = r if out is None else out.merge(r, on=["batter", "game_date"], how="outer")
        for g in game_windows:
            r = _roll_sum_games(daily, g)
            out = r if out is None else out.merge(r, on=["batter", "game_date"], how="outer")
        if out is None:
            print(f"[{season}] no windows requested — skipping season")
            continue
        out["season"] = season
        per_season.append(out)

    if not per_season:
        raise FileNotFoundError("No pitches parquet found for any season in --seasons.")

    combined = pd.concat(per_season, ignore_index=True)
    combined = combined.rename(columns={"batter": "mlbam_id"})
    # Normalize game_date to ISO string to match how build_features joins on the box "date" col.
    combined["date"] = pd.to_datetime(combined["game_date"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop(columns=["game_date"])

    out_path = clean_path("batter_rolling.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"wrote {len(combined):,} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2023, 2024, 2025, 2026],
        help="Seasons to build rolling features for. Skips any season whose pitches parquet doesn't exist.",
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[14, 30],
        help="Rolling window sizes in calendar days.",
    )
    parser.add_argument(
        "--game-windows", nargs="+", type=int, default=[3, 10],
        help="Rolling window sizes in PA-counted games (v4 features).",
    )
    args = parser.parse_args()
    df = build_batter_rolling(
        args.seasons, windows=args.windows, game_windows=args.game_windows,
    )
    print(df.head(15).to_string())
