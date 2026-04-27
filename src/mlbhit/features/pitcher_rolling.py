"""Leakage-safe 14/30-day rolling pitcher stats, derived from Statcast pitches.

Mirrors features/rolling.py for batters but groups by `pitcher` instead of
`batter`. Same `closed="left"` rolling-window semantics so a starter's row at
date D can never see his own start at D.

Output: data/clean/pitcher_rolling.parquet
Schema: mlbam_id, date, sp_xba_allowed_<W>d, sp_k_pct_<W>d,
        sp_hard_hit_allowed_<W>d, sp_contact_pct_allowed_<W>d
        plus exposure: TBF_<W>d (so we know whether the rolling number is real
        or noise — a starter coming back from the IL will have low TBF).

Join key in build_features: (opp_sp_id=mlbam_id, date).
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
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul",
    "foul_tip", "hit_into_play", "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}

_DAILY_NUMERIC = [
    "TBF", "AB", "H_allowed",
    "xba_sum", "xba_cnt",
    "K", "BB",
    "hard_hit_cnt", "bb_total",
    "swings", "whiffs",
]


def _daily_pitcher_stats(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, game_date). Numerator+denominator pairs so rolling
    sums aggregate correctly when normalized into rates."""
    p = pitches[["pitcher", "game_date", "events", "type", "description",
                 "estimated_ba_using_speedangle", "launch_angle", "launch_speed"]].copy()
    p["game_date"] = pd.to_datetime(p["game_date"]).dt.normalize()

    pa = p[p["events"].notna()].copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_ab"] = (~pa["events"].isin(NON_AB_EVENTS)).astype(int)
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)

    pa_daily = (
        pa.groupby(["pitcher", "game_date"])
        .agg(
            TBF=("events", "size"),
            AB=("is_ab", "sum"),
            H_allowed=("is_hit", "sum"),
            K=("is_k", "sum"),
            BB=("is_bb", "sum"),
            xba_sum=("estimated_ba_using_speedangle", "sum"),
            xba_cnt=("estimated_ba_using_speedangle", "count"),
        )
        .reset_index()
    )

    bb = p[(p["type"] == "X") & p["launch_speed"].notna()].copy()
    bb["is_hard_hit"] = (bb["launch_speed"] >= 95).astype(int)
    bb_daily = (
        bb.groupby(["pitcher", "game_date"])
        .agg(
            hard_hit_cnt=("is_hard_hit", "sum"),
            bb_total=("is_hard_hit", "size"),
        )
        .reset_index()
    )

    sw = p.copy()
    sw["is_swing"] = sw["description"].isin(SWING_DESCRIPTIONS).astype(int)
    sw["is_whiff"] = sw["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    sw_daily = (
        sw.groupby(["pitcher", "game_date"])
        .agg(swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"))
        .reset_index()
    )

    daily = pa_daily.merge(bb_daily, on=["pitcher", "game_date"], how="outer")
    daily = daily.merge(sw_daily, on=["pitcher", "game_date"], how="outer")
    for c in _DAILY_NUMERIC:
        if c not in daily.columns:
            daily[c] = 0
    daily[_DAILY_NUMERIC] = daily[_DAILY_NUMERIC].fillna(0)
    return daily


def _roll_sum(daily: pd.DataFrame, window_days: int) -> pd.DataFrame:
    d = daily.sort_values(["pitcher", "game_date"]).copy()
    d = d.set_index("game_date")
    rolled = (
        d.groupby("pitcher", group_keys=False)[_DAILY_NUMERIC]
        .rolling(f"{window_days}D", closed="left")
        .sum()
    )
    rolled = rolled.reset_index()
    suffix = f"_{window_days}d"
    rolled = rolled.rename(columns={c: f"{c}{suffix}" for c in _DAILY_NUMERIC})

    tbf = rolled[f"TBF{suffix}"].replace(0, pd.NA)
    bb_tot = rolled[f"bb_total{suffix}"].replace(0, pd.NA)
    swings = rolled[f"swings{suffix}"].replace(0, pd.NA)

    rolled[f"sp_xba_allowed{suffix}"] = rolled[f"xba_sum{suffix}"] / rolled[f"xba_cnt{suffix}"].replace(0, pd.NA)
    rolled[f"sp_k_pct{suffix}"] = rolled[f"K{suffix}"] / tbf
    rolled[f"sp_hard_hit_allowed{suffix}"] = rolled[f"hard_hit_cnt{suffix}"] / bb_tot
    # Contact% = (swings - whiffs) / swings
    rolled[f"sp_contact_pct_allowed{suffix}"] = (
        rolled[f"swings{suffix}"] - rolled[f"whiffs{suffix}"]
    ) / swings

    keep = [
        "pitcher", "game_date",
        f"TBF{suffix}",
        f"sp_xba_allowed{suffix}",
        f"sp_k_pct{suffix}",
        f"sp_hard_hit_allowed{suffix}",
        f"sp_contact_pct_allowed{suffix}",
    ]
    return rolled[keep]


def build_pitcher_rolling(
    seasons: Iterable[int],
    windows: Iterable[int] = (14, 30),
) -> pd.DataFrame:
    per_season = []
    for season in seasons:
        src = raw_path("statcast", f"pitches_{season}.parquet")
        if not src.exists():
            print(f"[{season}] pitches parquet not found — skipping")
            continue
        print(f"[{season}] loading pitches...")
        pitches = pd.read_parquet(src)
        print(f"[{season}] aggregating ({len(pitches):,} pitches)...")
        daily = _daily_pitcher_stats(pitches)
        print(f"[{season}] {len(daily):,} (pitcher, date) daily rows — rolling...")

        out = None
        for w in windows:
            r = _roll_sum(daily, w)
            out = r if out is None else out.merge(r, on=["pitcher", "game_date"], how="outer")
        out["season"] = season
        per_season.append(out)

    if not per_season:
        raise FileNotFoundError("No pitches parquet found for any --seasons.")

    combined = pd.concat(per_season, ignore_index=True)
    combined = combined.rename(columns={"pitcher": "mlbam_id"})
    combined["date"] = pd.to_datetime(combined["game_date"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop(columns=["game_date"])

    out_path = clean_path("pitcher_rolling.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"wrote {len(combined):,} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2023, 2024, 2025, 2026])
    parser.add_argument("--windows", nargs="+", type=int, default=[14, 30])
    args = parser.parse_args()
    df = build_pitcher_rolling(args.seasons, windows=args.windows)
    print(df.head(10).to_string())
