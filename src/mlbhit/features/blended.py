"""V1.5: PA/TBF-weighted blend of current-season-to-date stats with prior-season stats.

Core idea: early in a new season, a player's few current-season PAs carry almost
no signal. Their prior-season stats are much more informative. As current-season
PAs accumulate, the blend shifts toward current-season. Full signal at 150 PAs
(batters) / 100 TBF (pitchers).

    effective = w * current + (1 - w) * prior
    w_bat     = min(PA_current  / 150, 1.0)
    w_pit     = min(TBF_current / 100, 1.0)

Also exposes 14/30-day rolling-window aggregates from raw Statcast pitches for
future model retraining (not yet consumed by train.py).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from ..io import raw_path
from .batter import build_batter_features
from .pitcher import build_pitcher_features


# ---------------------------------------------------------------------------
# Season-to-date blend (main V1.5 feature)
# ---------------------------------------------------------------------------

_BAT_STAT_COLS = [
    "bat_xba_season",
    "bat_k_pct",
    "bat_hard_hit_pct",
    "bat_contact_pct",
    "bat_sweet_spot_pct",
    "bat_line_drive_pct",
    "bat_solid_contact_pct",
    # v3 platoon
    "bat_xba_vs_L",
    "bat_xba_vs_R",
    "bat_k_pct_vs_L",
    "bat_k_pct_vs_R",
]

_PIT_STAT_COLS = [
    "sp_xba_allowed",
    "sp_k_pct",
    "sp_hard_hit_allowed",
    "sp_sweet_spot_allowed",
    # v3 platoon + hittability
    "sp_xba_allowed_vs_L",
    "sp_xba_allowed_vs_R",
    "sp_k_pct_allowed_vs_L",
    "sp_k_pct_allowed_vs_R",
    "sp_zone_pct",
    "sp_contact_pct_allowed",
]


def _blend_columns(
    df: pd.DataFrame,
    stat_cols: list[str],
    weight_col: str,
) -> pd.DataFrame:
    """For each stat, blend the current-season value with the prior-season value
    using the precomputed per-row weight in `weight_col`. Missing values on
    either side are filled from the other side before blending."""
    for col in stat_cols:
        prior_col = f"{col}_prior"
        if col not in df.columns or prior_col not in df.columns:
            continue
        cur = df[col].astype("float64")
        pri = df[prior_col].astype("float64")
        # Symmetric fill: if current is missing use prior, and vice versa.
        cur_filled = cur.fillna(pri)
        pri_filled = pri.fillna(cur)
        df[col] = df[weight_col] * cur_filled + (1 - df[weight_col]) * pri_filled
    return df


def build_blended_batter_features(
    current_season: int,
    prior_season: int,
    full_signal_pa: int = 150,
) -> pd.DataFrame:
    """Drop-in replacement for build_batter_features(current_season) that blends
    with prior-season stats. Returns a DataFrame with the same columns as
    build_batter_features, tagged with season=current_season."""
    cur = build_batter_features(current_season)
    pri = build_batter_features(prior_season)

    if pri.empty:
        # Nothing to blend with — return current as-is.
        return cur

    # Identity columns stay on current side; everything else gets a _prior suffix.
    id_cols = {"mlbam_id", "player_name"}
    pri_rename = {c: f"{c}_prior" for c in pri.columns if c not in id_cols}
    pri = pri.rename(columns=pri_rename)
    pri_keep = ["mlbam_id"] + list(pri_rename.values())

    df = cur.merge(pri[pri_keep], on="mlbam_id", how="outer")

    # Backfill identity cols for prior-only players (shouldn't happen often
    # for current-roster scoring but keeps the output clean).
    for idc in ("player_name", "team", "season", "batter_hand"):
        if idc in df.columns and f"{idc}_prior" in df.columns:
            df[idc] = df[idc].fillna(df[f"{idc}_prior"])

    cur_pa = df["PA"].fillna(0).astype(float) if "PA" in df.columns else pd.Series(0.0, index=df.index)
    df["blend_w"] = (cur_pa / full_signal_pa).clip(0, 1)

    df = _blend_columns(df, _BAT_STAT_COLS, "blend_w")

    # Clean up: drop all _prior helper columns; set season.
    df = df.drop(columns=[c for c in df.columns if c.endswith("_prior")], errors="ignore")
    df["season"] = current_season
    return df


def build_blended_pitcher_features(
    current_season: int,
    prior_season: int,
    full_signal_tbf: int = 100,
) -> pd.DataFrame:
    cur = build_pitcher_features(current_season)
    pri = build_pitcher_features(prior_season)

    if pri.empty:
        return cur

    id_cols = {"mlbam_id", "pitcher_name"}
    pri_rename = {c: f"{c}_prior" for c in pri.columns if c not in id_cols}
    pri = pri.rename(columns=pri_rename)
    pri_keep = ["mlbam_id"] + list(pri_rename.values())

    df = cur.merge(pri[pri_keep], on="mlbam_id", how="outer")

    for idc in ("pitcher_name", "team", "season", "role", "pitcher_hand"):
        if idc in df.columns and f"{idc}_prior" in df.columns:
            df[idc] = df[idc].fillna(df[f"{idc}_prior"])

    cur_tbf = df["TBF"].fillna(0).astype(float) if "TBF" in df.columns else pd.Series(0.0, index=df.index)
    df["blend_w"] = (cur_tbf / full_signal_tbf).clip(0, 1)

    df = _blend_columns(df, _PIT_STAT_COLS, "blend_w")

    df = df.drop(columns=[c for c in df.columns if c.endswith("_prior")], errors="ignore")
    df["season"] = current_season
    return df


# ---------------------------------------------------------------------------
# Rolling windows (bonus — exposed for future model retraining)
# ---------------------------------------------------------------------------

HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "catcher_interf",
}


def _load_pitches(season: int) -> pd.DataFrame:
    path = raw_path("statcast", f"pitches_{season}.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _filter_window(pitches: pd.DataFrame, as_of: date, window_days: int) -> pd.DataFrame:
    """Return pitches in [as_of - window_days, as_of) — strictly prior to as_of.

    Strict "<" on as_of matches rolling.py's closed="left" semantics at training
    time, so live inference can't accidentally see games happening on the same
    day we're trying to predict (e.g. a day-game that finished before a
    night-game lineup we're scoring).
    """
    if pitches.empty:
        return pitches
    start = as_of - timedelta(days=window_days)
    return pitches[(pitches["game_date"] >= start) & (pitches["game_date"] < as_of)]


def compute_rolling_batter_stats(
    as_of: date,
    current_season: int,
    windows: Iterable[int] = (14, 30),
) -> pd.DataFrame:
    """One row per mlbam_id, columns suffixed with _{N}d (e.g. xba_14d, xba_30d).
    Returns an empty DataFrame if no pitch data is cached for `current_season`.
    """
    pitches = _load_pitches(current_season)
    if pitches.empty:
        return pd.DataFrame()

    out = None
    for w in windows:
        sub = _filter_window(pitches, as_of, w)
        if sub.empty:
            continue
        pa = sub[sub["events"].notna()].copy()
        pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
        pa["is_ab"] = (~pa["events"].isin(NON_AB_EVENTS)).astype(int)

        bb = sub[sub["type"] == "X"].copy()
        bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32)
        bb["is_hard_hit"] = bb["launch_speed"] >= 95

        agg = (
            pa.groupby("batter")
            .agg(
                PA=("events", "size"),
                AB=("is_ab", "sum"),
                H=("is_hit", "sum"),
                xba=("estimated_ba_using_speedangle", "mean"),
            )
            .reset_index()
            .rename(columns={"batter": "mlbam_id"})
        )
        bb_agg = (
            bb.groupby("batter")
            .agg(
                sweet_spot_pct=("is_sweet_spot", "mean"),
                hard_hit_pct=("is_hard_hit", "mean"),
            )
            .reset_index()
            .rename(columns={"batter": "mlbam_id"})
        )
        window_df = agg.merge(bb_agg, on="mlbam_id", how="left")
        window_df["ba"] = window_df["H"] / window_df["AB"].replace(0, pd.NA)

        rename = {c: f"{c}_{w}d" for c in window_df.columns if c != "mlbam_id"}
        window_df = window_df.rename(columns=rename)

        out = window_df if out is None else out.merge(window_df, on="mlbam_id", how="outer")

    return out if out is not None else pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=int, required=True, help="current season (e.g., 2026)")
    parser.add_argument("--prior", type=int, required=True, help="prior season (e.g., 2025)")
    args = parser.parse_args()

    bat = build_blended_batter_features(args.current, args.prior)
    pit = build_blended_pitcher_features(args.current, args.prior)
    print(f"blended batter rows: {len(bat)}  mean blend_w: {bat['blend_w'].mean():.3f}")
    print(f"blended pitcher rows: {len(pit)}  mean blend_w: {pit['blend_w'].mean():.3f}")
    print(bat.head(10).to_string())
