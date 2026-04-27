"""Aggregate Statcast pitches into season-level batter and pitcher stats.

v3 additions:
- Platoon splits: bat_xba_vs_L/R, bat_k_pct_vs_L/R, same for pitchers.
- Batter hand (mode of `stand`) and pitcher hand (mode of `p_throws`).
- Pitcher Zone% (pct of pitches in zone 1-9) and Contact% (1 - whiff/swing).
- Min-AB floor dropped to 1 in settings.yaml — rookies are kept, and the
  downstream regress() step in batter.py shrinks small samples toward the
  league mean so low-AB rows don't contaminate the model.

Outputs:
    data/clean/batter_season_stats.parquet
    data/clean/pitcher_season_stats.parquet

Usage:
    python -m mlbhit.pipeline.fetch_stats_from_statcast --seasons 2023 2024 2025 2026
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pybaseball import statcast

from ..io import clean_path, raw_path
from ..config import SETTINGS

MIN_AB = SETTINGS["filters"]["min_batter_ab"]
MIN_BF = SETTINGS["filters"]["min_pitcher_bf"]

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
K_EVENTS = {"strikeout", "strikeout_double_play"}

# Swing descriptors from Statcast `description`. Contact = swing that resulted
# in any outcome other than a whiff. We count foul tips (fouls) as contact.
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "foul_bunt",
    "missed_bunt",
    "bunt_foul_tip",
}
WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}


def _pull_pitches(year: int) -> pd.DataFrame:
    """Pull or load cached Statcast pitch-by-pitch data for one season."""
    path = raw_path("statcast", f"pitches_{year}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    df = statcast(f"{year}-03-20", f"{year}-11-05")
    df.to_parquet(path, index=False)
    return df


def _annotate_teams(pitches: pd.DataFrame) -> pd.DataFrame:
    """Add batter_team and pitcher_team per pitch via inning_topbot."""
    p = pitches.copy()
    top = p["inning_topbot"] == "Top"
    # Top of inning: away team bats, home team pitches.
    p["batter_team"] = np.where(top, p["away_team"], p["home_team"])
    p["pitcher_team"] = np.where(top, p["home_team"], p["away_team"])
    return p


def _pa_events(p: pd.DataFrame) -> pd.DataFrame:
    """One row per plate-appearance terminator event."""
    pa = p[p["events"].notna()].copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_ab"] = (~pa["events"].isin(NON_AB_EVENTS)).astype(int)
    pa["is_k"] = pa["events"].isin(K_EVENTS).astype(int)
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    return pa


def _batted_balls(p: pd.DataFrame) -> pd.DataFrame:
    """One row per batted ball for launch-angle / exit-velo features."""
    bb = p[p["type"] == "X"].copy()
    # Drop NaN launch metrics before boolean coercion (Statcast sometimes logs
    # a batted ball without launch_angle/speed — bunts, dropped reads).
    bb = bb[bb["launch_angle"].notna() & bb["launch_speed"].notna()].copy()
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32).astype(int)
    bb["is_hard_hit"] = (bb["launch_speed"] >= 95).astype(int)
    bb["is_line_drive"] = (bb["bb_type"] == "line_drive").astype(int)
    return bb


def _primary_team(p: pd.DataFrame, id_col: str, team_col: str) -> pd.DataFrame:
    """Most-appeared team per player (handles mid-season trades)."""
    g = p.groupby([id_col, team_col]).size().reset_index(name="n")
    g = g.sort_values([id_col, "n"], ascending=[True, False])
    return g.drop_duplicates(subset=[id_col]).rename(
        columns={id_col: "mlbam_id", team_col: "team"}
    )[["mlbam_id", "team"]]


def _primary_hand(p: pd.DataFrame, id_col: str, hand_col: str) -> pd.DataFrame:
    """Most-used handedness per player (switch-hitters get their majority side,
    which is fine since we also carry the explicit L/R platoon splits)."""
    sub = p[[id_col, hand_col]].dropna()
    if sub.empty:
        return pd.DataFrame({"mlbam_id": [], "hand": []})
    g = sub.groupby([id_col, hand_col]).size().reset_index(name="n")
    g = g.sort_values([id_col, "n"], ascending=[True, False])
    return g.drop_duplicates(subset=[id_col]).rename(
        columns={id_col: "mlbam_id", hand_col: "hand"}
    )[["mlbam_id", "hand"]]


def _platoon_split_batter(pa: pd.DataFrame) -> pd.DataFrame:
    """For each batter, compute xBA and K% split by opposing pitcher hand."""
    rows = []
    for hand, sub in pa.groupby("p_throws"):
        agg = sub.groupby("batter").agg(
            PA=("events", "size"),
            AB=("is_ab", "sum"),
            H=("is_hit", "sum"),
            K=("is_k", "sum"),
            xba=("estimated_ba_using_speedangle", "mean"),
        ).reset_index()
        agg["ba"] = agg["H"] / agg["AB"].replace(0, pd.NA)
        agg["k_pct"] = agg["K"] / agg["PA"]
        suffix = f"_vs_{hand}"
        agg = agg.rename(columns={
            "PA": f"PA{suffix}",
            "AB": f"AB{suffix}",
            "H": f"H{suffix}",
            "K": f"K{suffix}",
            "xba": f"xba{suffix}",
            "ba": f"ba{suffix}",
            "k_pct": f"k_pct{suffix}",
        }).rename(columns={"batter": "mlbam_id"})
        rows.append(agg)
    if not rows:
        return pd.DataFrame({"mlbam_id": []})
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="mlbam_id", how="outer")
    return out


def _platoon_split_pitcher(pa: pd.DataFrame) -> pd.DataFrame:
    """For each pitcher, compute xBA-allowed and K% split by opposing batter hand."""
    rows = []
    for hand, sub in pa.groupby("stand"):
        agg = sub.groupby("pitcher").agg(
            TBF=("events", "size"),
            H=("is_hit", "sum"),
            K=("is_k", "sum"),
            xba_allowed=("estimated_ba_using_speedangle", "mean"),
        ).reset_index()
        agg["k_pct_allowed"] = agg["K"] / agg["TBF"]
        suffix = f"_vs_{hand}"
        agg = agg.rename(columns={
            "TBF": f"TBF{suffix}",
            "H": f"H{suffix}",
            "K": f"K{suffix}",
            "xba_allowed": f"xba_allowed{suffix}",
            "k_pct_allowed": f"k_pct_allowed{suffix}",
        }).rename(columns={"pitcher": "mlbam_id"})
        rows.append(agg)
    if not rows:
        return pd.DataFrame({"mlbam_id": []})
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="mlbam_id", how="outer")
    return out


def _pitcher_zone_and_contact(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher Zone% and Contact% computed at the pitch level.

    - Zone%  = pct of pitches with zone in [1..9] (strike zone).
    - Contact% = pct of swings that made contact (i.e. not a whiff).
    """
    p = pitches.copy()
    # Zone: Statcast encodes zones 1-9 as in-zone, 11-14 as shadow/chase. Rows
    # without a zone value (very few) are dropped from the denominator.
    has_zone = p["zone"].notna()
    p.loc[has_zone, "is_in_zone"] = p.loc[has_zone, "zone"].between(1, 9).astype(int)

    # Swing/whiff classification from the pitch description.
    p["is_swing"] = p["description"].isin(SWING_DESCRIPTIONS).astype(int)
    p["is_whiff"] = p["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    agg = p.groupby("pitcher").agg(
        pitches_total=("description", "size"),
        in_zone=("is_in_zone", "sum"),
        zone_denom=("is_in_zone", "count"),  # count() ignores NaN
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
    ).reset_index()
    agg["zone_pct"] = agg["in_zone"] / agg["zone_denom"].replace(0, pd.NA)
    agg["contact_pct_allowed"] = 1 - agg["whiffs"] / agg["swings"].replace(0, pd.NA)
    agg = agg.rename(columns={"pitcher": "mlbam_id"})[
        ["mlbam_id", "zone_pct", "contact_pct_allowed"]
    ]
    return agg


def batter_stats(year: int, player_map: pd.DataFrame) -> pd.DataFrame:
    pitches = _annotate_teams(_pull_pitches(year))
    pa = _pa_events(pitches)
    bb = _batted_balls(pitches)

    agg = pa.groupby("batter").agg(
        PA=("events", "size"),
        AB=("is_ab", "sum"),
        H=("is_hit", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        xba=("estimated_ba_using_speedangle", "mean"),
    ).reset_index()

    bb_agg = bb.groupby("batter").agg(
        sweet_spot_pct=("is_sweet_spot", "mean"),
        line_drive_pct=("is_line_drive", "mean"),
        hard_hit_pct=("is_hard_hit", "mean"),
        mean_launch_angle=("launch_angle", "mean"),
        mean_exit_velocity=("launch_speed", "mean"),
    ).reset_index()

    out = agg.merge(bb_agg, on="batter", how="left")
    out["ba"] = out["H"] / out["AB"].replace(0, pd.NA)
    out["k_pct"] = out["K"] / out["PA"]
    out["bb_pct"] = out["BB"] / out["PA"]
    out = out[out["AB"] >= MIN_AB].reset_index(drop=True)
    out["season"] = year
    out = out.rename(columns={"batter": "mlbam_id"})

    # Platoon splits vs LHP / RHP.
    split = _platoon_split_batter(pa)
    out = out.merge(split, on="mlbam_id", how="left")

    # Primary handedness (mode of stand).
    hand = _primary_hand(pitches, "batter", "stand").rename(
        columns={"hand": "batter_hand"}
    )
    out = out.merge(hand, on="mlbam_id", how="left")

    teams = _primary_team(pitches, "batter", "batter_team")
    out = out.merge(teams, on="mlbam_id", how="left")
    out = out.merge(
        player_map[["mlbam_id", "player_name"]], on="mlbam_id", how="left"
    )
    return out


def pitcher_stats(year: int, player_map: pd.DataFrame) -> pd.DataFrame:
    pitches = _annotate_teams(_pull_pitches(year))
    pa = _pa_events(pitches)
    bb = _batted_balls(pitches)

    agg = pa.groupby("pitcher").agg(
        TBF=("events", "size"),
        H_allowed=("is_hit", "sum"),
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        xba_allowed=("estimated_ba_using_speedangle", "mean"),
    ).reset_index()

    bb_agg = bb.groupby("pitcher").agg(
        sweet_spot_pct_allowed=("is_sweet_spot", "mean"),
        hard_hit_pct_allowed=("is_hard_hit", "mean"),
        line_drive_pct_allowed=("is_line_drive", "mean"),
    ).reset_index()

    out = agg.merge(bb_agg, on="pitcher", how="left")
    out["k_pct_allowed"] = out["K"] / out["TBF"]
    out["bb_pct_allowed"] = out["BB"] / out["TBF"]
    out = out[out["TBF"] >= MIN_BF].reset_index(drop=True)
    out["season"] = year
    out = out.rename(columns={"pitcher": "mlbam_id"})

    # Estimate IP as TBF / 4.3 (league-average). Only used for a rough PA-cap
    # sanity filter downstream — not a model feature.
    out["IP"] = (out["TBF"] / 4.3).round(1)

    # Platoon splits vs LHB / RHB.
    split = _platoon_split_pitcher(pa)
    out = out.merge(split, on="mlbam_id", how="left")

    # Zone% and Contact%-allowed.
    zc = _pitcher_zone_and_contact(pitches)
    out = out.merge(zc, on="mlbam_id", how="left")

    # Primary handedness (mode of p_throws).
    hand = _primary_hand(pitches, "pitcher", "p_throws").rename(
        columns={"hand": "pitcher_hand"}
    )
    out = out.merge(hand, on="mlbam_id", how="left")

    # Role: pitchers with 5+ game starts are SP, else RP.
    starts = pitches[(pitches["inning"] == 1) & (pitches["pitch_number"] == 1)]
    gs = starts.groupby("pitcher")["game_pk"].nunique().reset_index(name="GS")
    gs = gs.rename(columns={"pitcher": "mlbam_id"})
    out = out.merge(gs, on="mlbam_id", how="left").fillna({"GS": 0})
    out["role"] = out["GS"].apply(lambda g: "SP" if g >= 5 else "RP")

    teams = _primary_team(pitches, "pitcher", "pitcher_team")
    out = out.merge(teams, on="mlbam_id", how="left")
    out = out.merge(
        player_map[["mlbam_id", "player_name"]], on="mlbam_id", how="left"
    )
    out = out.rename(columns={"player_name": "pitcher_name"})
    return out


def save_all(seasons: list[int]) -> None:
    player_map_path = clean_path("players.parquet")
    if not player_map_path.exists():
        raise FileNotFoundError(
            f"players.parquet not found at {player_map_path}. "
            f"Run: python -m mlbhit.pipeline.build_player_map"
        )
    player_map = pd.read_parquet(player_map_path)[["mlbam_id", "player_name"]]
    player_map["mlbam_id"] = player_map["mlbam_id"].astype("int64")

    bat_frames = []
    pit_frames = []
    for y in seasons:
        print(f"[{y}] building batter stats...")
        bat_frames.append(batter_stats(y, player_map))
        print(f"[{y}] building pitcher stats...")
        pit_frames.append(pitcher_stats(y, player_map))

    bat = pd.concat(bat_frames, ignore_index=True)
    pit = pd.concat(pit_frames, ignore_index=True)

    bat_path = clean_path("batter_season_stats.parquet")
    pit_path = clean_path("pitcher_season_stats.parquet")
    bat.to_parquet(bat_path, index=False)
    pit.to_parquet(pit_path, index=False)

    print("-" * 60)
    print(f"batter rows : {len(bat):>6}   -> {bat_path}")
    print(f"pitcher rows: {len(pit):>6}   -> {pit_path}")
    print()
    print("per-season batter counts:")
    print(bat.groupby("season").size().to_string())
    print()
    print("per-season pitcher counts:")
    print(pit.groupby("season").size().to_string())
    print()
    print(f"batter cols : {sorted(bat.columns.tolist())}")
    print(f"pitcher cols: {sorted(pit.columns.tolist())}")

    assert "team" in bat.columns, "BUG: batter output missing team"
    assert "team" in pit.columns, "BUG: pitcher output missing team"
    assert "TBF" in pit.columns, "BUG: pitcher output missing TBF"
    assert "player_name" in bat.columns, "BUG: batter missing player_name"
    assert "pitcher_name" in pit.columns, "BUG: pitcher missing pitcher_name"
    assert "batter_hand" in bat.columns, "BUG: batter missing batter_hand"
    assert "pitcher_hand" in pit.columns, "BUG: pitcher missing pitcher_hand"
    assert "zone_pct" in pit.columns, "BUG: pitcher missing zone_pct"
    assert "contact_pct_allowed" in pit.columns, "BUG: pitcher missing contact_pct_allowed"
    assert "xba_vs_L" in bat.columns, "BUG: batter missing platoon split xba_vs_L"
    assert "xba_allowed_vs_L" in pit.columns, "BUG: pitcher missing platoon split xba_allowed_vs_L"
    print("OK: all required columns present")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025, 2026],
        help="Seasons to (re)build stats for.",
    )
    args = parser.parse_args()
    save_all(args.seasons)
