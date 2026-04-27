from __future__ import annotations

import pandas as pd
from pybaseball import statcast

from ..io import clean_path, raw_path


def pull_statcast_season(season: int) -> pd.DataFrame:
    start = f"{season}-03-20"
    end = f"{season}-11-05"
    df = statcast(start_dt=start, end_dt=end)
    out = raw_path("statcast", f"pitches_{season}.parquet")
    df.to_parquet(out, index=False)
    return df


def derive_batter_la_features(pitches: pd.DataFrame) -> pd.DataFrame:
    bb = pitches[pitches["type"] == "X"].copy()
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32)
    bb["is_hard_hit"] = bb["launch_speed"] >= 95
    bb["is_line_drive"] = bb["bb_type"] == "line_drive"
    lsa = pd.to_numeric(bb["launch_speed_angle"], errors="coerce")
    bb["is_solid"] = lsa.isin([4, 5, 6])
    agg = bb.groupby(["batter", "game_year"]).agg(
        sweet_spot_pct=("is_sweet_spot", "mean"),
        line_drive_pct=("is_line_drive", "mean"),
        mean_launch_angle=("launch_angle", "mean"),
        mean_exit_velocity=("launch_speed", "mean"),
        hard_hit_pct=("is_hard_hit", "mean"),
        solid_contact_pct=("is_solid", "mean"),
        batted_balls=("is_sweet_spot", "size"),
    ).reset_index()
    agg = agg.rename(columns={"batter": "mlbam_id", "game_year": "season"})
    return agg


def derive_pitcher_la_features(pitches: pd.DataFrame) -> pd.DataFrame:
    bb = pitches[pitches["type"] == "X"].copy()
    bb["is_sweet_spot"] = bb["launch_angle"].between(8, 32)
    bb["is_hard_hit"] = bb["launch_speed"] >= 95
    bb["is_line_drive"] = bb["bb_type"] == "line_drive"
    agg = bb.groupby(["pitcher", "game_year"]).agg(
        sweet_spot_pct_allowed=("is_sweet_spot", "mean"),
        line_drive_pct_allowed=("is_line_drive", "mean"),
        hard_hit_pct_allowed=("is_hard_hit", "mean"),
        batted_balls_allowed=("is_sweet_spot", "size"),
    ).reset_index()
    agg = agg.rename(columns={"pitcher": "mlbam_id", "game_year": "season"})
    return agg


if __name__ == "__main__":
    for s in [2023, 2024]:
        print(f"pulling {s}…")
        pitches = pull_statcast_season(s)
        bat_la = derive_batter_la_features(pitches)
        pit_la = derive_pitcher_la_features(pitches)
        bat_la.to_parquet(clean_path(f"batter_la_{s}.parquet"), index=False)
        pit_la.to_parquet(clean_path(f"pitcher_la_{s}.parquet"), index=False)
        print(f"  {len(bat_la)} batters, {len(pit_la)} pitchers")
