from __future__ import annotations

import pandas as pd
from pybaseball import pitching_stats

from ..config import SETTINGS
from ..io import clean_path

MIN_BF = SETTINGS["filters"]["min_pitcher_bf"]

KEEP = [
    "IDfg",
    "Name",
    "Team",
    "Age",
    "G",
    "GS",
    "IP",
    "TBF",
    "ERA",
    "FIP",
    "xFIP",
    "K%",
    "BB%",
    "WHIP",
    "AVG",
    "GB%",
    "LD%",
    "HR/9",
    "HardHit%",
    "Barrel%",
    "xBA",
    "xwOBA",
    "LA",
    "EV",
    "vFA (pi)",
    "vFA (sc)",
]


def fetch_pitching(season: int) -> pd.DataFrame:
    df = pitching_stats(season, qual=0)
    existing = [c for c in KEEP if c in df.columns]
    df = df[existing].copy()
    df = df[df["TBF"] >= MIN_BF].reset_index(drop=True)
    df["season"] = season
    df = df.rename(
        columns={
            "IDfg": "fg_id",
            "Name": "pitcher_name",
            "Team": "team",
            "K%": "k_pct_allowed",
            "BB%": "bb_pct_allowed",
            "AVG": "ba_allowed",
            "GB%": "gb_pct_allowed",
            "LD%": "ld_pct_allowed",
            "HardHit%": "hard_hit_pct_allowed",
            "Barrel%": "barrel_pct_allowed",
            "xBA": "xba_allowed",
            "xwOBA": "xwoba_allowed",
        }
    )
    for c in [
        "k_pct_allowed",
        "bb_pct_allowed",
        "gb_pct_allowed",
        "ld_pct_allowed",
        "hard_hit_pct_allowed",
        "barrel_pct_allowed",
    ]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    df["role"] = (df["GS"] / df["G"].clip(lower=1)).apply(lambda r: "SP" if r >= 0.6 else "RP")
    return df


def save_pitching(seasons: list[int]) -> pd.DataFrame:
    frames = [fetch_pitching(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(clean_path("pitcher_season_stats.parquet"), index=False)
    return df


if __name__ == "__main__":
    df = save_pitching([2023, 2024])
    print(f"{len(df)} pitcher-seasons kept after BF>={MIN_BF} filter")
    print(df[["pitcher_name", "team", "season", "TBF", "xba_allowed", "k_pct_allowed", "role"]].head())
