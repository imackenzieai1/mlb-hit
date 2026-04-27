from __future__ import annotations

import pandas as pd
from pybaseball import batting_stats

from ..config import SETTINGS
from ..io import clean_path

MIN_AB = SETTINGS["filters"]["min_batter_ab"]

KEEP = [
    "IDfg",
    "Name",
    "Team",
    "Age",
    "G",
    "AB",
    "PA",
    "H",
    "AVG",
    "OBP",
    "SLG",
    "BB%",
    "K%",
    "Hard%",
    "LD%",
    "GB%",
    "FB%",
    "IFFB%",
    "Contact%",
    "Z-Contact%",
    "O-Swing%",
    "wRC+",
    "Bat",
    "xBA",
    "xwOBA",
    "Barrel%",
    "HardHit%",
    "maxEV",
    "EV",
    "LA",
]


def fetch_batting(season: int) -> pd.DataFrame:
    df = batting_stats(season, qual=0)
    existing = [c for c in KEEP if c in df.columns]
    df = df[existing].copy()
    df = df[df["AB"] >= MIN_AB].reset_index(drop=True)
    df["season"] = season
    df = df.rename(
        columns={
            "IDfg": "fg_id",
            "Name": "player_name",
            "Team": "team",
            "AVG": "ba",
            "OBP": "obp",
            "SLG": "slg",
            "BB%": "bb_pct",
            "K%": "k_pct",
            "Hard%": "hard_pct_fg",
            "LD%": "ld_pct",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
            "Contact%": "contact_pct",
            "Z-Contact%": "z_contact_pct",
            "O-Swing%": "chase_pct",
            "xBA": "xba",
            "xwOBA": "xwoba",
            "Barrel%": "barrel_pct",
            "HardHit%": "hard_hit_pct",
            "maxEV": "max_ev",
            "EV": "mean_ev",
            "LA": "mean_la",
        }
    )
    for c in [
        "bb_pct",
        "k_pct",
        "hard_pct_fg",
        "ld_pct",
        "gb_pct",
        "fb_pct",
        "contact_pct",
        "z_contact_pct",
        "chase_pct",
        "barrel_pct",
        "hard_hit_pct",
    ]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    return df


def save_batting(seasons: list[int]) -> pd.DataFrame:
    frames = [fetch_batting(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(clean_path("batter_season_stats.parquet"), index=False)
    return df


if __name__ == "__main__":
    df = save_batting([2023, 2024])
    print(f"{len(df)} batter-seasons kept after AB>={MIN_AB} filter")
    print(df[["player_name", "team", "season", "AB", "ba", "xba", "hard_hit_pct"]].head())
