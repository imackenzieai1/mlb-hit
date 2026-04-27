from __future__ import annotations

import pandas as pd
from pybaseball import chadwick_register

from ..io import clean_path


def build() -> pd.DataFrame:
    reg = chadwick_register()
    cols = [
        "key_mlbam",
        "key_fangraphs",
        "name_first",
        "name_last",
        "mlb_played_first",
        "mlb_played_last",
    ]
    cols = [c for c in cols if c in reg.columns]
    df = reg[cols].dropna(subset=["key_mlbam"])
    df["mlb_played_last"] = pd.to_numeric(df["mlb_played_last"], errors="coerce").fillna(0)
    df = df[df["mlb_played_last"] >= 2020]
    df["player_name"] = df["name_first"].str.strip() + " " + df["name_last"].str.strip()
    df["fg_id"] = pd.to_numeric(df["key_fangraphs"], errors="coerce")
    df = df.dropna(subset=["fg_id"])
    df["fg_id"] = df["fg_id"].astype(int)
    df = df.rename(columns={"key_mlbam": "mlbam_id"})
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    df = df.sort_values("mlb_played_last", ascending=False).drop_duplicates(subset=["fg_id"], keep="first")
    df.to_parquet(clean_path("players.parquet"), index=False)
    return df


if __name__ == "__main__":
    out = build()
    print(f"{len(out)} players mapped")
