from __future__ import annotations

import pandas as pd

from ..io import clean_path


def build_bullpen_features(seasons: list[int]) -> pd.DataFrame:
    pit = pd.read_parquet(clean_path("pitcher_season_stats.parquet"))
    pit = pit[pit["role"] == "RP"].copy()
    pit = pit[pit["season"].isin(seasons)]
    agg = pit.groupby(["team", "season"]).agg(
        pen_xba_allowed=("xba_allowed", "mean"),
        pen_k_pct=("k_pct_allowed", "mean"),
    ).reset_index()
    agg.to_parquet(clean_path("bullpen_features.parquet"), index=False)
    return agg
