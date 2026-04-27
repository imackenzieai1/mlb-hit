from __future__ import annotations

import pandas as pd

from ..io import clean_path
from .batter import regress


def build_pitcher_features(season: int) -> pd.DataFrame:
    pit = pd.read_parquet(clean_path("pitcher_season_stats.parquet"))
    pit = pit[pit["season"] == season].copy()
    players = pd.read_parquet(clean_path("players.parquet"))[["fg_id", "mlbam_id"]]
    if "fg_id" in pit.columns:
        pit = pit.merge(players, on="fg_id", how="inner")
    else:
        pit = pit.merge(players, on="mlbam_id", how="left")

    la_path = clean_path(f"pitcher_la_{season}.parquet")
    if la_path.exists():
        la = pd.read_parquet(la_path)
        pit = pit.merge(la, on=["mlbam_id", "season"], how="left")

    # Regress platoon splits toward league means with TBF_vs_hand as denominator.
    # Smaller n_prior (60/80) than season-overall — split samples are smaller,
    # so the prior should pull less aggressively.
    for hand in ("L", "R"):
        xba_col = f"xba_allowed_vs_{hand}"
        k_col = f"k_pct_allowed_vs_{hand}"
        tbf_col = f"TBF_vs_{hand}"
        if xba_col in pit.columns and pit[xba_col].notna().any():
            league = pit[xba_col].median()
            tbf_series = pit[tbf_col].fillna(0) if tbf_col in pit.columns else pd.Series(0, index=pit.index)
            pit[f"sp_xba_allowed_vs_{hand}"] = regress(pit[xba_col].fillna(league),
                                                       tbf_series, league, n_prior=60)
        if k_col in pit.columns and pit[k_col].notna().any():
            league_k = pit[k_col].median()
            tbf_series = pit[tbf_col].fillna(0) if tbf_col in pit.columns else pd.Series(0, index=pit.index)
            pit[f"sp_k_pct_allowed_vs_{hand}"] = regress(pit[k_col].fillna(league_k),
                                                         tbf_series, league_k, n_prior=80)

    pit = pit.rename(
        columns={
            "xba_allowed": "sp_xba_allowed",
            "k_pct_allowed": "sp_k_pct",
            "hard_hit_pct_allowed": "sp_hard_hit_allowed",
            "sweet_spot_pct_allowed": "sp_sweet_spot_allowed",
            "zone_pct": "sp_zone_pct",
            "contact_pct_allowed": "sp_contact_pct_allowed",
            "TBF_vs_L": "sp_TBF_vs_L",
            "TBF_vs_R": "sp_TBF_vs_R",
        }
    )
    keep = [
        "mlbam_id",
        "pitcher_name",
        "team",
        "season",
        "role",
        "pitcher_hand",
        "sp_xba_allowed",
        "sp_k_pct",
        "sp_hard_hit_allowed",
        "sp_sweet_spot_allowed",
        "sp_xba_allowed_vs_L",
        "sp_xba_allowed_vs_R",
        "sp_k_pct_allowed_vs_L",
        "sp_k_pct_allowed_vs_R",
        "sp_zone_pct",
        "sp_contact_pct_allowed",
        "sp_TBF_vs_L",
        "sp_TBF_vs_R",
        "IP",
        "TBF",
    ]
    return pit[[c for c in keep if c in pit.columns]]
