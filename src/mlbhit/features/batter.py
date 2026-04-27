from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import clean_path


def regress(x: pd.Series, n: pd.Series, prior: float, n_prior: int = 100) -> pd.Series:
    return (x * n + prior * n_prior) / (n + n_prior)


def build_batter_features(season: int) -> pd.DataFrame:
    bat = pd.read_parquet(clean_path("batter_season_stats.parquet"))
    bat = bat[bat["season"] == season].copy()
    players = pd.read_parquet(clean_path("players.parquet"))[["fg_id", "mlbam_id"]]
    if "fg_id" in bat.columns:
        bat = bat.merge(players, on="fg_id", how="inner")
    else:
        # Statcast fallback (and similar) tables are keyed by MLBAM only.
        bat = bat.merge(players, on="mlbam_id", how="left")

    la_path = clean_path(f"batter_la_{season}.parquet")
    if la_path.exists():
        la = pd.read_parquet(la_path)
        bat = bat.merge(la, on=["mlbam_id", "season"], how="left")
    else:
        # Do NOT wipe Statcast-derived LA cols if they're already present
        # (batter_season_stats.parquet from fetch_stats_from_statcast carries them).
        for col in (
            "sweet_spot_pct",
            "line_drive_pct",
            "mean_launch_angle",
            "mean_exit_velocity",
            "solid_contact_pct",
            "batted_balls",
        ):
            if col not in bat.columns:
                bat[col] = np.nan

    if "xba" in bat.columns and bat["xba"].notna().any():
        league_xba = bat["xba"].median()
        bat["xba_regressed"] = regress(bat["xba"], bat["AB"], league_xba, n_prior=100)
    else:
        bat["xba_regressed"] = bat.get("ba", pd.Series(0.25, index=bat.index))

    if "k_pct" in bat.columns and bat["k_pct"].notna().any():
        bat["k_pct_regressed"] = regress(bat["k_pct"], bat["PA"], bat["k_pct"].median(), n_prior=150)
    else:
        bat["k_pct_regressed"] = 0.22

    # Platoon splits (from fetch_stats_from_statcast). Regress toward league
    # mean using a smaller n_prior than season-overall — fewer split-PAs, so
    # we want a softer prior to avoid washing out real platoon talent.
    # Values are NaN for seasons that pre-date the Statcast platoon rebuild.
    for hand in ("L", "R"):
        xba_col = f"xba_vs_{hand}"
        k_col = f"k_pct_vs_{hand}"
        pa_col = f"PA_vs_{hand}"
        if xba_col in bat.columns and bat[xba_col].notna().any():
            league = bat[xba_col].median()
            pa_series = bat[pa_col].fillna(0) if pa_col in bat.columns else pd.Series(0, index=bat.index)
            bat[f"bat_xba_vs_{hand}"] = regress(bat[xba_col].fillna(league),
                                                pa_series, league, n_prior=60)
        if k_col in bat.columns and bat[k_col].notna().any():
            league_k = bat[k_col].median()
            pa_series = bat[pa_col].fillna(0) if pa_col in bat.columns else pd.Series(0, index=bat.index)
            bat[f"bat_k_pct_vs_{hand}"] = regress(bat[k_col].fillna(league_k),
                                                  pa_series, league_k, n_prior=80)

    bat = bat.rename(
        columns={
            "xba_regressed": "bat_xba_season",
            "k_pct_regressed": "bat_k_pct",
            "hard_hit_pct": "bat_hard_hit_pct",
            "contact_pct": "bat_contact_pct",
            "sweet_spot_pct": "bat_sweet_spot_pct",
            "line_drive_pct": "bat_line_drive_pct",
            "solid_contact_pct": "bat_solid_contact_pct",
            "PA_vs_L": "bat_PA_vs_L",
            "PA_vs_R": "bat_PA_vs_R",
        }
    )
    keep = [
        "mlbam_id",
        "player_name",
        "team",
        "season",
        "batter_hand",
        "bat_xba_season",
        "bat_k_pct",
        "bat_hard_hit_pct",
        "bat_contact_pct",
        "bat_sweet_spot_pct",
        "bat_line_drive_pct",
        "bat_solid_contact_pct",
        "bat_xba_vs_L",
        "bat_xba_vs_R",
        "bat_k_pct_vs_L",
        "bat_k_pct_vs_R",
        "bat_PA_vs_L",
        "bat_PA_vs_R",
        "AB",
        "PA",
    ]
    return bat[[c for c in keep if c in bat.columns]]
