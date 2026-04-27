"""Cohort analysis on the Filter E backtest universe.

Question we're answering: do the existing "confidence" flags
(pitcher_features_known, pitcher_low_sample) split bets into ROI cohorts
worth filtering on, BEFORE we go build a new feature?

Reuses historical_backtest.backtest()'s exact join + filter logic so
cohort sums = the 539 bets we already trust. Just adds groupby breakdowns
on top of the resulting bets dataframe.

Usage:
    python -m scripts.cohort_analysis --start 2026-03-20 --end 2026-04-23
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from mlbhit.io import modeling_path
from mlbhit.pipeline.historical_backtest import backtest


def _attach_modeling_flags(bets: pd.DataFrame) -> pd.DataFrame:
    """Pull pitcher_low_sample and rolling-vs-season disagreement onto the
    bets frame so we can cohort on them.

    The backtest's _score_historical_features only surfaces a small
    `keep` list — pitcher_low_sample and the rolling/season columns
    aren't in there — so we re-merge from the modeling parquet.
    """
    feat = pd.read_parquet(modeling_path("player_game_features.parquet"))
    feat = feat[["date", "player_id", "pitcher_low_sample",
                 "bat_xba_season", "xba_30d", "PA_30d"]].copy()
    feat["player_id"] = feat["player_id"].astype("Int64")
    feat["date"] = feat["date"].astype(str)
    # Doubleheader wart: modeling parquet has 2 rows per player on DH days.
    # Keep first to match backtest's drop_duplicates behaviour.
    feat = feat.drop_duplicates(subset=["date", "player_id"], keep="first")
    bets = bets.copy()
    bets["date"] = bets["date"].astype(str)
    bets["player_id"] = bets["player_id"].astype("Int64")
    out = bets.merge(feat, on=["date", "player_id"], how="left")

    # pitcher_features_known was computed in backtest from pitcher_hand.notna().
    # Surface it directly from the bets frame's pitcher_hand column.
    if "pitcher_hand" in out.columns:
        out["pitcher_features_known"] = out["pitcher_hand"].notna().astype(int)
    else:
        out["pitcher_features_known"] = 0

    # Rolling-vs-season disagreement on xBA (excludes truly tiny samples).
    out["rolling_disagreement"] = (out["xba_30d"] - out["bat_xba_season"]).abs()
    return out


def _print_cohort(label: str, bets: pd.DataFrame, mask: pd.Series, stake: float = 1.0) -> None:
    sub = bets[mask]
    n = len(sub)
    if n == 0:
        print(f"  {label:35s}    n=0  (empty cohort)")
        return
    hit = sub["got_hit"].astype(int).mean()
    pnl = sub["pnl"].sum()
    roi = pnl / (stake * n)
    avg_p = sub["p_model"].mean()
    avg_edge = sub["edge_over"].mean()
    avg_price = sub["over_price"].astype(int).mean()
    print(f"  {label:35s}    n={n:4d}  hit={hit:5.1%}  ROI={roi:+6.1%}  "
          f"P&L={pnl:+7.2f}  avg_p={avg_p:.3f}  avg_edge={avg_edge:+.1%}  avg_px={avg_price:+.0f}")


def cohort_analysis(start: date, end: date, stake: float = 1.0) -> None:
    print("Re-running Filter E backtest to get the bet universe...")
    bets = backtest(
        start=start, end=end, stake=stake,
        filter_e=True, require_pitcher=True,
    )
    if bets is None or bets.empty:
        print("No bets — nothing to analyze.")
        return

    bets = _attach_modeling_flags(bets)

    print()
    print("=" * 100)
    print(f"COHORT ANALYSIS  {start} -> {end}  (Filter E + require-pitcher universe)")
    print("=" * 100)

    # ---- Cohort 1: pitcher_features_known ----
    print()
    print("[1] pitcher_features_known  (was the pitcher hand actually known, vs league-mean fallback)")
    _print_cohort("pitcher_features_known = 1", bets, bets["pitcher_features_known"] == 1)
    _print_cohort("pitcher_features_known = 0", bets, bets["pitcher_features_known"] == 0)

    # ---- Cohort 2: pitcher_low_sample ----
    print()
    print("[2] pitcher_low_sample  (does the SP have <30 PA recently → noisy season stat)")
    _print_cohort("pitcher_low_sample = 0", bets, bets["pitcher_low_sample"] == 0)
    _print_cohort("pitcher_low_sample = 1", bets, bets["pitcher_low_sample"] == 1)

    # ---- Cohort 3: rolling_disagreement (continuous → bucketed) ----
    print()
    print("[3] rolling_disagreement = |xba_30d - bat_xba_season|  "
          "(higher = noisier model regime)")
    rd = bets["rolling_disagreement"]
    if rd.notna().any():
        b1 = rd < 0.025
        b2 = (rd >= 0.025) & (rd < 0.050)
        b3 = (rd >= 0.050) & (rd < 0.100)
        b4 = rd >= 0.100
        _print_cohort("disagreement < 0.025  (tight)",   bets, b1)
        _print_cohort("disagreement 0.025-0.050",        bets, b2)
        _print_cohort("disagreement 0.050-0.100",        bets, b3)
        _print_cohort("disagreement >= 0.100  (noisy)",  bets, b4)
    else:
        print("  no rolling_disagreement data — skipping")

    # ---- Cohort 4: combined pitcher confidence (both flags green) ----
    print()
    print("[4] Combined: pitcher_features_known=1 AND pitcher_low_sample=0  "
          "(both pitcher confidence checks pass)")
    both_green = (bets["pitcher_features_known"] == 1) & (bets["pitcher_low_sample"] == 0)
    _print_cohort("both green",   bets, both_green)
    _print_cohort("at least one red", bets, ~both_green)

    print()
    print("=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print("  - If a cohort has materially worse ROI than the +15.5% baseline, it's a")
    print("    candidate filter. 'Materially worse' = 5+ pp ROI gap on n>=30 bets.")
    print("  - If both cohorts are in the +12% to +18% band, the flag isn't predictive")
    print("    on top of Filter E and the model is already absorbing it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2026-03-20")
    p.add_argument("--end", type=str, default="2026-04-23")
    p.add_argument("--stake", type=float, default=1.0)
    args = p.parse_args()

    cohort_analysis(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        stake=args.stake,
    )
