"""Build a permanent audit log of (predictions x odds x outcomes) per day.

Why: we can't backtest against historical book odds (they're not available for
free, and paid archives rarely cover player props). So the alternative is to
log forward — every day we snapshot DK+FD prices + model predictions, and once
games finish we reconcile with actuals. After ~30 days you have a real ROI
backtest against the exact two books you'd have placed at.

Output: data/outputs/archive/YYYY-MM-DD_archive.parquet
Schema: date, player_id, player_name, book, over_price, under_price,
        p_model, fair_american, implied_prob_over, edge_over, edge_under,
        got_hit (populated later by reconcile_outcomes), fetched_at

Usage:
    # Morning: right after run_daily.py finishes, snapshot preds + odds
    python -m mlbhit.pipeline.archive_daily snapshot

    # Night: after games end, join in actual hits from that day's boxscores
    python -m mlbhit.pipeline.archive_daily reconcile --date 2026-04-24

    # Aggregate: compute rolling ROI across every archived day
    python -m mlbhit.pipeline.archive_daily report
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..io import clean_path, output_path, raw_path

# Directory containing all archived daily parquets. output_path() requires
# both a kind and a filename, so for directory-level operations (like report())
# we build the folder path directly from the same data_dir config.
_ARCHIVE_DIR = SETTINGS["paths"]["data_dir"] / "output" / "archive"
from ..utils.odds_math import american_to_prob, ev_per_unit


def _archive_path(d: date) -> Path:
    return output_path("archive", f"{d.isoformat()}_archive.parquet")


def snapshot(target: date) -> pd.DataFrame:
    """Morning step: join today's predictions with today's odds, save."""
    preds_path = output_path("predictions", f"{target.isoformat()}.parquet")
    odds_path = raw_path("props", f"{target.isoformat()}_props.parquet")

    if not preds_path.exists():
        raise FileNotFoundError(
            f"{preds_path} missing. Run run_daily.py first."
        )
    if not odds_path.exists():
        raise FileNotFoundError(
            f"{odds_path} missing. Run fetch_prop_odds (theodds or csv) first."
        )

    preds = pd.read_parquet(preds_path)
    odds = pd.read_parquet(odds_path)

    # Both player_id columns should be Int64 — enforce to be safe.
    preds["player_id"] = preds["player_id"].astype("Int64")
    odds["player_id"] = odds["player_id"].astype("Int64")

    # One row per (player, book) — a player usually has a DK price AND a FD price.
    m = odds.merge(
        preds[["player_id", "p_model", "fair_american"]],
        on="player_id", how="inner",
    )

    m["implied_prob_over"] = m["over_price"].apply(
        lambda px: american_to_prob(int(px)) if pd.notna(px) else np.nan
    )
    m["edge_over"] = m.apply(
        lambda r: ev_per_unit(r["p_model"], int(r["over_price"]))
        if pd.notna(r["over_price"]) else np.nan,
        axis=1,
    )
    m["edge_under"] = m.apply(
        lambda r: ev_per_unit(1 - r["p_model"], int(r["under_price"]))
        if pd.notna(r["under_price"]) else np.nan,
        axis=1,
    )
    m["got_hit"] = pd.NA  # reconciled later
    m["date"] = target.isoformat()

    out = _archive_path(target)
    m.to_parquet(out, index=False)
    print(f"snapshot: {len(m):,} rows written -> {out.name}")
    print(f"  books: {sorted(m['book'].unique())}")
    print(f"  edge_over >= 5%: {(m['edge_over'] >= 0.05).sum()} picks")
    return m


def reconcile(target: date) -> pd.DataFrame:
    """Night step: fill in got_hit from that day's boxscores."""
    arc_path = _archive_path(target)
    if not arc_path.exists():
        raise FileNotFoundError(f"{arc_path} missing — nothing to reconcile.")

    # Use the per-day boxscore cache that fetch_boxscores drops.
    day_box_path = clean_path("daily_boxscores") / f"{target.isoformat()}.parquet"
    if not day_box_path.exists():
        # Fall back to the season parquet
        season_box_path = clean_path(f"boxscores_{target.year}.parquet")
        if not season_box_path.exists():
            raise FileNotFoundError(
                f"Neither {day_box_path.name} nor {season_box_path.name} exists. "
                "Run fetch_boxscores for this date first."
            )
        box = pd.read_parquet(season_box_path)
        box = box[box["date"] == target.isoformat()]
    else:
        box = pd.read_parquet(day_box_path)

    if box.empty:
        print(f"WARN: no boxscore rows for {target} — games may not be final yet.")
        return pd.read_parquet(arc_path)

    # Collapse to (player_id, got_hit) — if a player appears twice (doubleheader),
    # count as hit if they got one in either game.
    outcomes = (
        box.groupby("player_id")["got_hit"]
        .max()
        .reset_index()
        .rename(columns={"player_id": "mlbam_id", "got_hit": "actual_hit"})
    )
    outcomes["mlbam_id"] = outcomes["mlbam_id"].astype("Int64")

    arc = pd.read_parquet(arc_path)
    arc["player_id"] = arc["player_id"].astype("Int64")
    arc = arc.drop(columns=["got_hit"], errors="ignore").merge(
        outcomes, left_on="player_id", right_on="mlbam_id", how="left"
    ).drop(columns=["mlbam_id"]).rename(columns={"actual_hit": "got_hit"})

    n_matched = arc["got_hit"].notna().sum()
    print(f"reconcile: {n_matched:,}/{len(arc):,} archive rows have outcomes "
          f"({n_matched/len(arc):.1%})")
    arc.to_parquet(arc_path, index=False)
    return arc


def report(stake: float = 1.0, edge_min: float = 0.05) -> None:
    """Aggregate ROI across every archived+reconciled day.

    Policy simulated: flat `stake` per OVER whenever edge_over >= edge_min,
    using the DK price (line-shop later if you want; DK was your default bet).
    """
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(_ARCHIVE_DIR.glob("*_archive.parquet"))
    if not paths:
        print(f"No archive files in {_ARCHIVE_DIR}. Run `snapshot` daily first.")
        return
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    # Pick one book per (date, player) — prefer DK, fall back to FD.
    dk = df[df["book"] == "draftkings"]
    fd = df[df["book"] == "fanduel"]
    picks = pd.concat([
        dk,
        fd[~fd.set_index(["date", "player_id"]).index.isin(
            dk.set_index(["date", "player_id"]).index
        )],
    ])

    bets = picks[
        (picks["edge_over"] >= edge_min) & picks["got_hit"].notna()
    ].copy()
    if bets.empty:
        print("No reconciled +EV bets yet. Need at least 1 day of data with "
              "both odds and outcomes.")
        return

    # Payout per $stake bet:
    #   win:  stake * (decimal_odds - 1)
    #   loss: -stake
    def _decimal(px: int) -> float:
        return 100 / abs(px) + 1 if px < 0 else px / 100 + 1

    bets["decimal"] = bets["over_price"].astype(int).apply(_decimal)
    bets["pnl"] = np.where(
        bets["got_hit"].astype(int) == 1,
        stake * (bets["decimal"] - 1),
        -stake,
    )

    n_bets = len(bets)
    hit_rate = bets["got_hit"].astype(int).mean()
    total_pnl = bets["pnl"].sum()
    total_staked = stake * n_bets
    roi = total_pnl / total_staked
    avg_price = bets["over_price"].astype(int).mean()

    print(f"=" * 60)
    print(f"FORWARD BACKTEST ({paths[0].stem[:10]} -> {paths[-1].stem[:10]})")
    print(f"=" * 60)
    print(f"  bets placed       {n_bets}")
    print(f"  hit rate          {hit_rate:.1%}")
    print(f"  avg price         {avg_price:+.0f}")
    print(f"  total staked      ${total_staked:.2f}")
    print(f"  total P&L         ${total_pnl:+.2f}")
    print(f"  ROI               {roi:+.1%}")
    print(f"  edge threshold    {edge_min:.0%}")
    by_day = bets.groupby("date").agg(
        bets=("pnl", "size"),
        hit_rate=("got_hit", "mean"),
        pnl=("pnl", "sum"),
    )
    print()
    print(by_day.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["snapshot", "reconcile", "report"])
    parser.add_argument("--date", type=str, default=None,
                        help="YYYY-MM-DD (default: today for snapshot, yesterday for reconcile)")
    parser.add_argument("--stake", type=float, default=1.0,
                        help="Dollar stake per bet for report (default $1).")
    parser.add_argument("--edge-min", type=float, default=0.05,
                        help="Minimum edge_over threshold (default 0.05).")
    args = parser.parse_args()

    if args.cmd == "snapshot":
        target = date.fromisoformat(args.date) if args.date else date.today()
        snapshot(target)
    elif args.cmd == "reconcile":
        # Default: yesterday (so games are final and boxscores exist)
        default = date.today() - timedelta(days=1)
        target = date.fromisoformat(args.date) if args.date else default
        reconcile(target)
    elif args.cmd == "report":
        report(stake=args.stake, edge_min=args.edge_min)
