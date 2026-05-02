#!/usr/bin/env python
"""Sweep Filter E thresholds over a backtest window to compare bet volume,
hit rate, ROI, and worst-day drawdown across the grid.

Use this when you want to tighten production thresholds. The default grid
covers a "loose vs tight" range; pass --edges and --prices to focus the
search. The current-production combo (xgb_v5_recal at 11.4%/-240) is
flagged in the output so you can see how alternatives compare.

Usage:
    python scripts/sweep_filter_e.py --start 2026-04-22 --end 2026-04-28
    python scripts/sweep_filter_e.py --start 2026-04-22 --end 2026-04-28 \\
        --edges 0.114 0.15 0.18 --prices -240 -200 -180

Output is one row per (edge, price) combination, sorted by weighted ROI desc
so the best-looking config floats to the top. Sharpe (daily P&L mean / std)
is the variance-aware tiebreaker — a 25% ROI on 5 wild days is worse than
a 20% ROI on 7 steady days, and Sharpe captures that.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlbhit.pipeline.historical_backtest import backtest  # noqa: E402

# Default grid. Loose-to-tight on edge, chalk-to-flat on price.
EDGE_FLOORS_DEFAULT = [0.114, 0.13, 0.15, 0.17, 0.20]
PRICE_FLOORS_DEFAULT = [-240, -220, -200, -180]

# Currently in production — flagged in output for reference.
PROD_EDGE = 0.11445569939746027
PROD_PRICE = -240


def summarize(bets: pd.DataFrame, stake: float = 1.0) -> dict:
    """Compute a one-line summary of a backtest run."""
    if bets.empty:
        return {
            "n_bets": 0, "hit_rate": 0.0, "roi": 0.0, "roi_weighted": 0.0,
            "pnl_total": 0.0, "pnl_weighted_total": 0.0,
            "worst_day": 0.0, "sharpe": 0.0, "n_days": 0, "bets_per_day": 0.0,
        }
    n = len(bets)
    hit_rate = float(bets["got_hit"].astype(int).mean())
    pnl_total = float(bets["pnl"].sum())
    roi = pnl_total / (stake * n)

    # Weighted P&L uses the 2x hot-streak sizing the production system actually
    # banks. Falls back to flat if the column isn't present (older backtest
    # runs without recent_form attached).
    if "pnl_weighted" in bets.columns and "recommended_units" in bets.columns:
        units = float((stake * bets["recommended_units"]).sum())
        pnl_w = float(bets["pnl_weighted"].sum())
        roi_w = pnl_w / units if units else 0.0
    else:
        pnl_w = pnl_total
        roi_w = roi

    daily = bets.groupby("date")["pnl"].sum().sort_index()
    n_days = int(len(daily))
    worst = float(daily.min()) if n_days else 0.0
    if n_days >= 2 and daily.std(ddof=0) > 0:
        sharpe = float(daily.mean()) / float(daily.std(ddof=0))
    else:
        sharpe = 0.0

    return {
        "n_bets": n,
        "hit_rate": hit_rate,
        "roi": roi,
        "roi_weighted": roi_w,
        "pnl_total": pnl_total,
        "pnl_weighted_total": pnl_w,
        "worst_day": worst,
        "sharpe": sharpe,
        "n_days": n_days,
        "bets_per_day": n / max(1, n_days),
    }


def run_one(start: date, end: date, edge_min: float, price_min: int,
            model: str | None) -> dict:
    """Run a single backtest config silently and return its summary."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bets = backtest(
            start, end,
            edge_min=edge_min,
            stake=1.0,
            model_name=model,
            filter_e=True,
            require_pitcher=True,
            price_max_negative=price_min,
        )
    return summarize(bets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Filter E thresholds across a backtest window.",
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--model", default=None,
        help="Model name (defaults to predict.DEFAULT_MODEL).",
    )
    parser.add_argument(
        "--edges", nargs="+", type=float, default=None,
        help="Edge floors to test (e.g. 0.114 0.15 0.18). "
             "Default: 0.114, 0.13, 0.15, 0.17, 0.20.",
    )
    parser.add_argument(
        "--prices", nargs="+", type=int, default=None,
        help="Price floors to test (e.g. -240 -200 -180). "
             "Default: -240, -220, -200, -180.",
    )
    parser.add_argument(
        "--sort-by", default="roi_weighted",
        choices=["roi_weighted", "roi", "sharpe", "pnl_weighted_total",
                 "n_bets", "hit_rate"],
        help="Sort the output table by this metric (descending).",
    )
    args = parser.parse_args()

    edges = args.edges or EDGE_FLOORS_DEFAULT
    prices = args.prices or PRICE_FLOORS_DEFAULT

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"\nFilter E threshold sweep: {start} -> {end}")
    print(f"  model: {args.model or 'default (predict.DEFAULT_MODEL)'}")
    print(f"  grid:  {len(edges)} edges x {len(prices)} prices = {len(edges)*len(prices)} runs")
    print(f"  sizing: 2x on hot bats (>.300/6g), 1x otherwise")
    print()

    rows = []
    for edge in edges:
        for price in prices:
            r = run_one(start, end, edge, price, args.model)
            r["edge_min"] = edge
            r["price_min"] = price
            rows.append(r)

    # Sort: best of the chosen metric at the top.
    rows.sort(key=lambda r: r[args.sort_by], reverse=True)

    # Header.
    header = (
        f"{'edge≥':>7}  {'price≥':>7}  {'bets':>5}  {'/day':>5}  "
        f"{'hit':>6}  {'ROI':>7}  {'wROI':>7}  "
        f"{'P&L':>9}  {'wP&L':>9}  {'worstD':>9}  {'Sharpe':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        is_prod = (abs(r["edge_min"] - PROD_EDGE) < 1e-3
                   and r["price_min"] == PROD_PRICE)
        tag = "  <-- current prod" if is_prod else ""
        print(
            f"{r['edge_min']:>6.1%}  {r['price_min']:>+7d}  "
            f"{r['n_bets']:>5d}  {r['bets_per_day']:>5.1f}  "
            f"{r['hit_rate']:>5.1%}  {r['roi']:>+6.1%}  {r['roi_weighted']:>+6.1%}  "
            f"${r['pnl_total']:>+7.2f}  ${r['pnl_weighted_total']:>+7.2f}  "
            f"${r['worst_day']:>+7.2f}  {r['sharpe']:>+6.2f}"
            f"{tag}"
        )

    print()
    print("How to read this table:")
    print("  - 'wROI' / 'wP&L'  = weighted by 2u sizing on hot bats (production policy).")
    print("  - 'worstD'        = worst single-day P&L; the closer to $0 the smoother.")
    print("  - 'Sharpe'        = daily mean / daily std. Higher = more consistent earnings.")
    print()
    print("Picking a winner (rule of thumb):")
    print("  - If two configs are close on ROI, prefer the higher Sharpe.")
    print("  - Beware configs with very few bets (<10): one lucky day skews the metrics.")
    print("  - To promote: edit FILTER_E_EDGE_MIN / FILTER_E_PRICE_MIN in")
    print("    src/mlbhit/pipeline/recommend.py and historical_backtest.py.")


if __name__ == "__main__":
    main()
