#!/usr/bin/env python
"""Honest holdout comparison: Optuna's winning config vs xgb_v3_recal at the
hand-tuned Filter E v2 gate, evaluated on the same holdout window.

This script is the "truth check" on whether Optuna's joint optimization
delivered real lift or just overfit to the validation window. The holdout
window (2026-04-20 -> 2026-04-26) is the same for both models — only
diff is the (model, gate) configuration.

Outputs to stdout. No artifacts written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlbhit.config import SETTINGS  # noqa: E402

# Reuse the helpers from optuna_joint.py — same data loading, same gate logic.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from optuna_joint import (  # noqa: E402
    HOLDOUT_END,
    HOLDOUT_START,
    HOT_STREAK_UNITS,
    _attach_hot_streak_to_eval,
    _load_historical_odds,
    _load_modeling_parquet,
    _score_window,
    evaluate_gate,
)

# Filter E v2 baseline thresholds (matches recommend.py / historical_backtest).
# hot_streak_units uses the same value as the joint search uses (fixed at 2.0)
# so the comparison is purely about whether the (model, gate) combo improved.
BASELINE_EDGE_MIN  = 0.15
BASELINE_PRICE_MAX = -200


def _load_v3_recal():
    """Load the production xgb_v3_recal model bundle."""
    p = Path(SETTINGS["paths"]["models_dir"]) / "xgb_v3_recal.joblib"
    bundle = joblib.load(p)
    return bundle["model"], bundle["features"]


def _load_optuna_winner():
    p = Path(SETTINGS["paths"]["models_dir"]) / "xgb_optuna_winner.json"
    with open(p) as f:
        return json.load(f)


def _load_optuna_model():
    p = Path(SETTINGS["paths"]["models_dir"]) / "xgb_optuna.joblib"
    bundle = joblib.load(p)
    return bundle["model"], bundle["features"]


def main():
    print("=" * 70)
    print("HOLDOUT COMPARISON  (both models, same holdout window)")
    print("=" * 70)
    print(f"  holdout: {HOLDOUT_START} -> {HOLDOUT_END}")
    print()

    # Data once — same eval pool for both models.
    df = _load_modeling_parquet()
    eval_df = df[(df["date"] >= HOLDOUT_START) & (df["date"] <= HOLDOUT_END)].copy()
    print(f"  eval rows: {len(eval_df):,}")

    odds = _load_historical_odds(HOLDOUT_START, HOLDOUT_END)
    print(f"  odds rows: {len(odds):,}")

    eval_df = _attach_hot_streak_to_eval(eval_df)
    print()

    # ---------------- Baseline: xgb_v3_recal at Filter E v2 ----------------
    print("Scoring with xgb_v3_recal...")
    base_model, base_feats = _load_v3_recal()
    base_scored = eval_df.copy()
    base_scored["p_model"] = _score_window(base_model, base_feats, eval_df).values
    base_result = evaluate_gate(
        base_scored, odds,
        edge_min=BASELINE_EDGE_MIN,
        price_max=BASELINE_PRICE_MAX,
        hot_streak_units=HOT_STREAK_UNITS,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
    )

    # ---------------- Optuna winner ----------------
    print("Scoring with xgb_optuna...")
    winner_meta = _load_optuna_winner()
    opt_model, opt_feats = _load_optuna_model()
    opt_scored = eval_df.copy()
    opt_scored["p_model"] = _score_window(opt_model, opt_feats, eval_df).values
    opt_params = winner_meta["best_params"]
    opt_result = evaluate_gate(
        opt_scored, odds,
        edge_min=opt_params["edge_min"],
        price_max=opt_params["price_max"],
        hot_streak_units=HOT_STREAK_UNITS,  # same as baseline — fair comparison
        start=HOLDOUT_START,
        end=HOLDOUT_END,
    )

    # ---------------- Side-by-side ----------------
    print()
    print(f"{'Metric':<20} {'v3_recal (FilterE v2)':>26} {'xgb_optuna (joint)':>22}")
    print("-" * 70)
    rows = [
        ("bets",         f"{base_result['n_bets']}",         f"{opt_result['n_bets']}"),
        ("hit rate",     f"{base_result['hit_rate']:.1%}",   f"{opt_result['hit_rate']:.1%}"),
        ("ROI flat",     f"{base_result['roi_flat']:+.1%}",  f"{opt_result['roi_flat']:+.1%}"),
        ("ROI weighted", f"{base_result['roi_weighted']:+.1%}", f"{opt_result['roi_weighted']:+.1%}"),
        ("P&L total",    f"${base_result['pnl_total']:+.2f}", f"${opt_result['pnl_total']:+.2f}"),
        ("daily Sharpe", f"{base_result['sharpe']:+.3f}",    f"{opt_result['sharpe']:+.3f}"),
        ("worst day",    f"${base_result['worst_day']:+.2f}", f"${opt_result['worst_day']:+.2f}"),
    ]
    for label, v_base, v_opt in rows:
        print(f"  {label:<18} {v_base:>26} {v_opt:>22}")
    print()
    print(f"hot_streak_units fixed at {HOT_STREAK_UNITS} for both (separate optimization).")
    print()
    print("Optuna winning config:")
    for k, v in opt_params.items():
        if isinstance(v, float):
            print(f"  {k:18s} {v:.4f}")
        else:
            print(f"  {k:18s} {v}")

    # ---------------- Verdict ----------------
    print()
    delta_sharpe = opt_result["sharpe"] - base_result["sharpe"]
    delta_roi = opt_result["roi_weighted"] - base_result["roi_weighted"]
    delta_pnl = opt_result["pnl_total"] - base_result["pnl_total"]
    val_sharpe_median = float(np.median([s["sharpe"] for s in winner_meta["val_subs"]]))
    sharpe_gap = opt_result["sharpe"] - val_sharpe_median

    print(f"Δ holdout Sharpe (opt - base): {delta_sharpe:+.3f}")
    print(f"Δ holdout ROI:                 {delta_roi:+.2%}")
    print(f"Δ holdout P&L:                 ${delta_pnl:+.2f}")
    print(f"opt val median Sharpe:         {val_sharpe_median:+.3f}")
    print(f"opt holdout Sharpe:            {opt_result['sharpe']:+.3f}")
    print(f"val→holdout Sharpe drop:       {sharpe_gap:+.3f}")
    print()

    # Verdict uses Sharpe (the optimized metric) as primary signal, with ROI
    # as secondary confirmation. Values calibrated to our sample size.
    if delta_sharpe > 0.10 and sharpe_gap > -0.20 and delta_roi > 0:
        verdict = ("WIN — Optuna config beats baseline on Sharpe with no "
                   "overfit warning. Promote to production.")
    elif delta_sharpe > 0 and sharpe_gap > -0.20:
        verdict = ("MARGINAL — small Sharpe lift, no overfit signal. Could "
                   "promote, but the gain may be sample noise. Re-run after "
                   "another few weeks of data accumulate.")
    elif sharpe_gap < -0.20:
        verdict = ("OVERFIT — holdout Sharpe substantially below val Sharpe. "
                   "Optuna found a config that won't generalize. "
                   "Stay on v3_recal at Filter E v2.")
    else:
        verdict = ("WASH or LOSS — holdout doesn't show Sharpe lift. "
                   "Stay on v3_recal at Filter E v2.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
