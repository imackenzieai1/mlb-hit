"""Refit a fresh isotonic head on top of xgb_v3 to correct calibration drift.

Why this exists
---------------
Cohort analysis showed xgb_v3 is overconfident by ~5pp on the bets Filter E
selects (avg p_model = 0.733, actual hit rate = 0.685). The XGBoost trees
look fine — top picks consistently profit — but the isotonic head, trained
alongside the trees in 2025, hasn't kept up with the league offensive
environment.

Fix: stack a NEW isotonic regression on top of the existing model output,
fit on data the original calibrator never saw and that doesn't overlap our
backtest window. This is a pure post-hoc recalibration — the trees are
unchanged, only the probability mapping at the very end gets retuned.

Data leakage guard
------------------
xgb_v3 was trained on indices 0-58628 of the modeling parquet, sorted by
date. That ends at 2025-08-01. The remaining 15% (val set) spans
2025-08-01 -> 2026-04-22 and was held out from training. Our backtest
window is 2026-03-20 -> 2026-04-23.

So we have a clean window for recalibration:
    2025-08-01 -> 2026-03-19    (held out from training, before backtest)
That's the calibration set. Backtest stays untouched at 2026-03-20+.

Usage:
    python scripts/recalibrate_isotonic.py
    python -m mlbhit.pipeline.historical_backtest \\
        --start 2026-03-20 --end 2026-04-23 \\
        --filter-e --require-pitcher --model xgb_v3_recal
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from mlbhit.config import SETTINGS
from mlbhit.io import modeling_path
from mlbhit.model.predict import load_model, predict
from mlbhit.model.recalibrated import StackedCalibratedModel

# Calibration window: post-train, pre-backtest. See module docstring.
RECAL_START = "2025-08-01"
RECAL_END   = "2026-03-19"

BASE_MODEL = "xgb_v3"
OUT_NAME   = "xgb_v3_recal"


def _calibration_curve(p, y, n_bins=10) -> pd.DataFrame:
    """Reliability table: bucket probabilities and report bucket hit rate."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    bucket = np.digitize(p, edges) - 1
    rows = []
    for b in range(n_bins):
        mask = bucket == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket":      b + 1,
            "p_lo":        float(edges[b]),
            "p_hi":        float(edges[b + 1]),
            "n":           int(mask.sum()),
            "avg_p":       float(p[mask].mean()),
            "hit_rate":    float(y[mask].mean()),
            "bias":        float(p[mask].mean() - y[mask].mean()),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print(f"RECALIBRATING {BASE_MODEL} -> {OUT_NAME}")
    print("=" * 80)
    print(f"  calibration window: {RECAL_START} -> {RECAL_END}")
    print()

    # 1. Load the calibration window
    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    df = df[(df["date"].astype(str) >= RECAL_START)
            & (df["date"].astype(str) <= RECAL_END)].copy()
    df = df.dropna(subset=["got_hit"])
    df = df.drop_duplicates(subset=["date", "player_id"], keep="first")
    print(f"  rows in calibration window: {len(df):,}")
    print(f"  unique dates:               {df['date'].nunique()}")
    print(f"  base hit rate:              {df['got_hit'].mean():.4f}")
    print()

    # 2. Score with the existing model
    print(f"  scoring calibration window with {BASE_MODEL}...")
    p_old = predict(df, name=BASE_MODEL).values
    y = df["got_hit"].astype(int).values
    print(f"  avg p_model (before recal): {p_old.mean():.4f}")
    print(f"  bias (p - actual):          {p_old.mean() - y.mean():+.4f}")
    print(f"  brier (before):             {brier_score_loss(y, p_old):.4f}")
    print(f"  log_loss (before):          {log_loss(y, p_old):.4f}")
    print()

    # 3. Reliability table BEFORE
    print("  Reliability BEFORE recalibration:")
    cal_before = _calibration_curve(p_old, y, n_bins=10)
    print(cal_before.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # 4. Fit a fresh isotonic on top
    print("  Fitting new IsotonicRegression on (p_old, y)...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_old, y)

    # 5. Verify
    p_new = iso.transform(p_old)
    print(f"  avg p_model (after recal):  {p_new.mean():.4f}")
    print(f"  bias (p - actual):          {p_new.mean() - y.mean():+.4f}")
    print(f"  brier (after):              {brier_score_loss(y, p_new):.4f}")
    print(f"  log_loss (after):           {log_loss(y, p_new):.4f}")
    print()

    print("  Reliability AFTER recalibration:")
    cal_after = _calibration_curve(p_new, y, n_bins=10)
    print(cal_after.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # 6. Wrap and save
    base_model, features = load_model(BASE_MODEL)
    stacked = StackedCalibratedModel(
        base=base_model,
        isotonic=iso,
        recal_meta={
            "base_model":           BASE_MODEL,
            "recal_window_start":   RECAL_START,
            "recal_window_end":     RECAL_END,
            "n_calibration_rows":   int(len(df)),
            "bias_before":          float(p_old.mean() - y.mean()),
            "bias_after":           float(p_new.mean() - y.mean()),
            "brier_before":         float(brier_score_loss(y, p_old)),
            "brier_after":          float(brier_score_loss(y, p_new)),
        },
    )

    out_path = Path(SETTINGS["paths"]["models_dir"]) / f"{OUT_NAME}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stacked, "features": features}, out_path)
    print(f"  saved: {out_path}")
    print()
    print("Next:")
    print(f"  python -m mlbhit.pipeline.historical_backtest \\")
    print(f"    --start 2026-03-20 --end 2026-04-23 \\")
    print(f"    --filter-e --require-pitcher --model {OUT_NAME}")


if __name__ == "__main__":
    main()
