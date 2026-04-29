from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import SETTINGS
from .train import prepare

# Default to v3_recal: same XGBoost trees as xgb_v3, with a fresh isotonic
# head stacked on top. The original isotonic was fit alongside the trees on
# 2023-03 → 2025-08 data; xgb_v3_recal refits a post-hoc isotonic on the
# held-out 2025-08 → 2026-03 window so the calibrator reflects the current
# offensive environment. Filter E backtest 2026-03-20 → 2026-04-23:
#   xgb_v3        539 bets   68.5% hit   +15.5% ROI   daily-Sharpe 0.70   worst -48.8%
#   xgb_v3_recal  558 bets   69.2% hit   +15.9% ROI   daily-Sharpe 0.80   worst -35.8%
# The ROI bump is modest, but variance reduction is meaningful — the
# worst-day drawdown shrunk from -49% to -36% and global log-loss /
# brier improved on the calibration window. v3 is kept as a fallback
# (--model xgb_v3) for A/B comparisons.
DEFAULT_MODEL = "xgb_v3_recal"


def load_model(name: str = DEFAULT_MODEL):
    p = Path(SETTINGS["paths"]["models_dir"]) / f"{name}.joblib"
    bundle = joblib.load(p)
    return bundle["model"], bundle["features"]


def predict(df: pd.DataFrame, name: str = DEFAULT_MODEL) -> pd.Series:
    model, feats = load_model(name)
    # Pass the model's saved feature list to prepare() so it knows which
    # columns to keep — without this, prepare() defaults to the module-level
    # FEATURES (v3) and strips any v4-specific columns out of df before
    # X[feats] can find them.
    X, _ = prepare(df.assign(got_hit=0), features=feats)
    X_feats = X[feats].copy()
    # XGBoost only accepts plain numpy dtypes. The rolling-features parquet
    # writes nullable Float64 (which pandas reports as "object" when it has
    # pd.NA values). Force-coerce every feature column to plain float64 so
    # NaN is numpy-native and dtype is what XGBoost expects.
    for c in X_feats.columns:
        X_feats[c] = pd.to_numeric(X_feats[c], errors="coerce").astype(np.float64)
    return pd.Series(model.predict_proba(X_feats)[:, 1], index=df.index, name="p_model")
