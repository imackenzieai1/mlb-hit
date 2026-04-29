from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import SETTINGS
from .train import prepare

# Default to v5_recal: Optuna joint-search winner (see scripts/optuna_joint.py
# and models/xgb_optuna_winner.json) trained with the v3 feature set, then
# stacked with a fresh isotonic head on the same 2025-08-01 -> 2026-03-19
# calibration window v3_recal used. Holdout 2026-04-20 -> 2026-04-26:
#   xgb_v3_recal  154 bets   71.4% hit   +18.7% ROI   daily-Sharpe 1.02
#   xgb_v5/optuna 170 bets   77.6% hit   +29.7% ROI   daily-Sharpe 1.99
# The v5 trees come from a 50-trial bounded search over (max_depth, lr,
# n_estimators, subsample, colsample_bytree, reg_lambda) that maximized
# median Filter-E Sharpe across 3 contiguous val sub-windows; holdout was
# never seen by the search.
#
# xgb_v3_recal is kept as the named-fallback prior-production model: it's
# the validated "no qualifier" production setup at Filter E v2.1 (edge>=15%,
# price>=-250, start_rate>=80% on projected lineups, 2x sizing on hot bats).
# Pass --model xgb_v3_recal anywhere a model is selectable to flip back.
# Don't delete xgb_v3_recal.joblib — it's the rollback artifact.
DEFAULT_MODEL = "xgb_v3_recal"

# Prior production model — kept available for rollback or A/B comparison.
# See module docstring at the top of this file for the v2.1 Filter E gate
# that pairs with this model.
PRIOR_PROD_MODEL = "xgb_v3_recal"


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
