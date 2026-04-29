#!/usr/bin/env python
"""Train xgb_v5: production promotion of the Optuna joint-search winner.

Bakes in the hyperparameters that scripts/optuna_joint.py discovered
(see models/xgb_optuna_winner.json). Same v3 feature set, same train cutoff
(2026-03-19), same monotone constraints, same isotonic calibration as v3.
The ONLY change vs v3 is the (max_depth, learning_rate, n_estimators,
subsample, colsample_bytree, reg_lambda) tuple.

Why "v5" not "xgb_optuna":
    * xgb_optuna.joblib already exists from the optuna_joint run, but it was
      produced inside a hyperparameter search loop with verbosity suppressed
      and no model JSON — fine for evaluation, but not the kind of artifact
      we want to ship to production. v5 is a fresh, deliberate train with a
      proper metadata file alongside.
    * Naming v3 -> v5 (skipping v4_recal which was a feature-set experiment
      that didn't ship) keeps the model lineage linear and matches how the
      rest of the codebase references models by integer version.

Outputs:
    models/xgb_v5.joblib   — XGBoost trees + isotonic calibration head
    models/xgb_v5.json     — features, hyperparams, monotone, val metrics

Next step (separately): scripts/recalibrate_isotonic.py --base-model xgb_v5
to fit a fresh post-hoc isotonic on the 2025-08-01 -> 2026-03-19 window,
producing xgb_v5_recal.joblib — the actual production-bound artifact.

Usage:
    python scripts/train_xgb_v5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlbhit.config import SETTINGS  # noqa: E402
from mlbhit.io import modeling_path  # noqa: E402
from mlbhit.model.train import (  # noqa: E402
    MONO,
    features_for,
    monotone_tuple,
    prepare,
)

# ---------------------------------------------------------------------------
# Hyperparameters — copied exactly from models/xgb_optuna_winner.json
# (see optuna_joint.py for the search space + holdout validation). Don't
# tweak by hand: if you want to change them, re-run optuna_joint.py and
# copy the new winner here so the lineage stays auditable.
# ---------------------------------------------------------------------------
OPTUNA_PARAMS = {
    "max_depth":        5,
    "learning_rate":    0.030890853612453147,
    "n_estimators":     200,
    "subsample":        0.8640475728584198,
    "colsample_bytree": 0.8125018731898727,
    "reg_lambda":       3.1023367934927295,
}

# Match optuna_joint.py exactly: train through 2026-03-19 so the same
# train/val/holdout boundaries hold and v5_recal can re-use the same
# 2025-08-01 -> 2026-03-19 calibration window we already validated.
TRAIN_END = "2026-03-19"
VAL_START = "2026-03-20"   # holdout-style val for in-script logging only
VAL_END   = "2026-04-26"

MODEL_NAME    = "xgb_v5"
FEATURE_SET   = "xgb_v3"   # Optuna optimized over the v3 feature set


def main() -> None:
    print("=" * 70)
    print(f"TRAIN  {MODEL_NAME}  (Optuna winner promoted)")
    print("=" * 70)
    print(f"  feature set:  {FEATURE_SET}")
    print(f"  train end:    {TRAIN_END}")
    print(f"  val window:   {VAL_START} -> {VAL_END}")
    for k, v in OPTUNA_PARAMS.items():
        print(f"  {k:18s} {v}")
    print()

    feats = features_for(FEATURE_SET)

    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    train_df = df[df["date"] <= TRAIN_END].copy()
    val_df   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    print(f"  train rows: {len(train_df):,}  ({train_df['date'].min()} -> {train_df['date'].max()})")
    print(f"  val rows:   {len(val_df):,}  ({val_df['date'].min()} -> {val_df['date'].max()})")
    print()

    X_tr, y_tr = prepare(train_df, features=feats)
    X_val, y_val = prepare(val_df, features=feats)

    print("Fitting CalibratedClassifierCV(method='isotonic', cv=3)...")
    base = XGBClassifier(
        **OPTUNA_PARAMS,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        monotone_constraints=monotone_tuple(feats),
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)

    # Coerce val features to plain float64 — predict.py does this in production
    # for the same reason: nullable Float64 trips XGBoost.
    X_val_n = X_val.copy()
    for c in X_val_n.columns:
        X_val_n[c] = pd.to_numeric(X_val_n[c], errors="coerce").astype(np.float64)

    p_val = calibrated.predict_proba(X_val_n)[:, 1]
    p_base_const = np.full(len(y_val), float(y_tr.mean()))

    metrics = {
        "log_loss":       float(log_loss(y_val, p_val)),
        "log_loss_const": float(log_loss(y_val, p_base_const)),
        "brier":          float(brier_score_loss(y_val, p_val)),
        "roc_auc":        float(roc_auc_score(y_val, p_val)),
        "n_train":        int(len(X_tr)),
        "n_val":          int(len(X_val)),
        "hit_rate_train": float(y_tr.mean()),
        "hit_rate_val":   float(y_val.mean()),
        "avg_p_val":      float(p_val.mean()),
        "bias_val":       float(p_val.mean() - y_val.mean()),
    }

    print()
    print("Validation metrics (this is global cal/discrimination, not the")
    print("Filter-E-tail ROI Optuna actually optimized for):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s} {v:.4f}")
        else:
            print(f"  {k:18s} {v}")
    print()
    print("(Expect log_loss < log_loss_const and ROC AUC > 0.5.")
    print(" The bigger ROI/Sharpe signal is in models/xgb_optuna_winner.json.)")

    # ------------------------------------------------------------------
    # Save the model bundle. Same shape as xgb_v1/v2/v3 so anything that
    # already loads via load_model() works unchanged when DEFAULT_MODEL is
    # flipped to xgb_v5_recal.
    # ------------------------------------------------------------------
    models_dir = Path(SETTINGS["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    out_model = models_dir / f"{MODEL_NAME}.joblib"
    out_meta  = models_dir / f"{MODEL_NAME}.json"
    joblib.dump({"model": calibrated, "features": feats}, out_model)
    with open(out_meta, "w") as f:
        json.dump(
            {
                "features":    feats,
                "feature_set": FEATURE_SET,
                "params":      OPTUNA_PARAMS,
                "train_end":   TRAIN_END,
                "val_window":  [VAL_START, VAL_END],
                "metrics":     metrics,
                "monotone":    {k: v for k, v in MONO.items() if k in feats},
                "calibration": "isotonic",
                "lineage":     "Optuna joint search winner (see models/xgb_optuna_winner.json)",
            },
            f, indent=2,
        )
    print(f"\nSaved: {out_model}")
    print(f"Saved: {out_meta}")
    print()
    print("Next:")
    print(f"  python scripts/recalibrate_isotonic.py --base-model {MODEL_NAME}")
    print(f"  # produces models/{MODEL_NAME}_recal.joblib for production wiring")


if __name__ == "__main__":
    main()
