from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from ..config import SETTINGS
from ..io import modeling_path

# Season-level features. platoon_advantage dropped — it's hardcoded to 0 in
# build_features.py pending real L/R handedness extraction, and a constant
# column under monotonic constraints is pure noise.
# bat_contact_pct and bat_solid_contact_pct dropped — not produced by
# build_batter_features so they become constants after median-fill (0 importance).
_SEASON_FEATURES = [
    "bat_xba_season",
    "bat_k_pct",
    "bat_hard_hit_pct",
    "bat_sweet_spot_pct",
    "bat_line_drive_pct",
    "sp_xba_allowed",
    "sp_k_pct",
    "sp_hard_hit_allowed",
    "sp_sweet_spot_allowed",
    "pen_xba_allowed",
    "pen_k_pct",
    "exp_pa",
    "batting_order",
    "home_away_is_home",
    "park_factor",
    "singles_park_factor",
    "xba_diff",
    "exposure_wtd_opp_xba",
    "pitcher_low_sample",
    # v3: matchup-aware platoon features. These pre-pick the right L/R column
    # at row-build time so XGB sees one number per row instead of two
    # half-relevant ones.
    "bat_xba_vs_opphand",
    "bat_k_pct_vs_opphand",
    "sp_xba_allowed_vs_bathand",
    "sp_k_pct_allowed_vs_bathand",
    "platoon_advantage",
    # v3: pitcher hittability. zone% = swings allowed in zone (Statcast zones
    # 1-9). contact% allowed = (swings - whiffs) / swings — a separate dial
    # from k%, since some pitchers have decent K rates but get torched on
    # contact.
    "sp_zone_pct",
    "sp_contact_pct_allowed",
]

# Leakage-safe rolling features from data/clean/batter_rolling.parquet.
# XGBoost handles NaN natively; we deliberately DO NOT median-fill these so
# that "no recent form" remains distinguishable from "recent form = league avg".
_ROLLING_FEATURES = [
    "PA_14d", "PA_30d",
    "ba_14d", "ba_30d",
    "xba_14d", "xba_30d",
    "hard_hit_pct_14d", "hard_hit_pct_30d",
    "sweet_spot_pct_14d", "sweet_spot_pct_30d",
    # v3: pitcher recent form (mirror of batter rolling, joined on opp_sp_id+date).
    "TBF_14d", "TBF_30d",
    "sp_xba_allowed_14d", "sp_xba_allowed_30d",
    "sp_k_pct_14d", "sp_k_pct_30d",
    "sp_hard_hit_allowed_14d", "sp_hard_hit_allowed_30d",
    "sp_contact_pct_allowed_14d", "sp_contact_pct_allowed_30d",
]

FEATURES = _SEASON_FEATURES + _ROLLING_FEATURES

MONO = {
    # Season-level
    "bat_xba_season": +1,
    "bat_hard_hit_pct": +1,
    "bat_sweet_spot_pct": +1,
    "bat_line_drive_pct": +1,
    "bat_k_pct": -1,
    "sp_xba_allowed": +1,
    "sp_hard_hit_allowed": +1,
    "sp_sweet_spot_allowed": +1,
    "sp_k_pct": -1,
    "pen_xba_allowed": +1,
    "pen_k_pct": -1,
    "exp_pa": +1,
    "xba_diff": +1,
    "park_factor": +1,
    "singles_park_factor": +1,
    # v3 matchup-aware
    "bat_xba_vs_opphand": +1,
    "bat_k_pct_vs_opphand": -1,
    "sp_xba_allowed_vs_bathand": +1,
    "sp_k_pct_allowed_vs_bathand": -1,
    "platoon_advantage": +1,
    # v3 pitcher hittability — more contact allowed = more hits surrendered.
    # Zone% intentionally unconstrained: throwing strikes is correlated with
    # both K% (good) and BB-avoidance (lets balls in play) — direction is
    # genuinely ambiguous, let XGB find it.
    "sp_contact_pct_allowed": +1,
    # Rolling — same sign priors as the season versions.
    "ba_14d": +1,
    "ba_30d": +1,
    "xba_14d": +1,
    "xba_30d": +1,
    "hard_hit_pct_14d": +1,
    "hard_hit_pct_30d": +1,
    "sweet_spot_pct_14d": +1,
    "sweet_spot_pct_30d": +1,
    "sp_xba_allowed_14d": +1,
    "sp_xba_allowed_30d": +1,
    "sp_k_pct_14d": -1,
    "sp_k_pct_30d": -1,
    "sp_hard_hit_allowed_14d": +1,
    "sp_hard_hit_allowed_30d": +1,
    "sp_contact_pct_allowed_14d": +1,
    "sp_contact_pct_allowed_30d": +1,
    # PA_14d / PA_30d / TBF_14d / TBF_30d intentionally unconstrained — more
    # exposure = more signal but doesn't monotonically push p(hit) either way.
}


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["home_away_is_home"] = (df["home_away"] == "H").astype(int)
    df["batting_order"] = df["batting_order"].fillna(5).astype(int)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan  # was 0.0 — NaN lets XGBoost route missing cleanly.

    # Median-fill season-level features only. Rolling features keep their NaNs
    # so that "no recent form" stays distinct from "form = league median".
    X = df[FEATURES].copy()
    season_medians = X[_SEASON_FEATURES].median(numeric_only=True)
    X[_SEASON_FEATURES] = X[_SEASON_FEATURES].fillna(season_medians)

    y = df["got_hit"].astype(int)
    return X, y


def monotone_tuple() -> tuple[int, ...]:
    return tuple(MONO.get(f, 0) for f in FEATURES)


def train(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    model_name: str = "xgb_v1",
    calibration: str = "isotonic",
) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    X, y = prepare(df)
    split = int(len(df) * (1 - val_frac))
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    base = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        monotone_constraints=monotone_tuple(),
    )
    if calibration not in ("isotonic", "sigmoid"):
        raise ValueError(f"calibration must be 'isotonic' or 'sigmoid'; got {calibration!r}")
    calibrated = CalibratedClassifierCV(base, method=calibration, cv=3)
    calibrated.fit(X_tr, y_tr)

    p_val = calibrated.predict_proba(X_val)[:, 1]
    p_base_const = np.full(len(y_val), y_tr.mean())

    metrics = {
        "log_loss": float(log_loss(y_val, p_val)),
        "log_loss_const": float(log_loss(y_val, p_base_const)),
        "brier": float(brier_score_loss(y_val, p_val)),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "hit_rate_train": float(y_tr.mean()),
    }

    out_model = Path(SETTINGS["paths"]["models_dir"]) / f"{model_name}.joblib"
    out_meta = Path(SETTINGS["paths"]["models_dir"]) / f"{model_name}.json"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": calibrated, "features": FEATURES}, out_model)
    with open(out_meta, "w") as f:
        json.dump(
            {"features": FEATURES, "metrics": metrics, "monotone": MONO,
             "calibration": calibration},
            f, indent=2,
        )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic",
                        help="Try 'sigmoid' (Platt) if isotonic looks unstable on small val sets.")
    parser.add_argument("--model-name", default="xgb_v1")
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    m = train(df, val_frac=args.val_frac, model_name=args.model_name,
              calibration=args.calibration)
    print(json.dumps(m, indent=2))
