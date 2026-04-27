"""Evaluate the trained hit-prop model: calibration, importance, decile lift."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..config import SETTINGS
from ..io import modeling_path
from .train import FEATURES, prepare


def _split(df: pd.DataFrame, val_frac: float = 0.15):
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * (1 - val_frac))
    return df.iloc[:split], df.iloc[split:]


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(set(bins)):
        m = bins == b
        rows.append({
            "decile": int(b) + 1,
            "n": int(m.sum()),
            "pred_mean": float(p[m].mean()),
            "actual_hit_rate": float(y[m].mean()),
            "gap": float(y[m].mean() - p[m].mean()),
        })
    return pd.DataFrame(rows)


def top_k_per_day_metrics(
    dates: pd.Series, y: np.ndarray, p: np.ndarray, ks: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """For each day in the val set, pick the top-K batters by predicted p(hit),
    then aggregate across days.

    This matches Ian's operational use case: we don't act on every batter, we
    only bet the top-K most confident picks per slate. The log_loss here is a
    much more useful signal of model quality *where it matters* than overall
    log_loss computed across all batters (which includes a long tail of
    "nobody cares" low-probability predictions).

    Columns returned:
        k              - picks per day
        n_picks        - total rows that made the top-K across all val days
        hit_rate       - fraction of picks that actually got 1+ hits
        mean_predicted - mean predicted p(hit) among picks (sanity: should be close to hit_rate)
        log_loss       - log loss computed only on picks (probability quality where we bet)
        calibration_gap - hit_rate - mean_predicted (positive = we under-predicted winners)
    """
    df = pd.DataFrame({"date": dates.values, "y": y, "p": p})
    rows = []
    for k in ks:
        picks = (
            df.sort_values(["date", "p"], ascending=[True, False])
            .groupby("date", group_keys=False)
            .head(k)
        )
        if picks.empty:
            continue
        yk = picks["y"].to_numpy()
        pk = picks["p"].to_numpy()
        # Clip predictions away from 0/1 for log_loss stability (same as sklearn).
        pk_clip = np.clip(pk, 1e-6, 1 - 1e-6)
        ll = float(-(yk * np.log(pk_clip) + (1 - yk) * np.log(1 - pk_clip)).mean())
        rows.append({
            "k": k,
            "n_picks": int(len(picks)),
            "hit_rate": float(yk.mean()),
            "mean_predicted": float(pk.mean()),
            "log_loss": ll,
            "calibration_gap": float(yk.mean() - pk.mean()),
        })
    return pd.DataFrame(rows)


def feature_importance(calibrated_model, features: list[str]) -> pd.DataFrame:
    # CalibratedClassifierCV stores fitted base estimators in .calibrated_classifiers_
    # Each has a .estimator attribute that is the fitted XGBClassifier.
    importances = []
    for cc in calibrated_model.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    if not importances:
        return pd.DataFrame({"feature": features, "importance": [np.nan] * len(features)})
    mean_imp = np.mean(np.stack(importances), axis=0)
    return (
        pd.DataFrame({"feature": features, "importance": mean_imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate(model_name: str = "xgb_v1") -> dict:
    model_path = Path(SETTINGS["paths"]["models_dir"]) / f"{model_name}.joblib"
    payload = joblib.load(model_path)
    model = payload["model"]
    feats = payload["features"]

    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    _, val = _split(df)
    X, y = prepare(val)
    p = model.predict_proba(X[feats])[:, 1]

    # Keep the val date series aligned with y/p for top-K-per-day metrics.
    val_dates = val.sort_values("date").reset_index(drop=True)["date"]

    metrics = {
        "n_val": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "log_loss_const": float(log_loss(y, np.full(len(y), y.mean()))),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "hit_rate_val": float(y.mean()),
    }
    print("=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")

    print("\n" + "=" * 60)
    print("RELIABILITY (per decile of predicted probability)")
    print("=" * 60)
    rel = reliability_table(y.values, p)
    print(rel.to_string(index=False))

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (mean across calibrated folds)")
    print("=" * 60)
    imp = feature_importance(model, feats)
    print(imp.to_string(index=False))

    # Flag launch-angle features position — V1 thesis check
    la_feats = {
        "bat_sweet_spot_pct",
        "bat_line_drive_pct",
        "bat_hard_hit_pct",
        "bat_solid_contact_pct",
    }
    la_ranks = imp[imp["feature"].isin(la_feats)]
    print("\n" + "=" * 60)
    print("LAUNCH-ANGLE FEATURE RANKS (V1 thesis check)")
    print("=" * 60)
    print(la_ranks.to_string())
    top5 = set(imp.head(5)["feature"].tolist())
    hits_top5 = la_feats & top5
    if hits_top5:
        print(f"  OK: {len(hits_top5)} launch-angle features in top 5: {sorted(hits_top5)}")
    else:
        print("  WARN: no launch-angle features in top 5 — monotonic constraints may need strengthening")

    # Top-K-per-day: how well-calibrated are we where we actually bet?
    print("\n" + "=" * 60)
    print("TOP-K PER DAY (operational metric — picks you'd actually bet)")
    print("=" * 60)
    topk = top_k_per_day_metrics(val_dates, y.values, p, ks=(10, 20))
    if not topk.empty:
        print(topk.to_string(index=False))
        topk.to_csv(out_dir_for_topk := Path(SETTINGS["paths"]["models_dir"])
                    / f"{model_name}_topk.csv", index=False)
        metrics["top10_hit_rate"] = float(topk.loc[topk["k"] == 10, "hit_rate"].iloc[0]) \
            if (topk["k"] == 10).any() else None
        metrics["top10_log_loss"] = float(topk.loc[topk["k"] == 10, "log_loss"].iloc[0]) \
            if (topk["k"] == 10).any() else None
        metrics["top20_hit_rate"] = float(topk.loc[topk["k"] == 20, "hit_rate"].iloc[0]) \
            if (topk["k"] == 20).any() else None
        metrics["top20_log_loss"] = float(topk.loc[topk["k"] == 20, "log_loss"].iloc[0]) \
            if (topk["k"] == 20).any() else None
    else:
        print("  (no data — check that val set has multiple dates)")

    out_dir = Path(SETTINGS["paths"]["models_dir"])
    rel.to_csv(out_dir / f"{model_name}_reliability.csv", index=False)
    imp.to_csv(out_dir / f"{model_name}_importance.csv", index=False)
    with open(out_dir / f"{model_name}_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nArtifacts written to {out_dir}")

    return {"metrics": metrics, "reliability": rel, "importance": imp, "topk": topk}


if __name__ == "__main__":
    evaluate()
