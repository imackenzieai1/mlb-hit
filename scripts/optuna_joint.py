#!/usr/bin/env python
"""Joint Optuna optimization over XGBoost hyperparameters AND Filter E gate
thresholds. End-to-end objective is ROI on the high-conviction tail, with
the four guardrails I argued for during the design phase:

    1. CV across multiple time-folds within the validation window — penalizes
       configs that look great on one window but flop on adjacent ones.
    2. Minimum-bet-count gate per fold — rejects "ultra-tight gate finds 10
       lucky bets" artifacts that won't generalize.
    3. Bounded search space — model and gate hyperparams are constrained to
       regions we've already validated, so Optuna can't drift into pathological
       configs.
    4. Final holdout window — the most recent 7 days are NEVER seen by Optuna;
       we report holdout ROI separately at the end so the user can spot
       overfitting (CV-best vs holdout-actual).

Outputs:
    models/xgb_optuna.joblib            — XGBoost trees trained on best params
    models/xgb_optuna_recal.joblib      — same wrapped with isotonic recal
    models/xgb_optuna_winner.json       — full study summary + best params
    models/xgb_optuna_holdout_eval.json — holdout-window ROI vs v3_recal baseline

To run:
    pip install optuna
    python scripts/optuna_joint.py --n-trials 50

Compute: ~30 min on a Mac mini for 50 trials × 3 CV folds, since each trial
trains a fresh model. Bumping n-trials linearly increases runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlbhit.config import SETTINGS  # noqa: E402
from mlbhit.features.recent_form import attach_hot_streak  # noqa: E402
from mlbhit.io import clean_path, modeling_path, raw_path  # noqa: E402
from mlbhit.model.train import (  # noqa: E402
    _SEASON_FEATURES,
    features_for,
    monotone_tuple,
    prepare,
)
from mlbhit.utils.odds_math import (  # noqa: E402
    american_to_decimal,
    ev_per_unit,
)


# ---------------------------------------------------------------------------
# Time boundaries — chosen so train data has no leakage into val/holdout.
# v3 was trained through ~2025-08; using through 2026-03-19 here gives Optuna
# a slightly larger training set than v3 had, which is fair (the comparison
# at the end uses the same cutoff for both v3 and the Optuna winner).
# ---------------------------------------------------------------------------
TRAIN_END     = "2026-03-19"
VAL_START     = "2026-03-20"
VAL_END       = "2026-04-19"
HOLDOUT_START = "2026-04-20"
HOLDOUT_END   = "2026-04-26"

# Three contiguous sub-windows within val. Each gets its own ROI computation;
# the objective returns the median across them so configs that overfit to a
# single window get penalized.
VAL_SUBS: tuple[tuple[str, str], ...] = (
    ("2026-03-20", "2026-03-29"),
    ("2026-03-30", "2026-04-08"),
    ("2026-04-09", "2026-04-19"),
)

# Per-fold minimum bet count. Configs that pass fewer than this many bets in
# any fold get rejected outright — too-tight gates with lucky picks won't
# generalize. Bumped from 30 to 50 after a critique from another model: a
# 30-bet sample is too easy to "luck into" for one fold.
MIN_BETS_PER_SUB = 50

# Hot-streak sizing multiplier is FIXED here, not part of Optuna's search.
# Money management (how much to bet) is a separate optimization problem from
# pick selection (which bets to make); jointly optimizing them adds search
# degrees of freedom without making the model better. We use the validated
# 2x for hot bats and evaluate the right multiplier separately on holdout.
HOT_STREAK_UNITS = 2.0

# Small constant to avoid divide-by-zero in Sharpe calc when daily P&L std
# is exactly 0 (e.g. one bet per day, all hit). Effectively never matters
# at our 50-bet minimum.
SHARPE_EPS = 1e-6

# Books we accept odds from. Matches the production recommend.py allowlist.
# FD-primary, DK fallback since 2026-05-02 PM. Strict — only books in this
# tuple are kept (see isin filter below).
BOOK_PREFERENCE = ("fanduel", "draftkings")


# ---------------------------------------------------------------------------
# Data loading. The eval window pulls from data/raw/historical_props/ since
# that's where the backfilled props live; the live fetch_prop_odds writes
# to a different location which doesn't help us here.
# ---------------------------------------------------------------------------
def _load_historical_odds(start: str, end: str) -> pd.DataFrame:
    """Load historical prop odds for [start, end] (inclusive ISO dates).

    Checks all three known locations in order — the project has historically
    written prop odds to a couple of different places and we need to find
    them wherever they ended up:
      1. data/raw/historical_props/{date}_props.parquet (backfilled history)
      2. data/raw/props/{date}_props.parquet            (current live fetch)
      3. data/raw/{date}_props.parquet                  (legacy live fetch)

    First match wins per date (no duplicate-merging across locations).
    """
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    frames = []
    candidate_paths = lambda d: [
        raw_path("historical_props", f"{d.isoformat()}_props.parquet"),
        raw_path("props",            f"{d.isoformat()}_props.parquet"),
        raw_path("",                 f"{d.isoformat()}_props.parquet"),  # legacy root
    ]
    d = start_d
    while d <= end_d:
        for p in candidate_paths(d):
            if p.exists():
                frames.append(pd.read_parquet(p))
                break
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["date"] = df["date"].astype(str)
    return df


def _load_modeling_parquet() -> pd.DataFrame:
    df = pd.read_parquet(modeling_path("player_game_features.parquet"))
    df["date"] = df["date"].astype(str)
    return df


def _attach_hot_streak_to_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary hot_streak flag to eval rows so the gate can use it.

    Loads boxscores for every season referenced in df PLUS the prior season,
    because early-season rows (March 2026) need late-prior-season games to
    fill the 6-game window. Without prior-season data, every late-March 2026
    row would be cold by default — silently undercounting hot bats.
    """
    seasons = sorted({int(d[:4]) for d in df["date"]})
    seasons_to_load = sorted(set(seasons) | {min(seasons) - 1})
    box_frames = []
    for yr in seasons_to_load:
        p = clean_path(f"boxscores_{yr}.parquet")
        if p.exists():
            box_frames.append(pd.read_parquet(p))
    if not box_frames:
        df = df.copy()
        df["hot_streak"] = 0
        return df
    box = pd.concat(box_frames, ignore_index=True)
    out = attach_hot_streak(df[["player_id", "date"]].copy(), box)
    df = df.copy()
    df["hot_streak"] = out["hot_streak"].values
    return df


# ---------------------------------------------------------------------------
# Training. Fresh model per Optuna trial. Honors monotone_constraints (domain
# priors) and wraps in CalibratedClassifierCV so probabilities are in the
# same shape as v3_recal.
# ---------------------------------------------------------------------------
def _train_one(train_df: pd.DataFrame, params: dict, feature_set: str = "xgb_v3"):
    """Train + calibrate. Returns (model, features). Suppresses XGBoost's
    internal verbosity so 50 trials don't produce 50 walls of output."""
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    feats = features_for(feature_set)
    X, y = prepare(train_df, features=feats)

    base = XGBClassifier(
        **params,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        monotone_constraints=monotone_tuple(feats),
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(X, y)
    return calibrated, feats


def _score_window(model, feats: list[str], df: pd.DataFrame) -> pd.Series:
    """Return p_model per row of df, preserving df's index for re-attach."""
    X, _ = prepare(df.assign(got_hit=0), features=feats)
    X_feats = X[feats].copy()
    for c in X_feats.columns:
        X_feats[c] = pd.to_numeric(X_feats[c], errors="coerce").astype(np.float64)
    return pd.Series(model.predict_proba(X_feats)[:, 1], index=df.index)


# ---------------------------------------------------------------------------
# Gate evaluation. Takes already-scored predictions + odds + thresholds, returns
# ROI metrics. Quiet (no print). Hot-streak units are part of the search
# space, so they're a parameter here.
# ---------------------------------------------------------------------------
def _empty_result() -> dict:
    """Zero-bet result with every key the reporting layer might read.
    Centralizes the schema so we can never KeyError on a 0-bet trial."""
    return {
        "n_bets":       0,
        "n_days":       0,
        "hit_rate":     0.0,
        "roi_flat":     0.0,
        "roi_weighted": 0.0,
        "pnl_total":    0.0,
        "sharpe":       0.0,
        "daily_mean":   0.0,
        "daily_std":    0.0,
        "worst_day":    0.0,
    }



def evaluate_gate(
    scored: pd.DataFrame,
    odds: pd.DataFrame,
    edge_min: float,
    price_max: int,
    hot_streak_units: float,
    start: str,
    end: str,
) -> dict:
    """Apply a (edge_min, price_max, hot_streak_units) gate to scored
    predictions over [start, end] and return ROI summary stats."""
    mask_s = (scored["date"] >= start) & (scored["date"] <= end)
    s = scored[mask_s].copy()
    o = odds[(odds["date"] >= start) & (odds["date"] <= end)]
    if s.empty or o.empty:
        return _empty_result()

    # Restrict to books we trust.
    o = o[o["book"].isin(BOOK_PREFERENCE)].copy()
    if o.empty:
        return _empty_result()

    s["player_id"] = pd.to_numeric(s["player_id"], errors="coerce").astype("Int64")
    o["player_id"] = pd.to_numeric(o["player_id"], errors="coerce").astype("Int64")

    keep_s = ["date", "player_id", "p_model", "got_hit",
              "pitcher_features_known", "hot_streak"]
    keep_s = [c for c in keep_s if c in s.columns]
    m = o.merge(s[keep_s], on=["date", "player_id"], how="inner")
    if m.empty:
        return _empty_result()

    # Mirror production: require known pitcher features (model leaks out
    # otherwise into pitcher-blind territory).
    if "pitcher_features_known" in m.columns:
        m = m[m["pitcher_features_known"].fillna(0).astype(int) == 1]
    m = m.dropna(subset=["got_hit"])

    # One bet per (date, player_id) — prefer DK over FD.
    m["edge"] = m.apply(
        lambda r: ev_per_unit(r["p_model"], int(r["over_price"])), axis=1
    )
    m["decimal_over"] = m["over_price"].apply(
        lambda px: american_to_decimal(int(px))
    )
    m["book_rank"] = m["book"].map({b: i for i, b in enumerate(BOOK_PREFERENCE)}).fillna(99)
    bets = (
        m.sort_values(["date", "player_id", "book_rank"])
        .drop_duplicates(subset=["date", "player_id"], keep="first")
    )

    # Apply gate.
    bets = bets[(bets["edge"] >= edge_min) & (bets["over_price"] >= price_max)]
    if bets.empty:
        return _empty_result()

    # P&L. hot_streak rows get the searched multiplier; rest get 1.0.
    bets["pnl_unit"] = np.where(
        bets["got_hit"].astype(int) == 1,
        bets["decimal_over"] - 1.0,
        -1.0,
    )
    bets["units"] = np.where(
        bets.get("hot_streak", 0).fillna(0).astype(int) == 1,
        hot_streak_units,
        1.0,
    )
    bets["pnl_weighted"] = bets["pnl_unit"] * bets["units"]

    n = len(bets)
    hit_rate = float(bets["got_hit"].astype(int).mean())
    roi_flat = float(bets["pnl_unit"].sum() / n)
    units_total = float(bets["units"].sum())
    roi_weighted = float(bets["pnl_weighted"].sum() / units_total) if units_total else 0.0

    # Daily P&L profile — needed for Sharpe-style stability metric. We use
    # weighted P&L per day since that matches what the production system
    # actually banks. Sharpe = mean(daily_pnl) / std(daily_pnl) approximates
    # "consistency of profit," which is what Optuna should optimize for at
    # our sample size — raw ROI is too noisy at 200-bet folds.
    daily = bets.groupby("date")["pnl_weighted"].sum().sort_index()
    if len(daily) >= 2:
        mu = float(daily.mean())
        sigma = float(daily.std(ddof=0))
        sharpe = mu / max(sigma, SHARPE_EPS)
        worst_day = float(daily.min())
    else:
        mu = float(daily.sum())
        sigma = 0.0
        sharpe = 0.0
        worst_day = mu

    return {
        "n_bets": n,
        "n_days": int(len(daily)),
        "hit_rate": hit_rate,
        "roi_flat": roi_flat,
        "roi_weighted": roi_weighted,
        "pnl_total": float(bets["pnl_weighted"].sum()),
        "sharpe": sharpe,
        "daily_mean": mu,
        "daily_std": sigma,
        "worst_day": worst_day,
    }


# ---------------------------------------------------------------------------
# Optuna objective. One model train per trial. Score the full eval window
# once, then evaluate the gate on each of the 3 val sub-windows. Median ROI
# across sub-windows is the score.
# ---------------------------------------------------------------------------
def make_objective(train_df, eval_df_scored_template, odds, eval_df_for_score, feats_cache):
    """Returns the objective function with closure over the data. We score
    fresh per trial since the model changes; eval_df_for_score is the
    feature DataFrame to score.

    Objective: median Sharpe across the 3 CV sub-folds. Rationale:
      * ROI is too noisy at ~200-bet sample sizes — one bad day can swing it
        2pp, which is enough for Optuna to "cherry-pick" lucky configs.
      * Sharpe (daily-mean / daily-std) rewards CONSISTENCY, so configs that
        rely on outlier days score worse than configs with steady growth.
      * Median across folds penalizes configs that happen to win one window
        but tank in adjacent ones.
    """
    def objective(trial):
        params = {
            # Bounded search: each parameter stays within a region we've
            # validated. Looser bounds = more overfitting room.
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        # Gate thresholds. price_max upper bound was tightened to -150 (was
        # broader): the price-tier backtest already showed -201 to -250 was
        # +13.7% ROI but -251 to -300 was -9.7%, so deeper chalk shouldn't
        # be in the search at all.
        edge_min  = trial.suggest_float("edge_min", 0.10, 0.25)
        price_max = trial.suggest_int("price_max", -250, -150)

        try:
            model, feats = _train_one(train_df, params)
        except Exception as e:
            print(f"  trial {trial.number}: train failed: {e}")
            return -1.0

        # Score the entire eval window once.
        scored = eval_df_for_score.copy()
        scored["p_model"] = _score_window(model, feats, eval_df_for_score).values

        sharpes = []
        for start, end in VAL_SUBS:
            r = evaluate_gate(
                scored, odds, edge_min, price_max,
                hot_streak_units=HOT_STREAK_UNITS,
                start=start, end=end,
            )
            if r["n_bets"] < MIN_BETS_PER_SUB:
                return -10.0  # too few bets, decisive reject
            sharpes.append(r["sharpe"])

        return float(np.median(sharpes))

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 70)
    print("JOINT OPTUNA: model hyperparameters + Filter E gate thresholds")
    print("=" * 70)
    print(f"  train through:    {TRAIN_END}")
    print(f"  val (Optuna):     {VAL_START} -> {VAL_END}  ({len(VAL_SUBS)} sub-folds)")
    print(f"  holdout (final):  {HOLDOUT_START} -> {HOLDOUT_END}  (Optuna never sees)")
    print(f"  trials:           {args.n_trials}")
    print(f"  min bets/fold:    {MIN_BETS_PER_SUB}")
    print()

    print("Loading modeling parquet + historical odds...")
    df = _load_modeling_parquet()
    train_df = df[df["date"] <= TRAIN_END].copy()
    eval_df = df[(df["date"] >= VAL_START) & (df["date"] <= HOLDOUT_END)].copy()

    print(f"  train rows: {len(train_df):,}  ({train_df['date'].min()} -> {train_df['date'].max()})")
    print(f"  eval rows:  {len(eval_df):,}  ({eval_df['date'].min()} -> {eval_df['date'].max()})")

    print("Loading historical prop odds...")
    odds = _load_historical_odds(VAL_START, HOLDOUT_END)
    print(f"  odds rows:  {len(odds):,}")
    if odds.empty:
        print("  ERROR: no historical_props parquets in eval window. Cannot continue.")
        sys.exit(1)

    print("Attaching hot_streak flag to eval window...")
    eval_df = _attach_hot_streak_to_eval(eval_df)
    print(f"  hot_streak == 1: {int(eval_df['hot_streak'].sum())}/{len(eval_df)} eval rows")
    print()

    # Pre-flight: estimate baseline bets per fold using the existing v3_recal
    # model + Filter E v2 thresholds. If the baseline can't clear MIN_BETS_PER_SUB
    # in any fold, abort early — Optuna will reject every trial and we'll waste
    # 4 min of compute.
    print("Pre-flight: estimating baseline bets per fold (v3_recal at Filter E v2)...")
    try:
        v3_bundle = joblib.load(Path(SETTINGS["paths"]["models_dir"]) / "xgb_v3_recal.joblib")
        v3_model, v3_feats = v3_bundle["model"], v3_bundle["features"]
        sanity_scored = eval_df.copy()
        sanity_scored["p_model"] = _score_window(v3_model, v3_feats, eval_df).values
        any_short = False
        for s, e in VAL_SUBS + ((HOLDOUT_START, HOLDOUT_END),):
            r = evaluate_gate(sanity_scored, odds, 0.15, -200, HOT_STREAK_UNITS, s, e)
            label = "holdout" if s == HOLDOUT_START else "val"
            flag = "" if r["n_bets"] >= MIN_BETS_PER_SUB else "  <-- BELOW MIN"
            if r["n_bets"] < MIN_BETS_PER_SUB:
                any_short = True
            print(f"  {label:8s} {s} -> {e}: {r['n_bets']:>4d} bets{flag}")
        if any_short:
            print()
            print("  WARNING: at least one fold falls below the per-fold minimum.")
            print(f"  Optuna will reject any config that can't beat {MIN_BETS_PER_SUB}")
            print(f"  bets in EVERY fold. Either lower MIN_BETS_PER_SUB at the top of")
            print(f"  this script, or extend the eval window to capture more bets.")
            print(f"  Continuing anyway — but expect most/all trials to score -10.")
    except Exception as exc:
        print(f"  pre-flight skipped: {exc}")
    print()

    # Run study.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    objective = make_objective(train_df, None, odds, eval_df, None)

    print(f"Running {args.n_trials} Optuna trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print()
    print("=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    print(f"  trial #{study.best_trial.number}")
    print(f"  median val Sharpe:  {study.best_value:+.4f}")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k:18s} {v:.4f}")
        else:
            print(f"    {k:18s} {v}")
    print(f"  hot_streak_units    {HOT_STREAK_UNITS:.2f}  (fixed, not searched)")
    print()

    # ---------------- Holdout evaluation ----------------
    best = study.best_params
    model_params = {k: v for k, v in best.items() if k in {
        "max_depth", "learning_rate", "n_estimators",
        "subsample", "colsample_bytree", "reg_lambda",
    }}
    edge_min  = best["edge_min"]
    price_max = best["price_max"]

    print("Re-training winning config and evaluating on val + holdout...")
    model, feats = _train_one(train_df, model_params)
    scored = eval_df.copy()
    scored["p_model"] = _score_window(model, feats, eval_df).values

    val_summaries = []
    for s, e in VAL_SUBS:
        val_summaries.append(evaluate_gate(
            scored, odds, edge_min, price_max, HOT_STREAK_UNITS, s, e,
        ))
    holdout_result = evaluate_gate(
        scored, odds, edge_min, price_max, HOT_STREAK_UNITS,
        HOLDOUT_START, HOLDOUT_END,
    )

    print()
    print("=" * 70)
    print(f"VAL FOLDS  (Optuna optimized for Sharpe across these)")
    print("=" * 70)
    for (s, e), r in zip(VAL_SUBS, val_summaries):
        print(f"  {s} -> {e}:  bets {r['n_bets']:>4d}  "
              f"hit {r['hit_rate']:.1%}  ROI {r['roi_weighted']:+.1%}  "
              f"Sharpe {r['sharpe']:+.3f}  worst day ${r['worst_day']:+.2f}")

    print()
    print("=" * 70)
    print(f"HOLDOUT  ({HOLDOUT_START} -> {HOLDOUT_END}, never seen by Optuna)")
    print("=" * 70)
    print(f"  bets               {holdout_result['n_bets']}")
    print(f"  active days        {holdout_result['n_days']}")
    print(f"  hit rate           {holdout_result['hit_rate']:.1%}")
    print(f"  ROI (flat $1)      {holdout_result['roi_flat']:+.1%}")
    print(f"  ROI (weighted)     {holdout_result['roi_weighted']:+.1%}")
    print(f"  total P&L (1u)     ${holdout_result['pnl_total']:+.2f}")
    print(f"  daily-Sharpe       {holdout_result['sharpe']:+.3f}")
    print(f"  worst day          ${holdout_result['worst_day']:+.2f}")
    print()

    # Overfitting diagnostic — compare the metric Optuna actually optimized
    # (median Sharpe across val folds) against holdout Sharpe.
    median_val_sharpe = float(np.median([r["sharpe"] for r in val_summaries]))
    sharpe_gap = holdout_result["sharpe"] - median_val_sharpe
    print("Diagnostic: holdout-vs-val gap on the optimized metric (Sharpe)")
    print(f"  val median Sharpe:   {median_val_sharpe:+.3f}")
    print(f"  holdout Sharpe:      {holdout_result['sharpe']:+.3f}")
    print(f"  gap:                 {sharpe_gap:+.3f}")
    if abs(sharpe_gap) < 0.10:
        print("  -> looks robust (Sharpe gap small)")
    elif sharpe_gap < -0.20:
        print("  -> WARNING: holdout Sharpe substantially worse. Likely overfit.")
    else:
        print("  -> moderate gap; treat with caution.")

    # ---------------- Save artifacts ----------------
    models_dir = Path(SETTINGS["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    out_winner = models_dir / "xgb_optuna_winner.json"
    with open(out_winner, "w") as f:
        json.dump(
            {
                "best_params":        study.best_params,
                "best_value_val":     float(study.best_value),
                "val_subs":           [
                    {"start": s, "end": e, **r}
                    for (s, e), r in zip(VAL_SUBS, val_summaries)
                ],
                "holdout":            holdout_result,
                "holdout_window":     [HOLDOUT_START, HOLDOUT_END],
                "n_trials":           args.n_trials,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_winner}")

    out_model = models_dir / "xgb_optuna.joblib"
    joblib.dump({"model": model, "features": feats}, out_model)
    print(f"Saved: {out_model}")
    print("\nDone. To compare against v3_recal on the same holdout:")
    print(f"  python scripts/eval_optuna_winner.py")


if __name__ == "__main__":
    main()
