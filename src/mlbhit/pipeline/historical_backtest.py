"""Compute historical ROI from historical odds + model predictions + outcomes.

Inputs (all already on disk by the time you run this):
    data/raw/historical_props/YYYY-MM-DD_props.parquet  (fetch_historical_odds)
    data/modeling/player_game_features.parquet          (build_features)
    data/clean/boxscores_YYYY.parquet                   (fetch_boxscores)

Flow per historical date:
  1. Load historical odds for that day
  2. Load model features for the same (player_id, date) rows
  3. Run the trained model on those features to get p_model
  4. Join odds + predictions + outcomes
  5. Compute edge per row; simulate flat-stake bets on edge >= threshold

Usage:
    python -m mlbhit.pipeline.historical_backtest --start 2026-03-20 --end 2026-04-23
    python -m mlbhit.pipeline.historical_backtest --start 2026-03-20 --end 2026-04-23 --edge-min 0.10
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import clean_path, modeling_path, raw_path
from ..features.recent_form import attach_hot_streak, attach_opp_grind
from ..model.predict import predict
from ..utils.odds_math import american_to_decimal, ev_per_unit
from .fetch_historical_odds import _date_path as historical_odds_path


def _odds_path_for_date(d: date) -> Path | None:
    """Return the on-disk path for a date's props, preferring the historical
    backfill location and falling back to the live-fetch location.

    The live `fetch_prop_odds` writes to `data/raw/{date}_props.parquet`, while
    `fetch_historical_odds` writes to `data/raw/historical_props/{date}_props.parquet`.
    Either is fine for backtesting — same schema — so accept whichever exists.
    """
    p_hist = historical_odds_path(d)
    if p_hist.exists():
        return p_hist
    p_live = raw_path("", f"{d.isoformat()}_props.parquet")
    if p_live.exists():
        return p_live
    return None


def _load_historical_odds(start: date, end: date) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        p = _odds_path_for_date(d)
        if p is not None:
            frames.append(pd.read_parquet(p))
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def _score_historical_features(dates: list[str], model_name: str | None = None) -> pd.DataFrame:
    """Load modeling feature rows for the given dates and score with the model.

    The modeling parquet already has features computed with leakage-safe
    rolling windows (closed='left'), so scoring is honest — no look-ahead.
    Pass `model_name` to override the default (e.g. "xgb_v2" to A/B compare).

    Also surfaces the columns Filter E needs (`home_away`, `batter_hand`,
    `pitcher_hand`, `pitcher_low_sample`) and derives `platoon_advantage` /
    `pitcher_features_known` to match score_today.py's logic exactly.
    """
    feat = pd.read_parquet(modeling_path("player_game_features.parquet"))
    feat = feat[feat["date"].astype(str).isin(dates)].copy()
    if feat.empty:
        return pd.DataFrame()
    feat["p_model"] = (predict(feat) if model_name is None
                       else predict(feat, name=model_name)).values
    feat["player_id"] = feat["player_id"].astype("Int64")

    # Derive Filter E gating columns on-the-fly so we don't depend on the
    # modeling parquet being regenerated. Mirrors score_today.py.
    bh = feat.get("batter_hand")
    ph = feat.get("pitcher_hand")
    if bh is not None and ph is not None:
        ph_known = ph.notna()
        feat["platoon_advantage"] = (
            ph_known & ((bh == "S") | (bh != ph))
        ).astype(int)
        feat["pitcher_features_known"] = ph_known.astype(int)
    else:
        feat["platoon_advantage"] = 0
        feat["pitcher_features_known"] = 0

    keep = [
        "date", "player_id", "p_model", "got_hit",
        "home_away", "batter_hand", "pitcher_hand",
        "platoon_advantage", "pitcher_features_known",
        # Carrying `opponent` through so backtest can compute opp-grind
        # streaks (recent_form.attach_opp_grind needs opp_team + date).
        "opponent",
    ]
    keep = [c for c in keep if c in feat.columns]
    out = feat[keep].drop_duplicates(subset=["date", "player_id"], keep="first")
    if "opponent" in out.columns:
        out = out.rename(columns={"opponent": "opp_team"})
    return out


def backtest(
    start: date,
    end: date,
    edge_min: float = 0.05,
    stake: float = 1.0,
    book_preference: tuple[str, ...] = ("draftkings", "fanduel"),
    model_name: str | None = None,
    filter_e: bool = False,
    require_pitcher: bool = False,
    price_max_negative: int = -200,
) -> pd.DataFrame:
    """Score historical odds + features and report ROI.

    `filter_e=True` applies the Filter E v2 gate identically to the one in
    recommend.py: edge >= 15% AND price >= -200. When enabled, this also
    overrides `edge_min` to 0.15 internally so the threshold matches the
    Filter E baseline (raised from 0.08 to 0.15 — the high-edge tail
    delivered better ROI than the looser gate; this is the target).
    v2 (2026-04-27) dropped the (away OR platoon) requirement that v1 had.

    NOTE: the `start_rate` projected-lineup gate from recommend.py has no
    effect here — backtest data is built from boxscores, so every row IS
    a confirmed actual starter. The gate is forward-looking only.

    `require_pitcher=True` drops rows where pitcher_hand was unknown at
    feature-build time. The xgb_v3 backtest assumed pitcher features were
    always present; running the live pipeline against TBD-starter games means
    pitcher features are league means and p_model is effectively a pure
    batter-quality bet — not what was validated. Strongly recommended for
    apples-to-apples comparison with the published ROI baseline.
    """
    if filter_e:
        edge_min = max(edge_min, 0.15)

    odds = _load_historical_odds(start, end)
    if odds.empty:
        print(f"No historical odds in {start} -> {end}. Run fetch_historical_odds first.")
        return pd.DataFrame()

    dates = sorted(odds["date"].astype(str).unique().tolist())
    preds = _score_historical_features(dates, model_name=model_name)
    if preds.empty:
        print("No modeling rows match the historical odds dates. "
              "Did you run build_features with the relevant seasons?")
        return pd.DataFrame()

    if require_pitcher and "pitcher_features_known" in preds.columns:
        before = len(preds)
        preds = preds[preds["pitcher_features_known"].astype(int) == 1]
        print(f"  --require-pitcher dropped {before - len(preds)}/{before} prediction rows "
              f"(pitcher_hand was NaN — features fell back to league means).")

    m = odds.merge(preds, on=["date", "player_id"], how="inner")

    m["edge_over"] = m.apply(
        lambda r: ev_per_unit(r["p_model"], int(r["over_price"]))
        if pd.notna(r["over_price"]) else np.nan,
        axis=1,
    )
    m["decimal_over"] = m["over_price"].apply(
        lambda px: american_to_decimal(int(px)) if pd.notna(px) else np.nan
    )

    # One bet per (date, player_id). Prefer DK when both books have a line;
    # otherwise fall back to FD. This matches your real-world line-shop.
    m["book_rank"] = m["book"].map({b: i for i, b in enumerate(book_preference)}).fillna(99)
    one_per_player = (
        m.sort_values(["date", "player_id", "book_rank"])
        .drop_duplicates(subset=["date", "player_id"], keep="first")
    )

    base_mask = (
        (one_per_player["edge_over"] >= edge_min)
        & one_per_player["got_hit"].notna()
    )

    if filter_e:
        # Filter E v2 (2026-04-27): edge >= 15% (already in base_mask) and
        # price >= -200 (chalk ceiling). v1 also required (away OR platoon);
        # that gate was dropped to widen the slate. Away/platoon counts are
        # still printed below for slate-composition visibility.
        try:
            price_int = one_per_player["over_price"].astype("Int64")
            price_mask = price_int >= price_max_negative
        except Exception:
            price_mask = one_per_player["over_price"].apply(
                lambda px: pd.notna(px) and int(px) >= price_max_negative
            )
        bets_mask = base_mask & price_mask
    else:
        bets_mask = base_mask

    bets = one_per_player[bets_mask].copy()
    if bets.empty:
        gate = "Filter E" if filter_e else f"edge >= {edge_min:.0%}"
        print(f"No bets cleared {gate} with outcomes.")
        return bets

    bets["pnl"] = np.where(
        bets["got_hit"].astype(int) == 1,
        stake * (bets["decimal_over"] - 1),
        -stake,
    )

    # Attach recent-form sizing/emphasis signals from boxscores. These DON'T
    # affect bet selection (already done) — they label each bet so we can
    # report unit-weighted P&L (hot_streak → 2x stake) and cohort breakdowns
    # (opp_grind → 11+ consecutive games for the opp). Loads boxscore parquets
    # for every season touched by the backtest window.
    seasons = sorted({pd.to_datetime(d).year for d in bets["date"]})
    box_frames = []
    for yr in seasons:
        p = clean_path(f"boxscores_{yr}.parquet")
        if p.exists():
            box_frames.append(pd.read_parquet(p))
    if box_frames:
        boxscores_all = pd.concat(box_frames, ignore_index=True)
        bets = attach_hot_streak(bets, boxscores_all)
        if "opp_team" in bets.columns:
            bets = attach_opp_grind(bets, boxscores_all)
    else:
        # No boxscores on disk → can't compute signals; backfill safe defaults
        # so downstream weighted-P&L math still works (multiplier becomes 1.0).
        bets["hot_streak"] = 0
        bets["recommended_units"] = 1.0
        bets["opp_grind"] = 0
        bets["opp_consec_games"] = 0

    bets["pnl_weighted"] = bets["pnl"] * bets["recommended_units"]
    total_units_staked = (stake * bets["recommended_units"]).sum()

    n = len(bets)
    hit_rate = bets["got_hit"].astype(int).mean()
    roi = bets["pnl"].sum() / (stake * n)
    avg_price = bets["over_price"].astype(int).mean()
    avg_edge = bets["edge_over"].mean()
    avg_p = bets["p_model"].mean()

    label = "FILTER E" if filter_e else f"edge>={edge_min:.0%}"
    print("=" * 60)
    print(f"HISTORICAL BACKTEST  {start} -> {end}  ({label})")
    print("=" * 60)
    print(f"  model             {model_name or 'default'}")
    if filter_e:
        print(f"  filter            edge>={edge_min:.0%} & price>={price_max_negative}  (Filter E v2)")
    else:
        print(f"  edge threshold    {edge_min:.0%}")
    print(f"  require_pitcher   {require_pitcher}")
    print(f"  stake             ${stake:.2f} flat per bet")
    print(f"  bets placed       {n}")
    print(f"  bets per day      {n / max(1, len(dates)):.1f}")
    print(f"  hit rate          {hit_rate:.1%}")
    print(f"  avg p_model       {avg_p:.3f}")
    print(f"  avg edge          {avg_edge:+.1%}")
    print(f"  avg price         {avg_price:+.0f}")
    print(f"  total P&L         ${bets['pnl'].sum():+.2f}")
    print(f"  ROI               {roi:+.1%}")
    if filter_e:
        away_n = int((bets["home_away"].astype(str) == "A").sum()) if "home_away" in bets.columns else 0
        plat_n = int((bets["platoon_advantage"].fillna(0).astype(int) == 1).sum()) if "platoon_advantage" in bets.columns else 0
        both_n = int(((bets["home_away"].astype(str) == "A") & (bets["platoon_advantage"].fillna(0).astype(int) == 1)).sum()) if {"home_away","platoon_advantage"}.issubset(bets.columns) else 0
        print(f"  away bets         {away_n}  ({away_n / n:.1%})")
        print(f"  platoon bets      {plat_n}  ({plat_n / n:.1%})")
        print(f"  away+platoon      {both_n}  ({both_n / n:.1%})")

    # Cohort breakdown: hot bats, opp-grind, and combos. Each cohort prints
    # its own bet count, hit rate, ROI per bet (flat $1 stake) so we can
    # see whether the signals are independently positive before treating
    # them as sizing levers.
    def _cohort(name: str, mask: pd.Series) -> None:
        m = mask & bets["got_hit"].notna()
        sub = bets[m]
        if len(sub) == 0:
            print(f"  {name:18s} 0 bets (skipped)")
            return
        sub_pnl = sub["pnl"].sum()
        sub_roi = sub_pnl / (stake * len(sub))
        sub_hr = sub["got_hit"].astype(int).mean()
        print(f"  {name:18s} {len(sub):>4d} bets  hit {sub_hr:.1%}  P&L ${sub_pnl:+7.2f}  ROI {sub_roi:+.1%}")

    print()
    print(f"COHORT BREAKDOWN  (flat $1 stake, all cohorts)")
    if "hot_streak" in bets.columns:
        _cohort("hot bats (>.300/6g)", bets["hot_streak"] == 1)
        _cohort("cold bats",            bets["hot_streak"] == 0)
    if "opp_grind" in bets.columns:
        _cohort("opp grind (11+ G)",    bets["opp_grind"] == 1)
        _cohort("opp rested (<=10 G)",  bets["opp_grind"] == 0)
    if {"hot_streak","opp_grind"}.issubset(bets.columns):
        _cohort("hot + grind combo",   (bets["hot_streak"] == 1) & (bets["opp_grind"] == 1))

    # Price-tier breakdown — answers "where does chalk earn vs lose?". The
    # singles-only structure on chalk is brutal (lose 1u, win <0.40u), so a
    # cohort that's only marginally +EV at -250 is much riskier than the
    # same +EV at -130. If you're loosening --price-min beyond -200, this
    # is the table to read.
    _price = bets["over_price"].astype("Int64")
    print()
    print(f"PRICE-TIER BREAKDOWN  (flat $1 stake, all in this run's bet pool)")
    _cohort("+odds (underdog)",   _price > 0)
    _cohort("-100 to -150",       (_price <= -100) & (_price >= -150))
    _cohort("-151 to -200",       (_price <= -151) & (_price >= -200))
    _cohort("-201 to -250",       (_price <= -201) & (_price >= -250))
    _cohort("-251 to -300",       (_price <= -251) & (_price >= -300))
    _cohort("worse than -300",    _price <= -301)

    # Unit-weighted summary (hot_streak triggers 2x stake by default — see
    # recent_form.attach_hot_streak for the multiplier). This is the headline
    # number to compare against the flat-stake ROI above when deciding whether
    # to actually bet more on hot bats live.
    if "pnl_weighted" in bets.columns and total_units_staked > 0:
        weighted_pnl = bets["pnl_weighted"].sum()
        weighted_roi = weighted_pnl / total_units_staked
        units_n_2x = int((bets["recommended_units"] >= 2.0 - 1e-9).sum())
        print()
        print(f"UNIT-WEIGHTED SIZING  (2x stake on hot bats, 1x otherwise)")
        print(f"  bets at 2x stake  {units_n_2x:>4d}  ({units_n_2x / n:.1%})")
        print(f"  total units staked  {total_units_staked:.1f}u")
        print(f"  total P&L           ${weighted_pnl:+.2f}")
        print(f"  ROI per unit        {weighted_roi:+.1%}")

    # Per-day breakdown
    by_day = bets.groupby("date").agg(
        bets=("pnl", "size"),
        hit=("got_hit", "sum"),
        pnl=("pnl", "sum"),
    )
    by_day["roi"] = by_day["pnl"] / (stake * by_day["bets"])
    print()
    print("Per-day P&L:")
    print(by_day.to_string())

    return bets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--edge-min", type=float, default=0.05)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. xgb_v2, xgb_v3). Defaults to predict.DEFAULT_MODEL.")
    parser.add_argument(
        "--filter-e",
        action="store_true",
        help=(
            "Apply Filter E v2: edge>=15%% & price>=-200. Identical gate to "
            "recommend.py --filter-e (the 2026-04-27 change dropped the "
            "(away OR platoon) clause that v1 required)."
        ),
    )
    parser.add_argument(
        "--require-pitcher",
        action="store_true",
        help=(
            "Drop rows where pitcher_hand is unknown (matches recommend.py's "
            "--require-pitcher; the validated backtest had pitcher features "
            "present, so this is the apples-to-apples comparison)."
        ),
    )
    parser.add_argument(
        "--price-min",
        type=int,
        default=-200,
        help=(
            "Lowest American price allowed when --filter-e is set (default -200). "
            "Use a more-negative number to admit chalkier picks: e.g. --price-min -250 "
            "lets in -201..-250, --price-min -1000 effectively removes the cap. "
            "The price-tier cohort breakdown in the output shows where chalk earns/loses."
        ),
    )
    args = parser.parse_args()

    backtest(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        edge_min=args.edge_min,
        stake=args.stake,
        model_name=args.model,
        filter_e=args.filter_e,
        require_pitcher=args.require_pitcher,
        price_max_negative=args.price_min,
    )
