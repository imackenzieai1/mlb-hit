from __future__ import annotations

from datetime import date

import pandas as pd

from datetime import datetime, timezone

from ..config import SETTINGS
from ..io import output_path, raw_path
from ..utils.odds_math import ev_per_unit, prob_to_american

EDGE_MIN = SETTINGS["edge_threshold"]

# Game statuses that indicate first pitch has been thrown (or the game is
# over). The model predicts P(1+ hit | full game, ~4 PAs). Once first pitch
# is thrown, books re-price to P(1+ hit | remaining PAs), which produces
# phantom edges (model thinks the batter still has 4 PAs when in reality he
# might have 1). Backtest was pre-game only — betting in-game lines is NOT
# what was validated.
#
# NOTE: "Delayed Start" means first pitch is delayed (hasn't happened yet),
# so it is NOT in this set. "Warmup", "Pre-Game", and "Scheduled" are also
# pre-first-pitch.
IN_PROGRESS_STATUSES = {
    "In Progress",
    "Final",
    "Game Over",
    "Completed Early",
    "Completed",
    "Suspended",
    "Delayed",  # mid-game pause
    "Manager challenge",  # only happens during a play
}

# Games that won't be played on the target date. Their players' props will
# be voided by the book, so we drop them from the bet list — but a postponed
# game does NOT mean the rest of the slate is unbettable.
POSTPONED_STATUSES = {"Postponed"}

# Filter E thresholds — v1 settings (Ian, 2026-04-25):
#   * Edge floor raised from 8% to 15%. The 8% gate produced +14.1% ROI on
#     YTD with ~28 bets/day; the high-edge tail (>=20%) hit +15.6%. 15% is
#     the sweet spot between sample and conviction — fewer, surer bets.
#   * OR gate retained: away OR platoon advantage. Both halves contribute
#     positive YTD ROI individually; the OR keeps the slate diversified.
#   * Price ceiling unchanged at -200.
#   * Projected lineups (lineup_confirmed=False) additionally require
#     start_rate >= FILTER_E_PROJECTED_MIN_START_RATE — see below.
# Re-tune these after each meaningful model bump or prop-market shift.
FILTER_E_EDGE_MIN = 0.15
FILTER_E_PRICE_MIN = -200  # American: anything >= -200 (i.e. -150, -100, +120, ...)

# Minimum recent start frequency for a projected (un-confirmed) lineup row to
# clear Filter E. 0.80 = "started 80%+ of the team's last 14 games" — i.e.
# essentially a regular. 0.50 was the projection threshold (see
# project_lineups.START_THRESHOLD); 0.80 is the betting threshold. Confirmed
# rows ignore this gate entirely (start_rate is NA for them).
FILTER_E_PROJECTED_MIN_START_RATE = 0.80


def _passes_filter_e(row) -> bool:
    """Score-and-odds-side gate for Filter E.

    score_today.py already exposes `platoon_or_away` (away OR platoon advantage).
    Here we add the odds-dependent half: edge floor + price ceiling on chalk +
    a start-rate floor for projected (unconfirmed) lineup rows.
    """
    if pd.isna(row.get("edge")) or pd.isna(row.get("over_price")):
        return False
    if row["edge"] < FILTER_E_EDGE_MIN:
        return False
    try:
        if int(row["over_price"]) < FILTER_E_PRICE_MIN:
            return False
    except (TypeError, ValueError):
        return False

    # Projected-lineup gate: only bet projected rows when the player is a
    # near-everyday regular. Confirmed rows have start_rate=NA and skip this.
    lineup_confirmed = bool(row.get("lineup_confirmed", True))
    if not lineup_confirmed:
        sr = row.get("start_rate")
        try:
            if pd.isna(sr) or float(sr) < FILTER_E_PROJECTED_MIN_START_RATE:
                return False
        except (TypeError, ValueError):
            return False

    away = str(row.get("home_away", "")) == "A"
    plat = int(row.get("platoon_advantage", 0) or 0) == 1
    return away or plat


def _postponed_game_pks(target_date: date) -> set[int]:
    """Return the set of game_pks marked Postponed for the target date."""
    sched_path = raw_path("schedule", f"{target_date.isoformat()}.parquet")
    if not sched_path.exists():
        return set()
    try:
        sched = pd.read_parquet(sched_path)
    except Exception:
        return set()
    if "status" not in sched.columns or "game_pk" not in sched.columns:
        return set()
    bad = sched[sched["status"].isin(POSTPONED_STATUSES)]
    return set(pd.to_numeric(bad["game_pk"], errors="coerce").dropna().astype(int).tolist())


def _pregame_game_pks(target_date: date) -> set[int] | None:
    """Return the set of game_pks for `target_date` whose first pitch is still
    in the future AND whose status is not in-progress/finished/postponed.

    Returns None if we can't determine pre-game state (no schedule on disk, no
    timestamps to read) — the caller should treat None as "don't filter."
    """
    sched_path = raw_path("schedule", f"{target_date.isoformat()}.parquet")
    if not sched_path.exists():
        return None
    try:
        sched = pd.read_parquet(sched_path)
    except Exception:
        return None
    if "game_pk" not in sched.columns:
        return None

    if "status" in sched.columns:
        skip = IN_PROGRESS_STATUSES | POSTPONED_STATUSES
        sched = sched[~sched["status"].isin(skip)]

    if "game_datetime" in sched.columns:
        try:
            gdt = pd.to_datetime(sched["game_datetime"], utc=True, errors="coerce")
            now_utc = datetime.now(tz=timezone.utc)
            sched = sched[gdt > now_utc]
        except Exception:
            return None

    return set(pd.to_numeric(sched["game_pk"], errors="coerce").dropna().astype(int).tolist())


def recommend(
    predictions: pd.DataFrame,
    prop_prices: pd.DataFrame | None = None,
    filter_e: bool = False,
    require_pitcher: bool = False,
    require_confirmed_lineup: bool = False,
    drop_postponed_for_date: date | None = None,
    pre_game_only_for_date: date | None = None,
) -> pd.DataFrame:
    """Rank +EV recommendations, optionally with Filter E + starter-known gates.

    `require_pitcher=True` drops any pick where `pitcher_features_known != 1`.
    When the probable starter isn't announced, every pitcher feature is the
    league mean — meaning p_model has zero pitcher signal. Betting in that
    state is effectively a pure batter-quality bet, which is not what the
    backtest validated. Strongly recommended for live (today) recommendations.

    `require_confirmed_lineup=True` drops any pick whose `lineup_confirmed != 1`.
    Default False because projected-lineup bets either play (book gives action)
    or don't (book voids), with no PnL exposure either way — so by default we
    let the bigger projected slate through. Pass True if you only want to bet
    rows where MLB has actually posted the starting lineup.
    """
    preds = predictions.copy()
    preds["fair_american"] = preds["p_model"].apply(prob_to_american)

    # Drop predictions for any postponed game on the target date — those props
    # will be voided by the book, so picking them is wasted edge.
    if drop_postponed_for_date is not None and "game_pk" in preds.columns:
        ppt = _postponed_game_pks(drop_postponed_for_date)
        if ppt:
            before = len(preds)
            preds = preds[~pd.to_numeric(preds["game_pk"], errors="coerce").isin(ppt)]
            print(f"  dropped {before - len(preds)} predictions from "
                  f"{len(ppt)} postponed game(s) on {drop_postponed_for_date}.")

    # Restrict to games whose first pitch is still in the future. Used for
    # mid-day re-runs on a partially-live slate, where some games are already
    # in progress (phantom-edge zone) but the late-window games are still
    # bettable. Filter is by game_pk, so it composes cleanly with the
    # postponed-drop above.
    if pre_game_only_for_date is not None and "game_pk" in preds.columns:
        keep = _pregame_game_pks(pre_game_only_for_date)
        if keep is None:
            print(f"  WARN: --pre-game-only requested but no schedule timestamps "
                  f"available for {pre_game_only_for_date}; not filtering.")
        else:
            before = len(preds)
            preds = preds[pd.to_numeric(preds["game_pk"], errors="coerce").isin(keep)]
            print(f"  --pre-game-only kept {len(preds)}/{before} rows "
                  f"({len(keep)} pre-first-pitch game(s) on {pre_game_only_for_date}).")

    if require_pitcher and "pitcher_features_known" in preds.columns:
        before = len(preds)
        preds = preds[preds["pitcher_features_known"].astype(int) == 1]
        dropped = before - len(preds)
        if dropped:
            print(f"  --require-pitcher dropped {dropped}/{before} rows "
                  f"(probable starter unknown — pitcher features were league means).")

    if require_confirmed_lineup and "lineup_confirmed" in preds.columns:
        before = len(preds)
        preds = preds[preds["lineup_confirmed"].fillna(False).astype(bool)]
        dropped = before - len(preds)
        if dropped:
            print(f"  --require-confirmed-lineup dropped {dropped}/{before} "
                  f"projected-lineup rows (set --require-confirmed-lineup off "
                  f"to bet projections too — books void unscratched players).")

    if prop_prices is None or prop_prices.empty:
        preds["edge"] = None
        preds["filter_e_pass"] = False
        return preds.head(25)

    m = preds.merge(prop_prices, on=["date", "player_id"], how="inner")
    m["edge"] = m.apply(lambda r: ev_per_unit(r["p_model"], int(r["over_price"])), axis=1)
    m["filter_e_pass"] = m.apply(_passes_filter_e, axis=1)

    if filter_e:
        m = m[m["filter_e_pass"]]
    else:
        m = m[m["edge"] >= EDGE_MIN]

    # Surface filter_e_pass at the top so it's visible when scanning the CSV.
    return m.sort_values(["filter_e_pass", "edge"], ascending=[False, False])


def _slate_state(target_date: date) -> tuple[str, list[str]]:
    """Inspect the schedule for `target_date` and return (state, reasons).

    state is one of:
      "PREGAME"  — every game is pre-first-pitch and target is today
      "PAST"     — target date is before today (safe to run, results settled)
      "LIVE"     — target is today AND at least one game has started
      "UNKNOWN"  — schedule parquet missing; caller decides
    """
    today_local = date.today()
    if target_date < today_local:
        return "PAST", []
    if target_date > today_local:
        return "PREGAME", []  # future slate: nothing has started

    sched_path = raw_path("schedule", f"{target_date.isoformat()}.parquet")
    if not sched_path.exists():
        return "UNKNOWN", [f"no schedule parquet at {sched_path}"]

    sched = pd.read_parquet(sched_path)
    # Exclude postponed games from the live-slate check — they're not played
    # today, so they can't be "in progress." They're handled separately by
    # dropping their batters from the bet list.
    if "status" in sched.columns:
        active_sched = sched[~sched["status"].isin(POSTPONED_STATUSES)]
        n_postponed = len(sched) - len(active_sched)
        sched = active_sched
    else:
        n_postponed = 0

    reasons: list[str] = []
    if n_postponed:
        # Informational only — does not block the slate.
        reasons.append(f"(info) {n_postponed} postponed game(s) excluded from live check")

    if "status" in sched.columns:
        bad = sched[sched["status"].isin(IN_PROGRESS_STATUSES)]
        if not bad.empty:
            reasons.append(
                f"{len(bad)}/{len(sched)} active games show in-progress/finished status: "
                f"{sorted(bad['status'].unique().tolist())}"
            )

    if "game_datetime" in sched.columns:
        try:
            gdt = pd.to_datetime(sched["game_datetime"], utc=True, errors="coerce")
            now_utc = datetime.now(tz=timezone.utc)
            started = gdt.dropna() <= now_utc
            if started.any():
                reasons.append(
                    f"{int(started.sum())}/{len(sched)} active games have first-pitch "
                    f"timestamp <= now ({now_utc.isoformat(timespec='minutes')})"
                )
        except Exception:
            pass

    # "(info)" entries are informational and don't trigger LIVE state on
    # their own. Block only when there's a real first-pitch reason.
    blocking = [r for r in reasons if not r.startswith("(info)")]
    return ("LIVE" if blocking else "PREGAME"), reasons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument(
        "--filter-e",
        action="store_true",
        help=(
            "Restrict to Filter E: edge>=15%% & price>=-200 & (away OR platoon "
            "advantage). Backtested at +15.9%% ROI / 69.2%% hit rate (xgb_v3_recal, "
            "2026-03-20 to 2026-04-23, 558 bets)."
        ),
    )
    parser.add_argument(
        "--require-pitcher",
        action="store_true",
        help=(
            "Drop picks where the probable starter is unknown (pitcher features "
            "fell back to league means). Recommended for live recommendations "
            "since the backtest assumed pitcher features were always present."
        ),
    )
    parser.add_argument(
        "--require-confirmed-lineup",
        action="store_true",
        help=(
            "Drop picks built from a PROJECTED lineup (i.e. MLB hasn't posted "
            "the actual lineup yet). Default off — projection lets the morning "
            "runner see the full slate; books void any prop where the player "
            "doesn't take a PA, so projected bets carry no extra downside."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "Cap rows printed to terminal (default 0 = print every passing bet). "
            "The CSV always contains the full filtered set."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the in-progress safety guard. Only use this if you know what "
            "you're doing — recommendations against in-game lines produce phantom "
            "edges because the model assumes ~4 PAs remaining."
        ),
    )
    parser.add_argument(
        "--pre-game-only",
        action="store_true",
        help=(
            "On a partially-live slate (some games started, others not yet), "
            "filter to ONLY games whose first pitch is still in the future. "
            "This is the safe way to do a mid-day re-run — the early window "
            "is dropped (phantom-edge risk), the late window stays bettable. "
            "Implies that the live-slate guard is satisfied for the kept games."
        ),
    )
    args = parser.parse_args()

    d = args.date or date.today().isoformat()
    target_d = date.fromisoformat(d)

    state, reasons = _slate_state(target_d)
    if state == "LIVE" and not (args.force or args.pre_game_only):
        print(f"\n{'!' * 60}")
        print(f"REFUSING TO RECOMMEND — slate for {d} is already live")
        for r in reasons:
            print(f"  - {r}")
        print("  Model predicts pre-game P(1+ hit | ~4 PAs); books re-price to")
        print("  P(1+ hit | remaining PAs) once first pitch is thrown. Edges seen")
        print("  in this state are PHANTOM and not bettable.")
        print("  Re-run before first pitch tomorrow, or pass --pre-game-only to")
        print("  bet only the games that haven't started yet, or --force for testing.")
        print(f"{'!' * 60}\n")
        raise SystemExit(2)
    if state == "LIVE" and args.force:
        print(f"  WARN: --force overriding live-slate guard ({len(reasons)} reasons).")
    elif state == "LIVE" and args.pre_game_only:
        print(f"  --pre-game-only: slate is partially live; will filter to "
              f"games still pre-first-pitch (live-slate reasons: "
              f"{len(reasons)}).")
    elif state == "PAST":
        print(f"  target date {d} is in the past — running historical validation mode.")
    elif state == "UNKNOWN":
        print(f"  WARN: schedule status unknown — proceeding without live-slate guard.")

    preds = pd.read_parquet(output_path("predictions", f"{d}.parquet"))

    # Pull today's prop odds. fetch_prop_odds writes to data/raw/props/;
    # the historical backfill writes to data/raw/historical_props/.
    prices = None
    for p in (raw_path("props", f"{d}_props.parquet"),
              raw_path("historical_props", f"{d}_props.parquet"),
              raw_path("", f"{d}_props.parquet")):  # legacy location, kept for safety
        if p.exists():
            prices = pd.read_parquet(p)
            prices["player_id"] = pd.to_numeric(prices["player_id"], errors="coerce").astype("Int64")
            preds["player_id"] = pd.to_numeric(preds["player_id"], errors="coerce").astype("Int64")
            print(f"  loaded {len(prices)} prop prices from {p}")
            break

    if prices is None:
        print(f"  WARN: no prop odds parquet found for {d}. "
              f"Run `python -m mlbhit.pipeline.fetch_prop_odds --date {d}` first.")
        if args.filter_e:
            print("  --filter-e requires odds; falling through to top picks by p_model only.")

    recs = recommend(
        preds,
        prop_prices=prices,
        filter_e=args.filter_e,
        require_pitcher=args.require_pitcher,
        require_confirmed_lineup=args.require_confirmed_lineup,
        drop_postponed_for_date=target_d,
        pre_game_only_for_date=target_d if args.pre_game_only else None,
    )
    label = "FILTER E" if args.filter_e else f"edge>={EDGE_MIN:.0%}"
    n_bets = len(recs)
    n_print = n_bets if args.top in (0, None) else min(args.top, n_bets)
    print(f"\n{'=' * 60}\nRECOMMENDATIONS  {d}  ({label})\n{'=' * 60}")
    print(f"  {n_bets} bets cleared the filter "
          f"({'all' if n_print == n_bets else f'top {n_print}'} shown below; "
          f"full list in CSV).")
    if n_bets:
        print(recs.head(n_print).to_string(index=False))
    suffix = "_filter_e" if args.filter_e else ""
    recs.to_csv(output_path("recommendations", f"{d}{suffix}.csv"), index=False)
