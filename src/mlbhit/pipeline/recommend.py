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

# Filter E thresholds — v3 settings (Ian, 2026-04-29):
#   * Edge floor 11.4%, price ceiling -240, start_rate floor 80% for projected
#     lineups, 2x sizing on hot bats (>0.300 BA over last 6 PA-counted games).
#   * v3 numbers come from scripts/optuna_joint.py — a 50-trial bounded TPE
#     search that jointly optimized model hyperparameters AND gate thresholds
#     against median Filter-E Sharpe across 3 val sub-windows, with a 7-day
#     held-out holdout never seen by the search. v5_recal is the corresponding
#     model (see model/predict.py). Holdout: 170 bets, 77.6% hit, +29.7% ROI,
#     daily-Sharpe 1.99 — strictly dominates v2.1 on the same window.
#   * Lineage of changes:
#       v1 (deprecated)  edge>=15%, price>=-200, (away OR platoon), start_rate>=0.80
#       v2  (2026-04-27) edge>=15%, price>=-200, start_rate>=0.80
#                         dropped the (away OR platoon) clause to widen slate.
#       v2.1 (2026-04-29) edge>=15%, price>=-250, start_rate>=0.80
#                         widened price band after price-tier backtest showed
#                         -201..-250 was +15.8% ROI / 79.7% hit (74 bets).
#                         This is the "prior production" preset — pair with
#                         xgb_v3_recal for the validated rollback config.
#       v3  (2026-04-29) edge>=11.4%, price>=-240, start_rate>=0.80
#                         Optuna joint search winner. Same start_rate gate
#                         (not part of Optuna's search). Pair with xgb_v5_recal.
# Re-tune these after each meaningful model bump or prop-market shift.

# --- Active production preset ---
FILTER_E_EDGE_MIN = 0.11445569939746027
FILTER_E_PRICE_MIN = -240

# Book preference for line dedupe. When the same player has prop odds from
# multiple books in the prop_prices parquet, the first book listed wins.
# Switched to FD-first 2026-05-02 (was DK-first since the project's start).
# historical_backtest.py and scripts/optuna_joint.py mirror this ordering.
BOOK_PREFERENCE: tuple[str, ...] = ("fanduel", "draftkings")

# Minimum recent start frequency for a projected (un-confirmed) lineup row to
# clear Filter E. 0.80 = "started 80%+ of the team's last 14 games" — i.e.
# essentially a regular. 0.50 was the projection threshold (see
# project_lineups.START_THRESHOLD); 0.80 is the betting threshold. Confirmed
# rows ignore this gate entirely (start_rate is NA for them). Not part of
# Optuna's search space — kept stable at 0.80 across both v2.1 and v3.
FILTER_E_PROJECTED_MIN_START_RATE = 0.80

# --- Named presets, available for explicit selection ---
# Pair (model, gate) presets. Default cron uses FILTER_E_PRESET_V3 (Optuna).
# The v2.1 entry preserves the "prior production no-qualifier" config so we
# can roll back model + gate together with one flag, not piecemeal — the v3
# Optuna gate is tighter on price ceiling (-240 vs -250) and looser on edge
# floor (11.4% vs 15%), so flipping just the model without the gate would
# be a configuration we never validated.
FILTER_E_PRESETS = {
    "v2_1": {
        "model":           "xgb_v3_recal",
        "edge_min":        0.15,
        "price_min":       -250,
        "start_rate_min":  0.80,
        "hot_streak_units": 2.0,
        "description":     ("Prior production (v2.1, 2026-04-29). Validated "
                            "in Filter E backtest 2026-03-20 -> 2026-04-23: "
                            "+15.9% ROI / 69.2% hit. Use for rollback."),
    },
    "v3": {
        "model":           "xgb_v5_recal",
        "edge_min":        0.11445569939746027,
        "price_min":       -240,
        "start_rate_min":  0.80,
        "hot_streak_units": 2.0,
        "description":     ("Active production (v3, 2026-04-29). Optuna "
                            "joint search winner. Holdout 2026-04-20 -> 26: "
                            "+29.7% ROI / 77.6% hit / Sharpe 1.99."),
    },
}
ACTIVE_FILTER_E_PRESET = "v3"


def _passes_filter_e(row, edge_floor: float | None = None,
                     price_floor: int | None = None) -> bool:
    """Score-and-odds-side gate for Filter E (v2).

    Gate components:
      * edge >= edge_floor (default FILTER_E_EDGE_MIN)
      * over_price >= price_floor (default FILTER_E_PRICE_MIN)
      * for projected (unconfirmed) lineups: start_rate >= 0.80
        (confirmed rows skip this — start_rate is NA there)

    v1 also required (away OR platoon_advantage); v2 removed that. The
    away/platoon columns are still surfaced in the CSV for ad-hoc filtering,
    just no longer a hard gate.

    edge_floor / price_floor parameters let callers override the production
    thresholds for ad-hoc analysis (e.g., comparing Optuna's preferred 0.12
    edge floor against production's 0.15) without mutating module constants.
    """
    if edge_floor is None:
        edge_floor = FILTER_E_EDGE_MIN
    if price_floor is None:
        price_floor = FILTER_E_PRICE_MIN

    if pd.isna(row.get("edge")) or pd.isna(row.get("over_price")):
        return False
    if row["edge"] < edge_floor:
        return False
    try:
        if int(row["over_price"]) < price_floor:
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

    return True


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


def _load_schedule_for_date(target_date: date) -> pd.DataFrame | None:
    """Return the schedule parquet for a date, or None if missing."""
    sched_path = raw_path("schedule", f"{target_date.isoformat()}.parquet")
    if not sched_path.exists():
        return None
    try:
        return pd.read_parquet(sched_path)
    except Exception:
        return None


def _format_first_pitch_et(game_datetime_utc) -> str:
    """Format a UTC ISO datetime as a short Eastern Time string (e.g. '7:05 PM ET').
    Returns empty string for invalid input.

    NOTE: Uses zoneinfo for proper EDT/EST handling — direct subtraction would
    silently wrong-shift during DST transitions. This is critical for users
    who bet near transition boundaries (Mar / Nov).
    """
    if game_datetime_utc is None or pd.isna(game_datetime_utc):
        return ""
    try:
        from zoneinfo import ZoneInfo
        ts = pd.to_datetime(game_datetime_utc, utc=True)
        et = ts.tz_convert(ZoneInfo("America/New_York"))
        # %-I strips the leading zero on hour; works on macOS/Linux glibc.
        return et.strftime("%-I:%M %p ET")
    except Exception:
        return ""


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


def _locked_game_pks(target_date: date) -> set[int]:
    """Return game_pks whose game has already started or finished as of now.

    Inverse of _pregame_game_pks: any game in IN_PROGRESS_STATUSES (covers
    Final/In Progress/Suspended/Delayed mid-game) OR with a first-pitch
    timestamp <= now. Used by the merge-with-existing-CSV logic to know
    which morning picks must NOT be wiped on an afternoon re-run — once a
    game is locked, the bet is already placed and the row needs to survive
    so day-grading and dashboard P&L still see it.

    Returns an empty set on any read error — the caller treats that as "no
    locks", which is the safe default (a fresh re-run with no existing CSV
    just writes the new picks).
    """
    sched_path = raw_path("schedule", f"{target_date.isoformat()}.parquet")
    if not sched_path.exists():
        return set()
    try:
        sched = pd.read_parquet(sched_path)
    except Exception:
        return set()
    if "game_pk" not in sched.columns:
        return set()

    locked: set[int] = set()

    if "status" in sched.columns:
        bad = sched[sched["status"].isin(IN_PROGRESS_STATUSES)]
        locked |= set(
            pd.to_numeric(bad["game_pk"], errors="coerce").dropna().astype(int).tolist()
        )

    if "game_datetime" in sched.columns:
        try:
            gdt = pd.to_datetime(sched["game_datetime"], utc=True, errors="coerce")
            now_utc = datetime.now(tz=timezone.utc)
            started = sched[gdt.notna() & (gdt <= now_utc)]
            locked |= set(
                pd.to_numeric(started["game_pk"], errors="coerce")
                .dropna().astype(int).tolist()
            )
        except Exception:
            pass

    return locked


def _compute_drop_rationale(
    prev_row: pd.Series,
    full_pool: pd.DataFrame,
    edge_floor: float,
    price_floor: int,
) -> str:
    """Produce a one-sentence reason a previously-picked row is no longer in
    the latest filter pass.

    `full_pool` is the CURRENT-RUN merged (predictions + odds) DataFrame
    BEFORE any Filter E gating — the pool we'd be filtering down. We look
    up the same (game_pk, player_id) in there and figure out what changed.

    Order of checks is "most diagnostic first" so the most informative
    reason wins when multiple changes happened at once (e.g., edge fell
    AND lineup unconfirmed → we report the edge change, since it's the
    binary gate decision).
    """
    pid = prev_row.get("player_id")
    gpk = prev_row.get("game_pk")
    if pd.isna(pid) or pd.isna(gpk):
        return "Could not identify pick in latest run (missing player_id/game_pk)."

    if full_pool.empty:
        return "Player no longer in scoring pool — likely lineup changed or no prop odds available."

    cur = full_pool[
        (full_pool["game_pk"] == int(gpk))
        & (full_pool["player_id"] == int(pid))
    ]
    if cur.empty:
        return ("Player no longer in projected/confirmed lineup or no longer "
                "has prop odds in the latest fetch.")

    cur = cur.iloc[0]
    cur_edge = cur.get("edge")
    cur_price = cur.get("over_price")

    # Edge fell below floor — the most common cause of a drop.
    try:
        if pd.notna(cur_edge):
            cur_edge_f = float(cur_edge)
            if cur_edge_f < edge_floor:
                prev_edge = prev_row.get("edge")
                if pd.notna(prev_edge):
                    return (f"Edge dropped from {float(prev_edge):.1%} to {cur_edge_f:.1%}, "
                            f"below {edge_floor:.1%} gate floor.")
                return f"Edge fell to {cur_edge_f:.1%}, below {edge_floor:.1%} gate floor."
    except Exception:
        pass

    # Price moved below floor (more negative than allowed).
    try:
        if pd.notna(cur_price):
            cur_price_i = int(cur_price)
            if cur_price_i < price_floor:
                prev_price = prev_row.get("over_price")
                if pd.notna(prev_price):
                    return (f"Price moved from {int(prev_price):+d} to {cur_price_i:+d}, "
                            f"below {price_floor} floor.")
                return f"Price moved to {cur_price_i:+d}, below {price_floor} floor."
    except Exception:
        pass

    # Pitcher feature flipped to unknown.
    cur_pf = cur.get("pitcher_features_known")
    if pd.notna(cur_pf):
        try:
            if int(cur_pf) == 0:
                return ("Probable pitcher no longer announced; pitcher features "
                        "fell back to league means (model output unreliable).")
        except Exception:
            pass

    # Lineup status flipped from confirmed to unconfirmed.
    cur_lc = cur.get("lineup_confirmed")
    prev_lc = prev_row.get("lineup_confirmed")
    try:
        if pd.notna(cur_lc) and not bool(cur_lc) and bool(prev_lc):
            return "Lineup unconfirmed in latest run; projected role uncertain."
    except Exception:
        pass

    # Projected start_rate fell below the gate.
    cur_sr = cur.get("start_rate")
    if pd.notna(cur_sr) and pd.notna(cur_lc):
        try:
            if not bool(cur_lc):
                sr = float(cur_sr)
                if sr < FILTER_E_PROJECTED_MIN_START_RATE:
                    return (f"Projected lineup start_rate {sr:.0%}, below "
                            f"{FILTER_E_PROJECTED_MIN_START_RATE:.0%} gate.")
        except Exception:
            pass

    return "No longer passing Filter E gate; specific cause not determinable from feature diff."


def recommend(
    predictions: pd.DataFrame,
    prop_prices: pd.DataFrame | None = None,
    filter_e: bool = False,
    require_pitcher: bool = False,
    require_confirmed_lineup: bool = False,
    drop_postponed_for_date: date | None = None,
    pre_game_only_for_date: date | None = None,
    edge_floor: float | None = None,
    price_floor: int | None = None,
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

    # When prop_prices has lines from multiple books for the same player, dedupe
    # to one row per (date, player_id) using BOOK_PREFERENCE order — FD wins
    # over DK by default, with both treated as fallback for any other book.
    # Without this step the merge produces 1-row-per-(player, book), which
    # would inflate the bet count and double-count the same player's hit.
    if "book" in m.columns and not m.empty:
        m["book_rank"] = m["book"].map(
            {b: i for i, b in enumerate(BOOK_PREFERENCE)}
        ).fillna(99)
        m = (
            m.sort_values(["date", "player_id", "book_rank"])
             .drop_duplicates(subset=["date", "player_id"], keep="first")
             .drop(columns=["book_rank"])
        )

    m["edge"] = m.apply(lambda r: ev_per_unit(r["p_model"], int(r["over_price"])), axis=1)
    m["filter_e_pass"] = m.apply(
        lambda r: _passes_filter_e(r, edge_floor=edge_floor, price_floor=price_floor),
        axis=1,
    )

    if filter_e:
        m = m[m["filter_e_pass"]]
    else:
        m = m[m["edge"] >= EDGE_MIN]

    # Attach first-pitch time per pick. The schedule parquet has game_datetime
    # in UTC; we surface a human-readable ET string in `first_pitch_et` so the
    # email digest and dashboard can show "7:05 PM ET" without re-deriving it.
    # Critical for the user knowing which picks have already locked-in vs which
    # are evening games still open for action.
    if drop_postponed_for_date is not None and "game_pk" in m.columns:
        sched = _load_schedule_for_date(drop_postponed_for_date)
        if sched is not None and "game_pk" in sched.columns and "game_datetime" in sched.columns:
            time_map = (
                sched[["game_pk", "game_datetime"]]
                .drop_duplicates(subset=["game_pk"], keep="first")
            )
            m = m.merge(time_map, on="game_pk", how="left")
            m["first_pitch_et"] = m["game_datetime"].apply(_format_first_pitch_et)
            # Sort by first-pitch time so closest games appear first — matters
            # for the cron's mid-day re-run when some games are minutes from
            # locking and others are hours away.
            m["_sort_pitch"] = pd.to_datetime(m["game_datetime"], utc=True, errors="coerce")
            m = m.sort_values(
                ["filter_e_pass", "_sort_pitch", "edge"],
                ascending=[False, True, False],
            ).drop(columns=["_sort_pitch", "game_datetime"])
            return m

    # Fallback: no schedule available, sort by edge alone.
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
            "Restrict to Filter E v3 (active production, Optuna-tuned): "
            "edge>=11.4%% & price>=-240, plus a start_rate>=80%% gate on "
            "projected (unconfirmed) lineups. Pair with xgb_v5_recal "
            "(predict.DEFAULT_MODEL). Pass --legacy-v2-1 to roll back to "
            "the v2.1 gate at edge>=15%% & price>=-250 with xgb_v3_recal."
        ),
    )
    parser.add_argument(
        "--legacy-v2-1",
        action="store_true",
        help=(
            "Roll back to the prior production preset: xgb_v3_recal at "
            "Filter E v2.1 (edge>=15%% & price>=-250 with start_rate>=80%%). "
            "Validated 2026-03-20 -> 2026-04-23: +15.9%% ROI / 69.2%% hit. "
            "Use only when you want to flip BOTH the model and the gate "
            "back together — flipping just one is a config we never tested."
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
    parser.add_argument(
        "--edge-floor",
        type=float,
        default=None,
        help=(
            "Override the Filter E edge floor (active default 11.4%% from "
            "Optuna). Use to test alternatives like 0.15 (v2.1 production) or "
            "0.20 (max conviction) without permanently changing the production "
            "constant. Backtest-only — for live picks this should match the "
            "validated production value, or pair with --legacy-v2-1."
        ),
    )
    parser.add_argument(
        "--price-floor",
        type=int,
        default=None,
        help=(
            "Override the Filter E price floor (active default -240 from "
            "Optuna). Use to test alternatives like -250 (v2.1 production), "
            "-200 (pre-2026-04-29 production) or -300 (extreme chalk "
            "inclusion). Same warning as --edge-floor: backtest-only knob, "
            "not for live picks."
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

    # --legacy-v2-1 swaps the gate constants for the v2.1 preset (paired with
    # xgb_v3_recal). If the predictions parquet was scored with xgb_v5_recal
    # (the new default), only the gate is rolled back — for a true full
    # rollback re-run score_today with --model xgb_v3_recal first.
    if args.legacy_v2_1:
        legacy = FILTER_E_PRESETS["v2_1"]
        if args.edge_floor is None:
            args.edge_floor = legacy["edge_min"]
        if args.price_floor is None:
            args.price_floor = legacy["price_min"]
        print(f"  --legacy-v2-1: applying Filter E v2.1 gate "
              f"(edge>={legacy['edge_min']:.0%} & price>={legacy['price_min']}). "
              f"Pair with --model xgb_v3_recal in score_today.py for the "
              f"validated v2.1 production config.")

    # Surface gate overrides in console output so the user always knows
    # what thresholds they're looking at — easy to forget when running ad-hoc.
    eff_edge_floor  = args.edge_floor  if args.edge_floor  is not None else FILTER_E_EDGE_MIN
    eff_price_floor = args.price_floor if args.price_floor is not None else FILTER_E_PRICE_MIN
    if args.filter_e:
        if args.edge_floor is not None or args.price_floor is not None:
            print(f"  Filter E gate (overridden): "
                  f"edge>={eff_edge_floor:.0%} & price>={eff_price_floor}")

    recs = recommend(
        preds,
        prop_prices=prices,
        filter_e=args.filter_e,
        require_pitcher=args.require_pitcher,
        require_confirmed_lineup=args.require_confirmed_lineup,
        drop_postponed_for_date=target_d,
        pre_game_only_for_date=target_d if args.pre_game_only else None,
        edge_floor=args.edge_floor,
        price_floor=args.price_floor,
    )
    if args.filter_e and (args.edge_floor is not None or args.price_floor is not None):
        label = f"FILTER E [edge>={eff_edge_floor:.0%}, price>={eff_price_floor}]"
    elif args.filter_e:
        label = "FILTER E"
    else:
        label = f"edge>={EDGE_MIN:.0%}"

    # How many came directly from this run's filter pass — kept around so the
    # post-merge summary can show "N new + M carried forward = total".
    n_new = len(recs)

    # Use a distinct suffix when gate overrides are in effect — otherwise the
    # experimental run would clobber the production _filter_e.csv on disk
    # (which feeds grade_picks.py + the dashboard).
    if args.filter_e and (args.edge_floor is not None or args.price_floor is not None):
        e_str = f"e{int(round(eff_edge_floor * 100))}"   # 0.12 -> "e12"
        p_str = f"p{abs(eff_price_floor)}"               # -250 -> "p250"
        suffix = f"_filter_e_{e_str}_{p_str}"
    elif args.filter_e:
        suffix = "_filter_e"
    else:
        suffix = ""

    out_csv = output_path("recommendations", f"{d}{suffix}.csv")

    # ----------------------------------------------------------------------
    # Merge with existing picks file (production runs only).
    # ----------------------------------------------------------------------
    # When the user re-runs the workflow during the day (e.g. 8 AM full slate
    # → 5:45 PM late slate refresh), we want to:
    #   1. PRESERVE picks for games that have already started/finished — those
    #      bets are committed and need to stay so the dashboard P&L works the
    #      next morning.
    #   2. REPLACE picks for still-pre-game games with the latest run's view
    #      (latest odds, lineup status, edge).
    #   3. RECORD the picks that were dropped (in old file, not in new run,
    #      game still pre-game) along with a one-sentence reason, so we can
    #      still see what would have been picked and why it changed.
    #
    # Only applies to the production CSV (suffix == "_filter_e"). Ad-hoc
    # tuning runs with --edge-floor/--price-floor write to suffixed files
    # and don't touch the canonical picks history.
    do_merge = (
        args.filter_e
        and args.edge_floor is None
        and args.price_floor is None
        and prices is not None
        and not prices.empty
    )

    if do_merge:
        # Build the unfiltered merged pool (preds + prices) so _compute_drop_
        # rationale can look up the CURRENT edge/price/lineup state for any
        # row we're about to drop.
        preds_n = preds.copy()
        preds_n["player_id"] = pd.to_numeric(
            preds_n["player_id"], errors="coerce"
        ).astype("Int64")
        prices_n = prices.copy()
        prices_n["player_id"] = pd.to_numeric(
            prices_n["player_id"], errors="coerce"
        ).astype("Int64")
        full_pool = preds_n.merge(prices_n, on=["date", "player_id"], how="inner")
        if not full_pool.empty:
            full_pool["edge"] = full_pool.apply(
                lambda r: ev_per_unit(r["p_model"], int(r["over_price"]))
                if pd.notna(r.get("over_price")) else None,
                axis=1,
            )

        existing = None
        if out_csv.exists():
            try:
                existing = pd.read_csv(out_csv)
            except Exception as e:
                print(f"  WARN: couldn't read existing {out_csv.name} for merge: {e}")
                existing = None

        if existing is not None and not existing.empty and "game_pk" in existing.columns:
            existing["game_pk"] = pd.to_numeric(
                existing["game_pk"], errors="coerce"
            ).astype("Int64")
            existing["player_id"] = pd.to_numeric(
                existing["player_id"], errors="coerce"
            ).astype("Int64")

            locked_pks = _locked_game_pks(target_d)

            # Locked picks: carry forward verbatim. Their bets are placed,
            # the row is already what got bet on, no re-scoring possible.
            locked_existing = existing[
                existing["game_pk"].isin(locked_pks)
            ].copy()

            # Pre-game picks from the existing file: candidates for diff.
            pregame_existing = existing[
                ~existing["game_pk"].isin(locked_pks)
            ].copy()

            # Identity keys for new picks. Latest run wins on overlap.
            if not recs.empty and "game_pk" in recs.columns:
                rec_keys = pd.DataFrame({
                    "game_pk":   pd.to_numeric(recs["game_pk"], errors="coerce").astype("Int64"),
                    "player_id": pd.to_numeric(recs["player_id"], errors="coerce").astype("Int64"),
                }).dropna()
                new_keys = set(zip(
                    rec_keys["game_pk"].astype(int).tolist(),
                    rec_keys["player_id"].astype(int).tolist(),
                ))
            else:
                new_keys = set()

            # Diff: existing pre-game picks not in new picks → DROPPED.
            dropped_rows: list[pd.Series] = []
            for _, row in pregame_existing.iterrows():
                try:
                    key = (int(row["game_pk"]), int(row["player_id"]))
                except (TypeError, ValueError):
                    continue
                if key in new_keys:
                    continue  # still picked, no need to track
                rationale = _compute_drop_rationale(
                    row, full_pool, eff_edge_floor, eff_price_floor,
                )
                r_out = row.copy()
                r_out["drop_reason"] = rationale
                # Stamp when the drop was observed so multiple re-runs the
                # same day show their own reasons in chronological order.
                r_out["dropped_at"] = datetime.now(tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                dropped_rows.append(r_out)

            # Defense: a locked-game row could in theory overlap with new_keys
            # if --pre-game-only was disabled. Drop any such overlap from
            # locked_existing so we don't double-count.
            if not locked_existing.empty:
                locked_existing = locked_existing.drop_duplicates(
                    subset=["game_pk", "player_id"], keep="last"
                )
                if new_keys:
                    locked_keep = []
                    for _, r in locked_existing.iterrows():
                        try:
                            k = (int(r["game_pk"]), int(r["player_id"]))
                        except (TypeError, ValueError):
                            locked_keep.append(True)
                            continue
                        locked_keep.append(k not in new_keys)
                    locked_existing = locked_existing[locked_keep]

            # Merge: locked + new (latest run wins on overlap by construction).
            if not locked_existing.empty:
                # Align columns — recs may have a column locked_existing doesn't,
                # or vice versa. concat handles that automatically by NaN-filling.
                recs = pd.concat([locked_existing, recs], ignore_index=True)
                print(
                    f"  carried forward {len(locked_existing)} pick(s) from "
                    f"{len(locked_pks)} locked game(s) (already started/finished)."
                )

            # Save dropped picks. Append to any existing file from earlier
            # re-runs the same day so we accumulate the full drop history;
            # dedupe on (game_pk, player_id) keeping the most recent reason.
            if dropped_rows:
                dropped_df = pd.DataFrame(dropped_rows)
                drop_path = output_path("dropped", f"{d}.csv")
                drop_path.parent.mkdir(parents=True, exist_ok=True)

                if drop_path.exists():
                    try:
                        prev_dropped = pd.read_csv(drop_path)
                        dropped_df = pd.concat(
                            [prev_dropped, dropped_df], ignore_index=True,
                        )
                        dropped_df = dropped_df.drop_duplicates(
                            subset=["game_pk", "player_id"], keep="last",
                        )
                    except Exception as e:
                        print(f"  WARN: couldn't append to existing {drop_path.name}: {e}")

                dropped_df.to_csv(drop_path, index=False)
                print(
                    f"  {len(dropped_rows)} pick(s) dropped from previous run; "
                    f"saved to {drop_path.name} (drop_reason column has the why)."
                )

    # Print summary AFTER the merge so the displayed bet count matches what's
    # actually written to the CSV (and what the dashboard will fetch).
    n_bets = len(recs)
    n_carried = max(0, n_bets - n_new)
    n_print = n_bets if args.top in (0, None) else min(args.top, n_bets)
    print(f"\n{'=' * 60}\nRECOMMENDATIONS  {d}  ({label})\n{'=' * 60}")
    if n_carried:
        print(f"  {n_bets} bets in final slate "
              f"({n_new} new + {n_carried} carried-forward from locked games; "
              f"{'all' if n_print == n_bets else f'top {n_print}'} shown below; "
              f"full list in CSV).")
    else:
        print(f"  {n_bets} bets cleared the filter "
              f"({'all' if n_print == n_bets else f'top {n_print}'} shown below; "
              f"full list in CSV).")
    if n_bets:
        print(recs.head(n_print).to_string(index=False))

    # Final write.
    recs.to_csv(out_csv, index=False)
