"""Recent-form features (game-window, not date-window).

The existing rolling features in ``batter_rolling.parquet`` are time-based
(14d, 30d). For a "last N games" stat — the way commentators / handicappers
talk about hot streaks — we have to walk the box scores per player and
take the trailing N games where the batter actually had a plate appearance,
regardless of calendar gap.

This module is intentionally separate from ``features/rolling.py`` because:
  * It's a downstream SIZING / EMPHASIS signal, not a model feature (the
    model is not retrained against it).
  * It's cheap enough to compute on-demand (per-player back-walk on the
    same boxscores parquet that the rolling builder already uses).

Two signals live here:
  * ``attach_hot_streak``  — batter-side: hits/AB over last 6 PA-counted games.
  * ``attach_opp_grind``   — opponent-side: opp team's consecutive-games-in-
                              -a-row streak (fatigue/bullpen-grind proxy).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def attach_hot_streak(
    targets: pd.DataFrame,
    boxscores: pd.DataFrame,
    n_games: int = 6,
    min_avg: float = 0.300,
    units_hot: float = 2.0,
    units_cold: float = 1.0,
) -> pd.DataFrame:
    """Append hot-streak columns to a (player_id, date)-keyed DataFrame.

    For each row in ``targets``, look at that player's last ``n_games``
    boxscore rows where ``pa >= 1`` AND ``date < targets.date``. Compute
    total hits / AB over that window. The "hot" flag fires when the batter
    has the full ``n_games`` of data AND batting average exceeds ``min_avg``.

    Returns a copy of ``targets`` with appended columns:
        hot_streak_n_games : int   (0 if no qualifying games found)
        hot_streak_h       : int
        hot_streak_ab      : int
        hot_streak_avg     : float (NaN if hot_streak_ab == 0)
        hot_streak         : 0 | 1
        recommended_units  : float (units_hot if hot, else units_cold)

    Notes
    -----
    * Skips boxscore rows where pa == 0 — bench games / pinch-defensive
      appearances shouldn't dilute "his last 6 at-bats" the way a strict
      games-on-schedule window would.
    * Requires the FULL ``n_games`` window to fire the flag. Early-season
      players with <6 prior games stay cold even if their tiny sample
      averages 1.000 — small samples shouldn't trigger 2x sizing.
    * Idempotent on date type: accepts string YYYY-MM-DD or pandas Timestamp.
    """
    if targets.empty:
        out = targets.copy()
        for c, default in (
            ("hot_streak_n_games", 0),
            ("hot_streak_h", 0),
            ("hot_streak_ab", 0),
            ("hot_streak_avg", np.nan),
            ("hot_streak", 0),
            ("recommended_units", units_cold),
        ):
            out[c] = default
        return out

    # Restrict box to games where batter actually came up. Coerce date to
    # Timestamp once so per-row comparisons are O(1).
    box = boxscores[pd.to_numeric(boxscores["pa"], errors="coerce").fillna(0) >= 1].copy()
    box["date_ts"] = pd.to_datetime(box["date"])
    box = box.sort_values(["player_id", "date_ts"])

    # Build a per-player view we can index into. groupby+get_group is fast
    # enough at our scale (~10k players, ~700 targets/day in backtest).
    by_player = {pid: g for pid, g in box.groupby("player_id", sort=False)}

    target = targets.copy()
    target["_date_ts"] = pd.to_datetime(target["date"])

    h_arr = np.zeros(len(target), dtype=np.int64)
    ab_arr = np.zeros(len(target), dtype=np.int64)
    n_arr = np.zeros(len(target), dtype=np.int64)

    for i, (pid, td) in enumerate(zip(target["player_id"].values, target["_date_ts"].values)):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        g = by_player.get(pid)
        if g is None:
            continue
        prior = g[g["date_ts"] < td].tail(n_games)
        if prior.empty:
            continue
        h_arr[i] = int(prior["hits"].sum())
        ab_arr[i] = int(prior["ab"].sum())
        n_arr[i] = len(prior)

    target["hot_streak_n_games"] = n_arr
    target["hot_streak_h"] = h_arr
    target["hot_streak_ab"] = ab_arr
    # np.divide with explicit `where` and `out` avoids the divide-by-zero
    # RuntimeWarning that np.where(cond, h/ab, nan) emits because both
    # branches are evaluated before the where mask is applied.
    avg_out = np.full(len(h_arr), np.nan, dtype=np.float64)
    np.divide(h_arr, ab_arr, out=avg_out, where=(ab_arr > 0))
    target["hot_streak_avg"] = avg_out
    target["hot_streak"] = (
        (target["hot_streak_n_games"] >= n_games)
        & (target["hot_streak_avg"] > min_avg)
    ).astype(int)
    target["recommended_units"] = np.where(
        target["hot_streak"] == 1, units_hot, units_cold
    )
    return target.drop(columns=["_date_ts"])


def attach_opp_grind(
    targets: pd.DataFrame,
    boxscores: pd.DataFrame,
    threshold: int = 10,
) -> pd.DataFrame:
    """Append opponent consecutive-games-streak columns.

    For each row in ``targets`` (which must have ``date`` and ``opp_team``),
    counts the opponent team's consecutive-games streak going INTO the
    target date. The count is "today + every prior calendar day where the
    opp team also played" — so a team coming off an off-day yesterday
    gets streak=1 (just today).

    The ``opp_grind`` flag fires when the opp's streak strictly exceeds
    ``threshold`` games — default 10, i.e., team is on game 11+ of their
    current run-in. Pitching staffs and bullpens get worn down; the
    hypothesis here is that hits-allowed climbs over long no-rest stretches.

    Returns a copy of ``targets`` with appended columns:
        opp_consec_games : int  (the streak length, including today)
        opp_grind        : 0 | 1

    Notes
    -----
    * Source of truth for the opp's schedule is ``boxscores`` — every
      historical game appears as a (team, date) tuple per row. For LIVE
      scoring, today's game won't be in boxscores yet; the count is from
      yesterday backward, then we add 1 for today. Equivalent result.
    * Computed in calendar-day space, not game-day space. A doubleheader
      on the same date counts as one day.
    * `targets.opp_team` must align with `boxscores.team` (same abbrev set).
    """
    if targets.empty:
        out = targets.copy()
        out["opp_consec_games"] = 0
        out["opp_grind"] = 0
        return out

    # Per-team set of dates they've played. Uses set lookup (O(1)) so the
    # streak walk per target is O(streak length), which is bounded by ~20.
    team_dates: dict[str, set] = {}
    for team, g in boxscores.groupby("team", sort=False):
        team_dates[str(team)] = set(pd.to_datetime(g["date"]).dt.date.unique().tolist())

    target = targets.copy()
    streak_arr = np.zeros(len(target), dtype=np.int64)

    one_day = timedelta(days=1)
    for i, (opp, td_str) in enumerate(zip(target["opp_team"].values, target["date"].values)):
        if pd.isna(opp):
            continue
        try:
            td = pd.to_datetime(td_str).date()
        except Exception:
            continue
        dates = team_dates.get(str(opp))
        if dates is None:
            # Today counts even if opp has no prior history (early season).
            streak_arr[i] = 1
            continue
        # Today always counts (we're betting against this game). Then walk
        # backward through calendar days and stop at the first off-day.
        streak = 1
        d = td - one_day
        while d in dates:
            streak += 1
            d -= one_day
        streak_arr[i] = streak

    target["opp_consec_games"] = streak_arr
    target["opp_grind"] = (streak_arr > threshold).astype(int)
    return target


def attach_rolling_game_ba(
    targets: pd.DataFrame,
    boxscores: pd.DataFrame,
    windows: tuple[int, ...] = (3, 10),
) -> pd.DataFrame:
    """Append game-window rolling batting averages (separate from the 14d/30d
    time-windowed features in batter_rolling.parquet).

    For each row in ``targets``, walks back through that player's box score
    rows where ``pa >= 1``, takes the last N rows STRICTLY BEFORE the target
    date, and computes:
        ba_{N}g  = sum(hits) / sum(ab)  (NaN if AB == 0)

    Why both: the 14d / 30d windows answer "form over the last calendar
    span"; game windows answer "form over the last N at-bat opportunities"
    (more robust to off-days, IL stints, doubleheaders).

    Notes
    -----
    * Returns NaN for windows where the player has fewer than N qualifying
      games. XGBoost handles NaN natively — DO NOT median-fill these, or
      cold-start rookies will look like league-average hitters.
    * Defaults to (3, 10) since the 6-game window is already exposed via
      ``hot_streak_avg`` in attach_hot_streak.
    """
    if targets.empty:
        out = targets.copy()
        for w in windows:
            out[f"ba_{w}g"] = np.nan
        return out

    box = boxscores[pd.to_numeric(boxscores["pa"], errors="coerce").fillna(0) >= 1].copy()
    box["date_ts"] = pd.to_datetime(box["date"])
    box = box.sort_values(["player_id", "date_ts"])
    by_player = {pid: g for pid, g in box.groupby("player_id", sort=False)}

    target = targets.copy()
    target["_date_ts"] = pd.to_datetime(target["date"])
    max_w = max(windows)

    arrays = {w: np.full(len(target), np.nan, dtype=np.float64) for w in windows}

    for i, (pid, td) in enumerate(zip(target["player_id"].values, target["_date_ts"].values)):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        g = by_player.get(pid)
        if g is None:
            continue
        prior = g[g["date_ts"] < td].tail(max_w)
        if prior.empty:
            continue
        # Compute each window from the SAME prior slice (avoiding repeated
        # filter passes). For w <= len(prior), take the last w rows.
        for w in windows:
            if len(prior) >= w:
                window_slice = prior.tail(w)
                ab = int(window_slice["ab"].sum())
                if ab > 0:
                    arrays[w][i] = float(window_slice["hits"].sum()) / ab

    for w in windows:
        target[f"ba_{w}g"] = arrays[w]
    return target.drop(columns=["_date_ts"])
