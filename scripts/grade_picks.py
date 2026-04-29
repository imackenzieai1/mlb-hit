#!/usr/bin/env python
"""Grade historical Filter E recommendations against actual game results.

For each `data/output/recommendations/{date}_filter_e.csv` whose date is in the
past, look up each pick's outcome from `data/clean/boxscores_{year}.parquet`
and append three columns:

    outcome     : "hit" | "miss" | "void" | "" (pending)
    hits_actual : Int64 (NaN if not yet graded or voided)
    pnl         : float (P&L per $1 unit stake, 0 for void/pending)

"void" covers postponed games, scratched-from-lineup players, or rows we
can't match in the boxscore (book would also void those). The CSV is
rewritten in-place, idempotently (re-running grades the same row to the
same answer; pending rows get filled in once the boxscore is available).

Run from the workflow daily, AFTER fetch_boxscores has refreshed the
season parquet, so yesterday's games are graded by the time the dashboard
manifest mirror runs.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RECS_DIR = REPO_ROOT / "data" / "output" / "recommendations"
BOX_DIR = REPO_ROOT / "data" / "clean"

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_filter_e\.csv$")

# Columns this script writes. Defined here so we can detect partially-graded
# CSVs and only re-grade pending rows. `pnl` is flat-$1 P&L; `pnl_weighted`
# applies the row's `recommended_units` multiplier (2x on hot bats by default,
# 1x otherwise — see src/mlbhit/features/recent_form.py).
GRADE_COLS = ["outcome", "hits_actual", "pnl", "pnl_weighted"]


def _payout_per_unit(odds_american: int) -> float:
    """Profit on a $1 winning bet at given American odds."""
    return odds_american / 100 if odds_american > 0 else 100 / -odds_american


def _grade_row(row, box_for_date: pd.DataFrame) -> tuple[str, float | None, float]:
    """Grade a single recommendation row.

    Match order:
      1. (player_id, game_pk) — exact, handles doubleheaders correctly
      2. (player_id) on the date — fallback when game_pk doesn't match
         (e.g., recommend.py picked game 1 of a doubleheader and the
         row in the box is game 2). Pragmatic for v1; books grade game 1
         only, so we'd over-credit a doubleheader hit-in-game-2 here, but
         the volume is tiny.
    """
    pid = row.get("player_id")
    gpk = row.get("game_pk")
    price = row.get("over_price")

    if pd.isna(pid) or pd.isna(price):
        return "", None, 0.0

    pid = int(pid)
    matched = box_for_date[box_for_date["player_id"] == pid]
    if matched.empty:
        # Player wasn't in the box at all — scratched / postponed / not
        # actually in the lineup. Books void → P&L 0.
        return "void", None, 0.0

    if not pd.isna(gpk):
        exact = matched[matched["game_pk"] == int(gpk)]
        if not exact.empty:
            matched = exact

    pa = matched["pa"].iloc[0]
    if pa == 0:
        # Player was on the card but never came up to bat (DH spot
        # cleared, defensive sub before any PA, etc.) — book voids.
        return "void", 0, 0.0

    hits = int(matched["hits"].iloc[0])
    got_hit = bool(matched["got_hit"].iloc[0])

    try:
        odds = int(price)
    except (TypeError, ValueError):
        return "", hits, 0.0

    if got_hit:
        return "hit", hits, _payout_per_unit(odds)
    return "miss", hits, -1.0


def _load_box_for_year(year: int) -> pd.DataFrame:
    p = BOX_DIR / f"boxscores_{year}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["date", "game_pk", "player_id", "pa", "hits", "got_hit"])
    return pd.read_parquet(p, columns=["date", "game_pk", "player_id", "pa", "hits", "got_hit"])


def grade_one(csv_path: Path, today: date) -> tuple[int, int, float]:
    """Grade one CSV in-place. Returns (newly_graded, total_graded, day_pnl)."""
    m = DATE_RE.match(csv_path.name)
    if not m:
        return 0, 0, 0.0
    target = date.fromisoformat(m.group(1))
    if target >= today:
        # Don't grade today or future — games haven't settled.
        return 0, 0, 0.0

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0, 0, 0.0

    # Initialize grade columns if missing (preserves existing graded rows).
    for col, default in (
        ("outcome", ""),
        ("hits_actual", pd.NA),
        ("pnl", 0.0),
        ("pnl_weighted", 0.0),
    ):
        if col not in df.columns:
            df[col] = default

    box = _load_box_for_year(target.year)
    if box.empty:
        return 0, 0, 0.0
    box_day = box[box["date"] == target.isoformat()]
    if box_day.empty:
        # Date not yet in boxscores parquet — likely the parquet was
        # refreshed but this date wasn't included (off-day, future game,
        # API miss). Leave pending rows alone; revisit next run.
        return 0, 0, 0.0

    newly_graded = 0
    for i, row in df.iterrows():
        already = str(row.get("outcome", "")).strip()
        if already in {"hit", "miss", "void"}:
            continue
        outcome, hits_actual, pnl = _grade_row(row, box_day)
        if outcome:
            # Apply the row's recommended_units multiplier if present, else
            # 1.0 (so legacy rows graded before the column existed still
            # produce a sensible pnl_weighted = pnl).
            try:
                units = float(row.get("recommended_units", 1.0))
                if pd.isna(units) or units <= 0:
                    units = 1.0
            except (TypeError, ValueError):
                units = 1.0
            df.at[i, "outcome"] = outcome
            df.at[i, "hits_actual"] = hits_actual if hits_actual is not None else pd.NA
            df.at[i, "pnl"] = pnl
            df.at[i, "pnl_weighted"] = pnl * units
            newly_graded += 1

    df.to_csv(csv_path, index=False)

    graded_mask = df["outcome"].isin(["hit", "miss", "void"])
    total_graded = int(graded_mask.sum())
    day_pnl = float(df.loc[graded_mask, "pnl"].sum())
    return newly_graded, total_graded, day_pnl


def main() -> None:
    today = date.today()
    if not RECS_DIR.exists():
        print(f"  no recommendations dir at {RECS_DIR}; nothing to grade.")
        return

    season_pnl = 0.0
    season_bets = 0
    season_hits = 0

    csvs = sorted(p for p in RECS_DIR.iterdir() if DATE_RE.match(p.name))
    for csv_path in csvs:
        newly, total, day_pnl = grade_one(csv_path, today)
        if total == 0:
            continue
        df = pd.read_csv(csv_path)
        graded = df[df["outcome"].isin(["hit", "miss", "void"])]
        bets = int((graded["outcome"] != "void").sum())
        hits = int((graded["outcome"] == "hit").sum())
        season_bets += bets
        season_hits += hits
        season_pnl += day_pnl
        flag = f"  (+{newly} new)" if newly else ""
        print(f"  {csv_path.name}: {bets} bets, {hits} hits ({hits/bets*100:.1f}% hit rate), "
              f"P&L ${day_pnl:+.2f}{flag}")

    if season_bets:
        roi = season_pnl / season_bets * 100
        print(f"\n  SEASON-TO-DATE: {season_bets} bets, {season_hits} hits "
              f"({season_hits/season_bets*100:.1f}%), P&L ${season_pnl:+.2f} (ROI {roi:+.1f}%)")


if __name__ == "__main__":
    main()
