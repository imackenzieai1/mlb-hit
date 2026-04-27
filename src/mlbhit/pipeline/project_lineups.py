"""Projected lineups for games whose actual lineup hasn't been posted yet.

Why this exists
---------------
fetch_lineups.py only returns rows for games where MLB's boxscore endpoint
has populated `battingOrder`. That's typically 2–4 hours before first pitch
for the home team and as little as ~30 min for some visiting teams. If the
morning runner only scores those games, you only see bets in 2–3 of the
slate's 12–15 games — exactly the bug we hit on 2026-04-25.

Books, on the other hand, post hit-prop lines as soon as the probable starting
pitcher is announced (often the night before). Any prop line whose batter
doesn't take an at-bat is voided by the book — so we can confidently bet
projected starters: confirmed plays = action, scratched plays = no action.

Approach
--------
For each team playing on `target_date`:
  1. Pull last LOOKBACK_DAYS of boxscores for that team.
  2. Group by player_id and count how many of those games they STARTED
     (i.e. boxscore row exists with batting_order in 1..9).
  3. Keep players whose start frequency >= START_THRESHOLD.
  4. Each kept player's projected batting_order = the mode of their
     recent starts. If tied, take the lower (more PAs).
  5. If after filtering we have fewer than 9 starters, fall back to
     the top-9 by start count so we always emit a complete lineup.

We never over-fill: if step 3 yields ≥9 players, we keep all of them
(switch-hitters / platoons can legitimately produce a "10th regular").
The model decides via p_model + edge whether to bet any of them; we don't
prune at projection time.

Output schema matches fetch_lineups.fetch_lineups exactly, plus
`lineup_confirmed=False` on every projected row. The wrapper in
score_today.py drops projected rows for any (game_pk, player_id) that's
already present in the confirmed lineups parquet.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..io import clean_path

# Players with a >= 50% start rate over the last 14 days are "regulars."
# Tighter threshold (e.g. 0.7) misses platoon batters who legitimately split
# starts; looser (e.g. 0.3) brings in 4th outfielders who are 30% likely to
# play and would inflate phantom edges. 0.5 was the inflection in a quick
# 2026-YTD scan: regulars who started 50%+ of recent games started ~85% of
# subsequent games on average.
START_THRESHOLD = 0.5

# 14 days ≈ 12-13 games for most teams. Long enough to smooth past a single
# day off; short enough that an injured player who's been out a week falls
# below the 50% threshold and stops being projected.
LOOKBACK_DAYS = 14

# Minimum games observed in the lookback window. Below this we don't trust
# the frequency estimate and skip projection for that team. Hits early-season
# (March 21–28) and off-day-heavy stretches.
MIN_LOOKBACK_GAMES = 5

# Schedule parquet stores MLB team IDs in `home_abbr`/`away_abbr` (legacy
# column name — they're numeric team_ids, e.g. 110 for BAL). Boxscores keep
# the proper short code in `team` (BAL, BOS, ...). Map between them here.
# IDs are stable across seasons; sourced from
# https://statsapi.mlb.com/api/v1/teams?sportId=1
TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA", 109: "AZ",  110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def _abbr_from_team_id(team_id) -> str | None:
    try:
        return TEAM_ID_TO_ABBR.get(int(team_id))
    except (TypeError, ValueError):
        return None


def _load_recent_boxscores(target: date, season: int) -> pd.DataFrame:
    """All boxscore starts in [target - LOOKBACK_DAYS, target - 1]."""
    p: Path = clean_path(f"boxscores_{season}.parquet")
    if not p.exists():
        return pd.DataFrame()
    box = pd.read_parquet(p)
    if box.empty:
        return box
    box["date"] = box["date"].astype(str)
    cutoff_lo = (target - timedelta(days=LOOKBACK_DAYS)).isoformat()
    cutoff_hi = (target - timedelta(days=1)).isoformat()
    mask = (box["date"] >= cutoff_lo) & (box["date"] <= cutoff_hi)
    out = box.loc[mask].copy()
    # Only count actual STARTS — boxscore rows for pinch hitters have a
    # batting_order outside 1..9 or NaN. Project only legitimate starters.
    if "batting_order" in out.columns:
        bo = pd.to_numeric(out["batting_order"], errors="coerce")
        out = out.loc[bo.between(1, 9, inclusive="both")].copy()
        out["batting_order"] = bo
    return out


def _project_team(
    box_recent: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """Project a single team's expected starting lineup."""
    tb = box_recent[box_recent["team"] == team]
    if tb.empty:
        return pd.DataFrame()
    games_seen = tb["game_pk"].nunique()
    if games_seen < MIN_LOOKBACK_GAMES:
        return pd.DataFrame()

    # Per-player aggregates: how often did they start, and what spot.
    grp = tb.groupby("player_id", as_index=False).agg(
        starts=("game_pk", "nunique"),
        last_name=("player_name", "last"),  # newest in iteration order
    )
    grp["start_rate"] = grp["starts"] / games_seen
    regulars = grp[grp["start_rate"] >= START_THRESHOLD].copy()

    if len(regulars) < 9:
        # Fallback: take top 9 by raw starts. Better to over-project than to
        # leave a team partially blank — the user's already gated this slate
        # on edge + Filter E, and a miss-projected bet just voids.
        regulars = grp.nlargest(9, "starts").copy()

    # Mode batting order per regular. ties -> smallest (top of order = more PAs).
    spot_mode = (
        tb.groupby("player_id")["batting_order"]
        .agg(lambda s: int(s.value_counts().sort_index().idxmax()))
        .rename("projected_spot")
        .reset_index()
    )
    regulars = regulars.merge(spot_mode, on="player_id", how="left")
    regulars["projected_spot"] = regulars["projected_spot"].fillna(5).astype(int)
    return regulars[["player_id", "last_name", "projected_spot", "start_rate"]]


def project_lineups(
    target_date: date,
    schedule: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Return a projected lineup DataFrame for `target_date`.

    Output columns mirror fetch_lineups.fetch_lineups:
      date, game_pk, team, opponent, home_away, player_id, player_name,
      lineup_spot, lineup_confirmed (always False here), start_rate

    `start_rate` is extra context (not in the confirmed schema) so downstream
    code can optionally weight or filter on projection confidence.
    """
    if schedule.empty:
        return pd.DataFrame()

    box_recent = _load_recent_boxscores(target_date, season)
    if box_recent.empty:
        return pd.DataFrame()

    # Schedule parquet has team_ids in {home,away}_abbr (legacy name — the
    # values are numeric: 110, 111, ...). Boxscores use the short code in
    # `team` (BAL, BOS, ...). Convert via TEAM_ID_TO_ABBR before matching.
    rows: list[dict] = []
    for _, g in schedule.iterrows():
        pk = g["game_pk"]
        for side, opp_side in (("home", "away"), ("away", "home")):
            team_col = f"{side}_abbr"
            opp_col = f"{opp_side}_abbr"
            if team_col not in g.index or opp_col not in g.index:
                continue
            team = _abbr_from_team_id(g[team_col])
            opp = _abbr_from_team_id(g[opp_col])
            if not team or not opp:
                continue
            proj = _project_team(box_recent, team)
            if proj.empty:
                continue
            for _, p in proj.iterrows():
                rows.append({
                    "date": target_date.isoformat(),
                    "game_pk": pk,
                    "team": team,
                    "opponent": opp,
                    "home_away": "H" if side == "home" else "A",
                    "player_id": int(p["player_id"]),
                    "player_name": p["last_name"],
                    "lineup_spot": int(p["projected_spot"]),
                    "lineup_confirmed": False,
                    "start_rate": float(p["start_rate"]),
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def merge_confirmed_with_projected(
    confirmed: pd.DataFrame,
    projected: pd.DataFrame,
) -> pd.DataFrame:
    """Combine confirmed lineups with projected ones; confirmed always wins.

    Collision rule: for any (game_pk) that has confirmed rows, drop ALL
    projected rows for that game. Don't mix-and-match — once MLB has posted
    the actual lineup we should trust it end-to-end (a regular sitting that
    day is information we want to honor).

    For games with no confirmed lineup posted yet, take the projection in full.
    """
    if confirmed is None or confirmed.empty:
        out = projected.copy() if projected is not None else pd.DataFrame()
        if "lineup_confirmed" not in out.columns and not out.empty:
            out["lineup_confirmed"] = False
        return out
    if projected is None or projected.empty:
        out = confirmed.copy()
        if "lineup_confirmed" not in out.columns:
            # fetch_lineups already sets this from `bool(order)`.
            out["lineup_confirmed"] = True
        return out

    confirmed_games = set(confirmed["game_pk"].unique().tolist())
    projected_kept = projected[~projected["game_pk"].isin(confirmed_games)]

    # Align columns: confirmed has lineup_confirmed (True/False from
    # `bool(order)`); projected has it as False. start_rate only on projected.
    if "start_rate" not in confirmed.columns:
        confirmed = confirmed.copy()
        confirmed["start_rate"] = pd.NA

    return pd.concat(
        [confirmed, projected_kept],
        ignore_index=True,
        sort=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a projected lineup for a given date for inspection."
    )
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--season", type=int, default=None,
                        help="Season for boxscore lookback (default: year of --date)")
    args = parser.parse_args()

    from .fetch_schedule import fetch_schedule

    d = date.fromisoformat(args.date)
    season = args.season or d.year
    sched = fetch_schedule(d)
    proj = project_lineups(d, sched, season=season)
    if proj.empty:
        print(f"No projection produced for {d} (no boxscore lookback?)")
    else:
        print(f"Projected {len(proj)} batter-rows across "
              f"{proj['game_pk'].nunique()} games for {d}")
        print(proj.head(20).to_string(index=False))
