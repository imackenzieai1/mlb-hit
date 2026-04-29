from __future__ import annotations

from datetime import date

import pandas as pd

from ..features.batter import build_batter_features
from ..features.blended import (
    build_blended_batter_features,
    build_blended_pitcher_features,
    compute_rolling_batter_stats,
)
from ..features.park_weather import attach_park
from ..features.pa import expected_pa
from ..features.pitcher import build_pitcher_features
from ..features.recent_form import attach_hot_streak, attach_opp_grind
from ..io import clean_path, output_path
from ..model.predict import predict
from .fetch_lineups import fetch_lineups
from .fetch_schedule import fetch_schedule
from .project_lineups import merge_confirmed_with_projected, project_lineups


def score_for_date(
    d: date,
    season: int,
    prior_season: int | None = None,
    use_projection: bool = True,
) -> pd.DataFrame:
    """Score today's games.

    If `prior_season` is provided, uses PA/TBF-weighted blended features
    (current season to date <> prior season) — recommended during early
    regular season when current-season samples are small. Leave None to
    use current-season stats as-is (recommended late in the season).

    `use_projection=True` (default) fills in any game whose actual lineup
    hasn't posted yet with a projected lineup built from the last 14 days
    of starts. This is the v1 fix for the morning-runner blind spot where
    only a handful of early games had lineups, collapsing the slate.
    Confirmed lineups always win on collision; projected rows carry
    `lineup_confirmed=False` so downstream filters can require confirmed.
    """
    sched = fetch_schedule(d)
    confirmed = fetch_lineups(d)

    if use_projection:
        projected = project_lineups(d, sched, season=season)
        lineups = merge_confirmed_with_projected(confirmed, projected)
        n_conf = int(lineups["lineup_confirmed"].sum()) if not lineups.empty else 0
        n_proj = len(lineups) - n_conf if not lineups.empty else 0
        print(f"  lineups: {n_conf} confirmed + {n_proj} projected "
              f"({lineups['game_pk'].nunique() if not lineups.empty else 0} games)")
    else:
        lineups = confirmed
        if not lineups.empty and "lineup_confirmed" not in lineups.columns:
            lineups = lineups.assign(lineup_confirmed=True)

    if lineups.empty:
        print("No lineups available (confirmed or projected); skipping.")
        return pd.DataFrame()

    sched_lite = sched[["game_pk", "home_probable_pitcher_id", "away_probable_pitcher_id"]]
    df = lineups.merge(sched_lite, on="game_pk", how="left")
    df["season"] = season
    df["opp_sp_id"] = df.apply(
        lambda r: r["away_probable_pitcher_id"]
        if r["home_away"] == "H"
        else r["home_probable_pitcher_id"],
        axis=1,
    )
    # Cast to nullable Int64 so the pitcher-feature merge below doesn't blow up
    # with "merging object and Int64" — MLB's schedule endpoint sometimes returns
    # probable pitcher ids as strings, sometimes as NaN, producing an object col.
    df["opp_sp_id"] = pd.to_numeric(df["opp_sp_id"], errors="coerce").astype("Int64")

    if prior_season is not None:
        all_bat = build_blended_batter_features(season, prior_season)
    else:
        all_bat = build_batter_features(season)
    # Drop player_name AND team — both already populated correctly by the
    # lineup fetch (and `team` is the matchup team, which is what we want;
    # bat_m's `team` is the batter's season-stats team and would collide).
    bat_m = all_bat.drop(columns=["player_name", "team"], errors="ignore")
    df = df.merge(
        bat_m,
        left_on=["player_id", "season"],
        right_on=["mlbam_id", "season"],
        how="inner",
    ).drop(columns=["mlbam_id"], errors="ignore")

    if prior_season is not None:
        all_pit = build_blended_pitcher_features(season, prior_season)
    else:
        all_pit = build_pitcher_features(season)
    pit_m = (
        all_pit.rename(columns={"mlbam_id": "opp_pitcher_mlbam_id"})
        # Drop pitcher's team — it's NOT the batter's matchup team. Without
        # this drop, the batter row's `team` column gets overwritten with the
        # opposing pitcher's team (which equals the batter's opponent), making
        # `team == opponent` for every row.
        .drop(columns=["team"], errors="ignore")
    )
    # Match dtype of the left-side merge key so pandas doesn't reject the join.
    pit_m["opp_pitcher_mlbam_id"] = pit_m["opp_pitcher_mlbam_id"].astype("Int64")

    # Diagnostic: probable pitchers populated? If MLB hasn't announced them yet,
    # opp_sp_id is all NaN and every downstream platoon/handedness feature is
    # bogus — we'd rather print a loud warning than silently regress to means.
    n_missing_sp = int(df["opp_sp_id"].isna().sum())
    n_total = len(df)
    if n_total and n_missing_sp == n_total:
        print(f"  WARN: opp_sp_id is NaN for ALL {n_total} rows — "
              f"probable pitchers likely not announced yet for {d}. "
              f"Pitcher features will fall back to league means; platoon flags will be 0.")
    elif n_missing_sp:
        print(f"  WARN: opp_sp_id NaN for {n_missing_sp}/{n_total} rows.")

    df = df.merge(
        pit_m,
        left_on=["opp_sp_id", "season"],
        right_on=["opp_pitcher_mlbam_id", "season"],
        how="left",
    ).drop(columns=["opp_pitcher_mlbam_id"], errors="ignore")
    # Diagnostic: how many actually matched a pitcher row?
    if "pitcher_hand" in df.columns:
        n_ph = int(df["pitcher_hand"].notna().sum())
        print(f"  pitcher features matched for {n_ph}/{n_total} rows")
    stat_cols = [
        "sp_xba_allowed", "sp_k_pct", "sp_hard_hit_allowed", "sp_sweet_spot_allowed",
        "sp_xba_allowed_vs_L", "sp_xba_allowed_vs_R",
        "sp_k_pct_allowed_vs_L", "sp_k_pct_allowed_vs_R",
        "sp_zone_pct", "sp_contact_pct_allowed",
    ]
    league = all_pit[[c for c in stat_cols if c in all_pit.columns]].mean()
    pitcher_low_sample = df["sp_xba_allowed"].isna().astype(int)
    for c in stat_cols:
        if c in df.columns:
            df[c] = df[c].fillna(league[c])
    df["pitcher_low_sample"] = pitcher_low_sample

    pen = pd.read_parquet(clean_path("bullpen_features.parquet"))
    df = df.merge(
        pen[pen["season"] == season].rename(columns={"team": "opponent"}),
        on=["opponent", "season"],
        how="left",
    )

    df["exp_pa"] = df.apply(
        lambda r: expected_pa(int(r["lineup_spot"]), r["home_away"]),
        axis=1,
    )

    df = attach_park(df)

    # Matchup-aware platoon features — mirrors build_features.py.
    import numpy as np
    bh = df.get("batter_hand")
    ph = df.get("pitcher_hand")
    if bh is not None and ph is not None:
        bat_side = bh.where(bh != "S", ph.map({"L": "R", "R": "L"}).fillna("R"))
        df["bat_xba_vs_opphand"] = np.where(
            ph == "L",
            df.get("bat_xba_vs_L", df.get("bat_xba_season")),
            df.get("bat_xba_vs_R", df.get("bat_xba_season")),
        )
        df["bat_k_pct_vs_opphand"] = np.where(
            ph == "L",
            df.get("bat_k_pct_vs_L", df.get("bat_k_pct")),
            df.get("bat_k_pct_vs_R", df.get("bat_k_pct")),
        )
        df["sp_xba_allowed_vs_bathand"] = np.where(
            bat_side == "L",
            df.get("sp_xba_allowed_vs_L", df.get("sp_xba_allowed")),
            df.get("sp_xba_allowed_vs_R", df.get("sp_xba_allowed")),
        )
        df["sp_k_pct_allowed_vs_bathand"] = np.where(
            bat_side == "L",
            df.get("sp_k_pct_allowed_vs_L", df.get("sp_k_pct")),
            df.get("sp_k_pct_allowed_vs_R", df.get("sp_k_pct")),
        )
        # Only claim platoon advantage when we actually know the pitcher's hand.
        # NaN pitcher_hand previously evaluated `bat_side != ph` as True for
        # everyone, falsely flagging the entire slate as platoon-edge.
        ph_known = ph.notna()
        df["platoon_advantage"] = (
            ph_known & ((bh == "S") | (bat_side != ph))
        ).astype(int)
        n_unknown_ph = int((~ph_known).sum())
        if n_unknown_ph:
            print(f"  WARN: pitcher_hand missing for {n_unknown_ph} rows — "
                  f"platoon_advantage forced to 0 for those (probable pitcher unknown "
                  f"or pitcher merge failed; check opp_sp_id matching).")
    else:
        df["bat_xba_vs_opphand"] = df.get("bat_xba_season")
        df["bat_k_pct_vs_opphand"] = df.get("bat_k_pct")
        df["sp_xba_allowed_vs_bathand"] = df.get("sp_xba_allowed")
        df["sp_k_pct_allowed_vs_bathand"] = df.get("sp_k_pct")
        df["platoon_advantage"] = 0

    # Attach rolling 14/30d batter features computed as-of the game date.
    # Uses current-season pitch data; returns empty if that parquet is absent.
    roll = compute_rolling_batter_stats(d, season, windows=(14, 30))
    if not roll.empty:
        roll["mlbam_id"] = roll["mlbam_id"].astype("int64")
        df = df.merge(roll, left_on="player_id", right_on="mlbam_id", how="left")
        df = df.drop(columns=[c for c in ["mlbam_id"] if c in df.columns])
        print(f"  rolling features attached for {roll['PA_14d'].notna().sum() if 'PA_14d' in roll.columns else 0} batters")
    else:
        print("  no current-season pitches parquet yet — rolling features absent (model will fall back to season)")

    # Pitcher rolling 14/30d as-of the game date.
    p_roll_path = clean_path("pitcher_rolling.parquet")
    if p_roll_path.exists():
        p_roll = pd.read_parquet(p_roll_path)
        # Filter to the most recent date <= d for each pitcher in the current season.
        p_roll = p_roll[p_roll["season"] == season]
        p_roll["date"] = p_roll["date"].astype(str)
        cutoff = d.isoformat()
        p_roll = p_roll[p_roll["date"] < cutoff]  # strictly before — matches closed="left"
        if not p_roll.empty:
            p_roll = (
                p_roll.sort_values(["mlbam_id", "date"])
                .groupby("mlbam_id", as_index=False)
                .tail(1)
                .drop(columns=["date", "season"], errors="ignore")
            )
            p_roll = p_roll.rename(columns={"mlbam_id": "opp_sp_id"})
            p_roll["opp_sp_id"] = pd.to_numeric(p_roll["opp_sp_id"], errors="coerce").astype("Int64")
            df["opp_sp_id"] = pd.to_numeric(df["opp_sp_id"], errors="coerce").astype("Int64")
            df = df.merge(p_roll, on="opp_sp_id", how="left")
            n_pr = df["sp_xba_allowed_14d"].notna().sum() if "sp_xba_allowed_14d" in df.columns else 0
            print(f"  pitcher rolling attached for {n_pr} starters")

    df["xba_diff"] = df["bat_xba_season"] - df["sp_xba_allowed"]
    pen_sub = pen[pen["season"] == season]
    pen_mean = (
        pen_sub[["pen_xba_allowed", "pen_k_pct"]].mean()
        if not pen_sub.empty
        else pen[["pen_xba_allowed", "pen_k_pct"]].mean()
    )
    df["pen_xba_allowed"] = df["pen_xba_allowed"].fillna(pen_mean["pen_xba_allowed"])
    df["pen_k_pct"] = df["pen_k_pct"].fillna(pen_mean["pen_k_pct"])
    df["exposure_wtd_opp_xba"] = 0.7 * df["sp_xba_allowed"] + 0.3 * df["pen_xba_allowed"]
    df["date"] = d.isoformat()
    df["batting_order"] = df["lineup_spot"]

    df["p_model"] = predict(df).values

    # Filter E score-side flag: away game OR platoon advantage. The edge/price
    # half of Filter E lives in recommend.py (needs odds). Surfacing here so the
    # daily output makes the platoon edge visible before odds even arrive.
    if "platoon_advantage" not in df.columns:
        df["platoon_advantage"] = 0
    df["platoon_or_away"] = (
        (df["home_away"] == "A") | (df["platoon_advantage"].astype(int) == 1)
    ).astype(int)

    # `pitcher_features_known` is the honest gate for emphasis on starters.
    # When the probable pitcher hasn't been announced (or the merge failed)
    # every pitcher feature is the league mean — the model is making a
    # pitcher-blind prediction. Surface this so downstream filters can require
    # it. This is a stronger guarantee than just `pitcher_low_sample`, which
    # only flags low-TBF; here we flag entirely-missing too.
    if "pitcher_hand" in df.columns:
        df["pitcher_features_known"] = df["pitcher_hand"].notna().astype(int)
    else:
        df["pitcher_features_known"] = 0

    # Surface lineup_confirmed (and start_rate when projected) so recommend.py
    # can optionally gate on confirmed-only.
    if "lineup_confirmed" not in df.columns:
        df["lineup_confirmed"] = True
    if "start_rate" not in df.columns:
        df["start_rate"] = pd.NA

    # Attach recent-form sizing/emphasis signals from boxscores. These are
    # NOT model features (the model wasn't retrained against them); they're
    # downstream layers that drive `recommended_units` (2x stake on hot bats)
    # and a `opp_grind` flag for slate-color visibility on the dashboard.
    # Loading the same season's boxscores parquet that scoring already used
    # for rolling features — same data source, no extra fetch needed.
    box_path = clean_path(f"boxscores_{season}.parquet")
    box_for_recent = None
    if box_path.exists():
        box_for_recent = pd.read_parquet(box_path)
    if prior_season:
        prior_path = clean_path(f"boxscores_{prior_season}.parquet")
        if prior_path.exists():
            prior_box = pd.read_parquet(prior_path)
            box_for_recent = (prior_box if box_for_recent is None
                              else pd.concat([prior_box, box_for_recent], ignore_index=True))
    if box_for_recent is not None and not box_for_recent.empty:
        targets = df[["player_id", "date"]].copy()
        # attach_hot_streak needs player_id + date and returns the same rows
        # in order; we then attach back to df by index alignment.
        hot = attach_hot_streak(targets, box_for_recent)
        for c in ["hot_streak_n_games", "hot_streak_h", "hot_streak_ab",
                  "hot_streak_avg", "hot_streak", "recommended_units"]:
            df[c] = hot[c].values
        # attach_opp_grind also needs an `opp_team` alias for `opponent`.
        targets_g = df[["opp_team" if "opp_team" in df.columns else "opponent",
                         "date"]].copy()
        targets_g.columns = ["opp_team", "date"]
        grind = attach_opp_grind(targets_g, box_for_recent)
        df["opp_consec_games"] = grind["opp_consec_games"].values
        df["opp_grind"] = grind["opp_grind"].values
    else:
        # No boxscores on disk → safe defaults so the parquet schema stays
        # stable and recommend.py / the dashboard never see KeyError.
        df["hot_streak"] = 0
        df["recommended_units"] = 1.0
        df["opp_grind"] = 0
        df["opp_consec_games"] = 0
        df["hot_streak_avg"] = pd.NA

    cols_out = [
        "date",
        "game_pk",
        "player_id",
        "player_name",
        "team",
        "opponent",
        "home_away",
        "lineup_spot",
        "lineup_confirmed",
        "start_rate",
        "exp_pa",
        "batter_hand",
        "pitcher_hand",
        "platoon_advantage",
        "platoon_or_away",
        "pitcher_features_known",
        "pitcher_low_sample",
        "p_model",
        # Recent-form columns (sizing/emphasis layers, not model features):
        "hot_streak",
        "hot_streak_avg",
        "recommended_units",
        "opp_grind",
        "opp_consec_games",
    ]
    cols_out = [c for c in cols_out if c in df.columns]
    out = df[cols_out].sort_values("p_model", ascending=False)
    out.to_parquet(output_path("predictions", f"{d.isoformat()}.parquet"), index=False)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default today)")
    parser.add_argument("--season", type=int, default=None, help="current season (default: year of date)")
    parser.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Prior season for PA-weighted blend. Recommended early/mid current season.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help=(
            "How many rows to PRINT to the terminal (default 50). The full prediction "
            "set always lands in the predictions parquet — this is just a display cap. "
            "Pass 0 to print every row."
        ),
    )
    parser.add_argument(
        "--no-projection",
        action="store_true",
        help=(
            "Disable projected lineups (revert to confirmed-only). Only use "
            "this for diagnostic comparisons against the pre-v1 behavior — "
            "you'll see the slate collapse to whichever games already had "
            "actual lineups posted."
        ),
    )
    args = parser.parse_args()

    d = date.fromisoformat(args.date) if args.date else date.today()
    season = args.season or d.year

    out = score_for_date(
        d,
        season=season,
        prior_season=args.prior_season,
        use_projection=not args.no_projection,
    )
    if out.empty:
        raise SystemExit(0)

    # Mark platoon-advantage picks with a leading * so they pop in the printed
    # output. The parquet itself keeps the clean numeric platoon_advantage col.
    display = out.copy()
    if "platoon_advantage" in display.columns:
        display["pick"] = display.apply(
            lambda r: f"* {r['player_name']}" if int(r.get("platoon_advantage", 0) or 0) == 1
            else f"  {r['player_name']}",
            axis=1,
        )
        order = ["pick"] + [c for c in display.columns if c not in ("pick", "player_name")]
        display = display[order]

    n_total = len(out)
    top_n = n_total if args.top in (0, None) else min(args.top, n_total)
    print()
    print("=" * 60)
    print(f"TOP {top_n} PICKS BY P_MODEL  (* = platoon advantage)")
    print(f"  {n_total} batters scored; full set written to predictions/{d.isoformat()}.parquet")
    print("  NOTE: this is RANKED BY p_model only. Bet selection still requires odds:")
    print("    `python -m mlbhit.pipeline.fetch_prop_odds --date {d}`")
    print("    `python -m mlbhit.pipeline.recommend --date {d} --filter-e --require-pitcher`")
    print("=" * 60)
    print(display.head(top_n).to_string(index=False))

    # Separate spotlight for platoon-advantage batters — the score-side half of
    # Filter E. Edge/price filtering happens in recommend.py once odds land.
    if "platoon_advantage" in out.columns:
        plat = out[out["platoon_advantage"].astype(int) == 1].head(15)
        if not plat.empty:
            print()
            print("=" * 60)
            print(f"PLATOON-ADVANTAGE SPOTLIGHT  ({len(plat)} of top picks)")
            print("=" * 60)
            spot_cols = [c for c in [
                "player_name", "team", "opponent", "home_away",
                "batter_hand", "pitcher_hand", "lineup_spot", "exp_pa", "p_model",
            ] if c in plat.columns]
            print(plat[spot_cols].to_string(index=False))
