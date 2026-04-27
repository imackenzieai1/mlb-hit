from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..features.batter import build_batter_features
from ..features.bullpen import build_bullpen_features
from ..features.park_weather import attach_park
from ..features.pa import expected_pa
from ..features.pitcher import build_pitcher_features
from ..io import clean_path, modeling_path, raw_path


def _load_schedules(training_seasons: list[int]) -> pd.DataFrame | None:
    sched_dir = REPO_ROOT / "data" / "raw" / "schedule"
    if not sched_dir.exists():
        return None
    paths = []
    for s in training_seasons:
        paths.extend(sorted(sched_dir.glob(f"{s}-*.parquet")))
    if not paths:
        return None
    sched = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return sched.drop_duplicates(subset=["game_pk"], keep="last")


def _ensure_bullpen(training_seasons: list[int]) -> pd.DataFrame:
    path = clean_path("bullpen_features.parquet")
    if not path.exists():
        return build_bullpen_features(training_seasons)
    return pd.read_parquet(path)


def build_modeling_table(training_seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in training_seasons:
        p = clean_path(f"boxscores_{s}.parquet")
        if not p.exists():
            continue
        b = pd.read_parquet(p)
        b["season"] = s
        frames.append(b)
    if not frames:
        raise FileNotFoundError("No boxscores parquet found for training seasons.")
    box = pd.concat(frames, ignore_index=True)

    # Drop spring-training games. The boxscore pull includes every game MLB
    # reports — spring ("S"), regular ("R"), and occasionally exhibition ("E").
    # Statcast only covers regular-season play in its park-installed cameras,
    # so a spring-training row has no matching pitch data and gets dropped by
    # the inner merge below anyway. More importantly, spring lineups, intensity,
    # and rule experiments produce outcome distributions that don't match the
    # live betting market we care about. We filter here to keep training pure.
    spring_gt_frames = []
    for s in training_seasons:
        sc_path = raw_path("statcast", f"pitches_{s}.parquet")
        if not sc_path.exists():
            continue
        gt = pd.read_parquet(sc_path, columns=["game_pk", "game_type"]).drop_duplicates()
        spring_gt_frames.append(gt)
    if spring_gt_frames:
        gt_all = pd.concat(spring_gt_frames, ignore_index=True).drop_duplicates("game_pk")
        gt_all["game_pk"] = gt_all["game_pk"].astype("int64")
        box["game_pk"] = box["game_pk"].astype("int64")
        before_gt = len(box)
        box = box.merge(gt_all, on="game_pk", how="left")
        # Drop confirmed non-regular games (spring, exhibition, postseason).
        # Rows with NaN game_type (statcast hasn't been pulled for that
        # date yet) are KEPT — these are in-season games whose statcast
        # data is simply not on disk, not spring training. Previously this
        # filter dropped NaN too, which silently killed every recent
        # in-season date until statcast was backfilled.
        is_known_non_regular = box["game_type"].notna() & (box["game_type"] != "R")
        n_non_regular = int(is_known_non_regular.sum())
        nan_mask = box["game_type"].isna()
        n_nan = int(nan_mask.sum())
        # Surface which in-season dates have unmatched statcast — these are
        # the rows that previously got silently dropped. Visible drops let
        # us tell "0 rows for 2026-04-24" apart from "this date is missing".
        if n_nan:
            unmatched_dates = (
                box.loc[nan_mask, "date"]
                .astype(str).value_counts().sort_index()
            )
            recent = unmatched_dates.tail(5).to_dict()
            print(f"  statcast-unmatched dates (kept as in-season): "
                  f"{len(unmatched_dates)} dates total; recent: {recent}")
        box = box[~is_known_non_regular].drop(columns=["game_type"])
        print(f"game_type filter: {len(box):,}/{before_gt:,} rows kept "
              f"(dropped {n_non_regular:,} non-regular; kept {n_nan:,} "
              f"statcast-unmatched as in-season)")

    # Filter to ACTUAL STARTING-LINEUP batters. MLB's battingOrder codes are
    # 3-digit: last digit "1" = starter, "2"+ = sub. We previously lost that
    # info when truncating to bo // 100, so batting_order.notna() was a no-op
    # (every sub still has a non-null batting_order).
    # Workaround without re-fetching: for each (game_pk, batting_order), keep
    # the player with the most PAs — that's the starter by construction.
    # Subs typically see 1-2 PA; starters see 3-5. No ties in practice.
    # This is the real V1 "top-10/20 population" filter Ian asked for:
    #   - lifts training base rate from 0.58 to ~0.65 (real starter hit rate)
    #   - reduces outcome variance (starters see 3-5 PAs, subs see 1)
    #   - tightens calibration in the probability range we care about
    before = len(box)
    box = (
        box.sort_values(["game_pk", "batting_order", "pa"], ascending=[True, True, False])
        .drop_duplicates(subset=["game_pk", "batting_order"], keep="first")
        .reset_index(drop=True)
    )
    print(f"starter filter: {len(box):,}/{before:,} rows kept ({len(box)/before:.1%}) — "
          f"new hit rate {box['got_hit'].mean():.3f}")

    players = pd.read_parquet(clean_path("players.parquet"))[["mlbam_id", "fg_id"]]
    box = box.merge(players, left_on="player_id", right_on="mlbam_id", how="left")

    # Backfill mlbam_id from player_id for any players missing from players.parquet
    # (typically 2026 rookies / mid-season call-ups added to MLBAM after the last
    # players-table refresh). player_id IS the MLBAM id directly — the players
    # table only exists to attach fg_id, but its absence shouldn't kill the row.
    # Without this, the inner merge with all_bat below silently dropped ~22% of
    # 2026 starter rows and ~16% of 2025 rows. Cast to nullable Int64 so the
    # merge keys line up cleanly.
    n_missing_in_players = box["mlbam_id"].isna().sum()
    if n_missing_in_players:
        print(f"players.parquet missing {box[box['mlbam_id'].isna()]['player_id'].nunique():,} "
              f"player_ids ({n_missing_in_players:,} rows) — backfilling mlbam_id from player_id")
    box["mlbam_id"] = pd.to_numeric(box["mlbam_id"], errors="coerce").fillna(
        pd.to_numeric(box["player_id"], errors="coerce")
    ).astype("Int64")

    all_bat = pd.concat([build_batter_features(s) for s in training_seasons], ignore_index=True)
    all_bat["mlbam_id"] = pd.to_numeric(all_bat["mlbam_id"], errors="coerce").astype("Int64")
    box = box.merge(all_bat, on=["mlbam_id", "season"], how="inner")

    # Actual starting pitcher per game (from boxscore). This parquet is built
    # by fetch_game_starters.py. For historical games MLB's /schedule endpoint
    # does NOT return probable pitchers, so the schedule-based join gave us
    # NaN on 100% of training rows — which is why sp_* features all had
    # zero importance in the first eval run.
    starters_frames = []
    for s in training_seasons:
        p = clean_path(f"game_starters_{s}.parquet")
        if p.exists():
            starters_frames.append(pd.read_parquet(p))
    if starters_frames:
        starters = pd.concat(starters_frames, ignore_index=True)
        starters = starters.drop_duplicates(subset=["game_pk"], keep="last")
        starters["game_pk"] = starters["game_pk"].astype("int64")
        box["game_pk"] = box["game_pk"].astype("int64")
        box = box.merge(
            starters[["game_pk", "home_starter_id", "away_starter_id"]],
            on="game_pk", how="left",
        )
        box["opp_sp_id"] = np.where(
            box["home_away"] == "H",
            box["away_starter_id"],
            box["home_starter_id"],
        )
        # Cast to a numeric (nullable) dtype so the downstream merge key matches mlbam_id.
        box["opp_sp_id"] = pd.to_numeric(box["opp_sp_id"], errors="coerce").astype("Int64")
        n_matched = box["opp_sp_id"].notna().sum()
        print(f"opp_sp_id populated on {n_matched:,}/{len(box):,} box rows ({n_matched/len(box):.1%})")
    else:
        print("WARN: no game_starters_{season}.parquet files — pitcher features will be zeroed out. "
              "Run `python -m mlbhit.pipeline.fetch_game_starters --season <YEAR>` for each training season.")
        box["opp_sp_id"] = pd.Series([pd.NA] * len(box), dtype="Int64")

    all_pit = pd.concat([build_pitcher_features(s) for s in training_seasons], ignore_index=True)
    pit_m = all_pit.rename(columns={"mlbam_id": "opp_pitcher_mlbam_id"})
    # Match merge-key dtypes — opp_sp_id is Int64 (nullable) since some games
    # may not have a starter resolved. Cast the right side to match.
    pit_m["opp_pitcher_mlbam_id"] = pit_m["opp_pitcher_mlbam_id"].astype("Int64")
    box = box.merge(
        pit_m,
        left_on=["opp_sp_id", "season"],
        right_on=["opp_pitcher_mlbam_id", "season"],
        how="left",
    ).drop(columns=["opp_pitcher_mlbam_id"], errors="ignore")

    stat_cols = [
        "sp_xba_allowed", "sp_k_pct", "sp_hard_hit_allowed", "sp_sweet_spot_allowed",
        "sp_xba_allowed_vs_L", "sp_xba_allowed_vs_R",
        "sp_k_pct_allowed_vs_L", "sp_k_pct_allowed_vs_R",
        "sp_zone_pct", "sp_contact_pct_allowed",
    ]
    league = all_pit[[c for c in stat_cols if c in all_pit.columns]].mean()
    pitcher_low_sample = box["sp_xba_allowed"].isna().astype(int)
    for c in stat_cols:
        if c in box.columns:
            box[c] = box[c].fillna(league[c])
    box["pitcher_low_sample"] = pitcher_low_sample

    pen = _ensure_bullpen(training_seasons)
    box = box.merge(pen.rename(columns={"team": "opponent"}), on=["opponent", "season"], how="left")

    # Rolling 14/30d batter features. Merge on (mlbam_id, date). Leakage-safe
    # because rolling.py uses closed="left" so the current day is excluded.
    roll_path = clean_path("batter_rolling.parquet")
    if roll_path.exists():
        rolling = pd.read_parquet(roll_path)
        before = len(box)
        # Cast types to match the box df's merge keys.
        rolling["mlbam_id"] = rolling["mlbam_id"].astype("int64")
        rolling["date"] = rolling["date"].astype(str)
        box["date"] = box["date"].astype(str)
        box["mlbam_id"] = pd.to_numeric(box["mlbam_id"], errors="coerce").astype("Int64")
        rolling["mlbam_id"] = rolling["mlbam_id"].astype("Int64")
        box = box.merge(rolling.drop(columns=["season"], errors="ignore"),
                        on=["mlbam_id", "date"], how="left")
        # Rolling NaN is a signal ("no recent form"), not missing-at-random — we let
        # XGBoost route NaN via its default direction. Do NOT median-fill here.
        assert len(box) == before, "BUG: rolling merge changed row count"
        n_with_roll = box["PA_14d"].notna().sum() if "PA_14d" in box.columns else 0
        print(f"rolling features merged: {n_with_roll:,}/{before:,} rows have 14d history")
    else:
        print(f"WARN: no {roll_path.name} — rolling features absent. Run "
              f"`python -m mlbhit.features.rolling` to build them.")

    # Pitcher rolling 14/30d. Same logic but join key is (opp_sp_id, date).
    p_roll_path = clean_path("pitcher_rolling.parquet")
    if p_roll_path.exists():
        p_roll = pd.read_parquet(p_roll_path)
        before = len(box)
        p_roll = p_roll.rename(columns={"mlbam_id": "opp_sp_id"})
        p_roll["opp_sp_id"] = pd.to_numeric(p_roll["opp_sp_id"], errors="coerce").astype("Int64")
        p_roll["date"] = p_roll["date"].astype(str)
        box["opp_sp_id"] = pd.to_numeric(box["opp_sp_id"], errors="coerce").astype("Int64")
        box = box.merge(p_roll.drop(columns=["season"], errors="ignore"),
                        on=["opp_sp_id", "date"], how="left")
        assert len(box) == before, "BUG: pitcher rolling merge changed row count"
        n_with_p_roll = box["sp_xba_allowed_14d"].notna().sum() if "sp_xba_allowed_14d" in box.columns else 0
        print(f"pitcher rolling merged: {n_with_p_roll:,}/{before:,} rows have 14d sp history")
    else:
        print(f"WARN: no {p_roll_path.name} — pitcher rolling absent. Run "
              f"`python -m mlbhit.features.pitcher_rolling` to build it.")

    # Matchup-aware platoon features. For each row select the column whose label
    # matches the *opposing* hand, giving us a single per-row platoon-aware
    # feature instead of two perpetually-half-irrelevant ones.
    #
    # Switch hitters (batter_hand == "S") show up in Statcast as whichever side
    # they batted from per PA — _primary_hand() in fetch_stats picks the mode,
    # so they get bucketed into "L" or "R" depending on usage. For matchup
    # purposes we treat the unobserved side as the opposite of the pitcher's
    # hand (i.e. "switch flips to whatever helps") which is the closest cheap
    # approximation without a full pitch-by-pitch L/R indicator.
    bh = box.get("batter_hand")
    ph = box.get("pitcher_hand")
    if bh is not None and ph is not None:
        # Effective batter side: switch hitters take the opposite of the pitcher
        bat_side = bh.where(bh != "S",
                            ph.map({"L": "R", "R": "L"}).fillna("R"))

        # bat_xba_vs_opphand: batter's xBA against pitchers throwing with hand `ph`
        box["bat_xba_vs_opphand"] = np.where(
            ph == "L",
            box.get("bat_xba_vs_L", box.get("bat_xba_season")),
            box.get("bat_xba_vs_R", box.get("bat_xba_season")),
        )
        box["bat_k_pct_vs_opphand"] = np.where(
            ph == "L",
            box.get("bat_k_pct_vs_L", box.get("bat_k_pct")),
            box.get("bat_k_pct_vs_R", box.get("bat_k_pct")),
        )
        # sp_xba_allowed_vs_bathand: pitcher's xBA allowed to batters of side `bat_side`
        box["sp_xba_allowed_vs_bathand"] = np.where(
            bat_side == "L",
            box.get("sp_xba_allowed_vs_L", box.get("sp_xba_allowed")),
            box.get("sp_xba_allowed_vs_R", box.get("sp_xba_allowed")),
        )
        box["sp_k_pct_allowed_vs_bathand"] = np.where(
            bat_side == "L",
            box.get("sp_k_pct_allowed_vs_L", box.get("sp_k_pct")),
            box.get("sp_k_pct_allowed_vs_R", box.get("sp_k_pct")),
        )
        # platoon_advantage = 1 when batter sees opposite-handed pitching
        # (LHB vs RHP or RHB vs LHP). Switch hitters always get the advantage.
        box["platoon_advantage"] = (
            (bh == "S") | (bat_side != ph)
        ).astype(int)
        n_known_hand = ((bh.notna()) & (ph.notna())).sum()
        print(f"platoon features: hand known on {n_known_hand:,}/{len(box):,} rows "
              f"({n_known_hand/len(box):.1%})")
    else:
        box["bat_xba_vs_opphand"] = box.get("bat_xba_season", np.nan)
        box["bat_k_pct_vs_opphand"] = box.get("bat_k_pct", np.nan)
        box["sp_xba_allowed_vs_bathand"] = box.get("sp_xba_allowed", np.nan)
        box["sp_k_pct_allowed_vs_bathand"] = box.get("sp_k_pct", np.nan)
        box["platoon_advantage"] = 0

    def _exp_pa_row(r: pd.Series) -> float:
        spot = int(r["batting_order"]) if pd.notna(r.get("batting_order")) else 5
        return expected_pa(spot, r["home_away"])

    box["exp_pa"] = box.apply(_exp_pa_row, axis=1)

    box = attach_park(box)

    box["xba_diff"] = box["bat_xba_season"] - box["sp_xba_allowed"]
    pen_mean = pen[["pen_xba_allowed", "pen_k_pct"]].mean()
    box["pen_xba_allowed"] = box["pen_xba_allowed"].fillna(pen_mean["pen_xba_allowed"])
    box["pen_k_pct"] = box["pen_k_pct"].fillna(pen_mean["pen_k_pct"])
    box["exposure_wtd_opp_xba"] = 0.7 * box["sp_xba_allowed"] + 0.3 * box["pen_xba_allowed"]

    out = modeling_path("player_game_features.parquet")
    box.to_parquet(out, index=False)
    return box


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        type=str,
        default="2023,2024,2025,2026",
        help=(
            "Comma-separated list of seasons to include. Default 2023-2026 "
            "covers the trained model's seasons plus the live season — "
            "important so historical_backtest can score in-season dates."
        ),
    )
    args = parser.parse_args()
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    df = build_modeling_table(seasons)
    print(df.shape)
    print(df.columns.tolist())
    print("hit rate:", df["got_hit"].mean())
