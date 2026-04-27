from __future__ import annotations

import pandas as pd

from ..config import REPO_ROOT


def load_pa_lookup() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "config" / "pa_lookup.csv")


def expected_pa(lineup_spot: int, home_away: str, game_total: float | None = None) -> float:
    tbl = load_pa_lookup()
    base = tbl[(tbl["lineup_spot"] == lineup_spot) & (tbl["home_away"] == home_away)]["exp_pa"].iloc[0]
    if game_total is not None:
        base += 0.10 * ((game_total - 8.5) / 0.5)
    return float(max(3.0, min(5.5, base)))
