from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import REPO_ROOT


def load_stadiums() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "config" / "stadiums.csv")


def attach_park(df: pd.DataFrame) -> pd.DataFrame:
    s = load_stadiums().rename(columns={"team": "park_team_abbr"})
    out = df.copy()
    is_home = out["home_away"] == "H"
    out["_park_team"] = np.where(is_home, out["team"], out["opponent"])
    return out.merge(s, left_on="_park_team", right_on="park_team_abbr", how="left").drop(
        columns=["_park_team", "park_team_abbr"], errors="ignore"
    )
