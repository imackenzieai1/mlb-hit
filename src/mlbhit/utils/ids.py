from __future__ import annotations

import re

from pybaseball import playerid_lookup


def normalize_name(name: str) -> str:
    n = name.strip()
    n = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", n, flags=re.IGNORECASE)
    n = n.replace(".", "").replace(",", "")
    return n.lower()


def lookup_mlbam(first: str, last: str) -> int | None:
    try:
        df = playerid_lookup(last, first, fuzzy=True)
        if df.empty:
            return None
        active = df.sort_values("mlb_played_last", ascending=False).iloc[0]
        return int(active["key_mlbam"])
    except Exception:
        return None
