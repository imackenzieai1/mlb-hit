from __future__ import annotations

from datetime import date, datetime, timedelta

import pytz

# Project-wide display + day-boundary timezone. Switched to Central 2026-05-03
# to match the operator's home timezone — "today's slate" rolls over at
# midnight CT, and first-pitch labels render in CT.
CT = pytz.timezone("America/Chicago")

# Backwards-compat alias. Kept so any external script still importing ET
# doesn't break — same Chicago timezone now, just the old name. Remove
# after a deprecation cycle.
ET = CT


def today_ct() -> date:
    return datetime.now(CT).date()


# Backwards-compat alias for today_et — same return value, new name.
today_et = today_ct


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")
