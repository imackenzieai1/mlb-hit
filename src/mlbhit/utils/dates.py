from __future__ import annotations

from datetime import date, datetime, timedelta

import pytz

ET = pytz.timezone("America/New_York")


def today_et() -> date:
    return datetime.now(ET).date()


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")
