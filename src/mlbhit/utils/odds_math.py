from __future__ import annotations


def prob_to_american(p: float) -> int:
    p = max(min(p, 0.9999), 0.0001)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / -odds


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    return p_a / s, p_b / s


def ev_per_unit(p_model: float, odds_american: int) -> float:
    """Expected $ return per $1 wagered at given American odds."""
    payout = odds_american / 100 if odds_american > 0 else 100 / -odds_american
    return p_model * payout - (1 - p_model)


def kelly_fraction(p: float, odds_american: int) -> float:
    d = american_to_decimal(odds_american)
    b = d - 1
    q = 1 - p
    return max(0.0, (b * p - q) / b) if b > 0 else 0.0
