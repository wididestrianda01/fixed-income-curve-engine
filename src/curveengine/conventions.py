"""Day-count conventions and compounding bases.

Two distinct notions of time live in this library and confusing them is the
classic source of small, plausible-looking pricing errors:

* **Curve time** is always ACT/365F years from the curve's reference date. Every
  ``t`` passed to a curve method uses it, without exception.
* **Accrual time** uses the instrument's own day count, and appears only in
  coupon and accrued-interest arithmetic.
"""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum


class DayCount(StrEnum):
    """Day-count conventions used by the markets in scope."""

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360_BOND = "30/360 Bond Basis"
    ACT_ACT_ICMA = "ACT/ACT ICMA"


class Compounding(StrEnum):
    """Compounding bases for quoted rates."""

    SIMPLE = "Simple"
    ANNUAL = "Annual"
    SEMIANNUAL = "Semiannual"
    CONTINUOUS = "Continuous"


_PERIODS_PER_YEAR = {Compounding.ANNUAL: 1, Compounding.SEMIANNUAL: 2}


def _thirty_360_days(start: date, end: date) -> int:
    d1 = min(start.day, 30)
    d2 = end.day
    if d1 == 30:
        d2 = min(d2, 30)
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def year_fraction(
    start: date,
    end: date,
    dc: DayCount,
    *,
    period_start: date | None = None,
    period_end: date | None = None,
    frequency: int | None = None,
) -> float:
    """Return the year fraction between ``start`` and ``end`` under ``dc``.

    The keyword-only arguments apply to ACT/ACT ICMA only, which is undefined
    without the enclosing coupon period and the coupon frequency.
    """
    if dc is DayCount.ACT_360:
        return (end - start).days / 360.0
    if dc is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if dc is DayCount.THIRTY_360_BOND:
        return _thirty_360_days(start, end) / 360.0

    if period_start is None or period_end is None or frequency is None:
        raise ValueError(
            "ACT/ACT ICMA requires period_start, period_end and frequency: "
            "the fraction is defined relative to the enclosing coupon period."
        )
    period_days = (period_end - period_start).days
    if period_days <= 0:
        raise ValueError(f"period_end {period_end} must fall after period_start {period_start}")
    return ((end - start).days / period_days) / frequency


def discount_factor(rate: float, t: float, compounding: Compounding) -> float:
    """Discount factor for ``rate`` over ``t`` years on the given basis."""
    if compounding is Compounding.SIMPLE:
        return 1.0 / (1.0 + rate * t)
    if compounding is Compounding.CONTINUOUS:
        return math.exp(-rate * t)
    n: float = _PERIODS_PER_YEAR[compounding]
    return float((1.0 + rate / n) ** (-n * t))


def to_continuous(rate: float, t: float, compounding: Compounding) -> float:
    """Convert a quoted rate to its continuously compounded equivalent over ``t``.

    Simple rates are horizon-dependent, which is why ``t`` is required rather
    than optional: there is no single continuous rate equivalent to a simple one.
    """
    if compounding is Compounding.CONTINUOUS:
        return rate
    if t <= 0.0:
        raise ValueError(f"t must be positive to convert a rate, got {t}")
    return -math.log(discount_factor(rate, t, compounding)) / t
