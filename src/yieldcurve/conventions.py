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
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from itertools import pairwise

from yieldcurve.calendars import Calendar


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


def _validate_interval(start: date, end: date) -> None:
    if end <= start:
        raise ValueError(f"end {end} must fall after start {start}")


def _validate_frequency(frequency: int) -> None:
    if frequency <= 0 or 12 % frequency != 0:
        raise ValueError(f"frequency must divide 12 evenly, got {frequency}")


def _last_day(year: int, month: int) -> int:
    return (date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day


def _is_month_end(d: date) -> bool:
    return d.day == _last_day(d.year, d.month)


def _anchored_date(anchor: date, months: int, *, anchor_day: int, end_of_month: bool) -> date:
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    last_day = _last_day(year, month)
    day = last_day if end_of_month else min(anchor_day, last_day)
    return date(year, month, day)


def _reference_date(period_start: date, period_end: date, months: int) -> date:
    return _anchored_date(
        period_end,
        months,
        anchor_day=max(period_start.day, period_end.day),
        end_of_month=_is_month_end(period_start) and _is_month_end(period_end),
    )


def _quasi_coupon_periods(
    start: date, end: date, period_start: date, period_end: date, frequency: int
) -> list[tuple[date, date]]:
    step = 12 // frequency
    periods = [(period_start, period_end)]
    offset = -2
    while start < periods[0][0]:
        previous = _reference_date(period_start, period_end, offset * step)
        periods.insert(0, (previous, periods[0][0]))
        offset -= 1
    offset = 1
    while end > periods[-1][1]:
        following = _reference_date(period_start, period_end, offset * step)
        periods.append((periods[-1][1], following))
        offset += 1
    return periods


def _quasi_coupon_fraction(
    start: date, end: date, reference_start: date, reference_end: date, frequency: int
) -> float:
    overlap_start = max(start, reference_start)
    overlap_end = min(end, reference_end)
    if overlap_end <= overlap_start:
        return 0.0
    return (overlap_end - overlap_start).days / (reference_end - reference_start).days / frequency


def _act_act_icma(
    start: date, end: date, period_start: date, period_end: date, frequency: int
) -> float:
    periods = _quasi_coupon_periods(start, end, period_start, period_end, frequency)
    return sum(
        (
            _quasi_coupon_fraction(start, end, reference_start, reference_end, frequency)
            for reference_start, reference_end in periods
        ),
        0.0,
    )


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

    ACT/ACT ICMA requires one regular reference period and the coupon frequency.
    Irregular accrual intervals are decomposed into quasi-coupon periods.
    """
    _validate_interval(start, end)
    if dc is DayCount.ACT_360:
        return (end - start).days / 360.0
    if dc is DayCount.ACT_365F:
        return (end - start).days / 365.0
    if dc is DayCount.THIRTY_360_BOND:
        return _thirty_360_days(start, end) / 360.0
    if period_start is None or period_end is None or frequency is None:
        raise ValueError("ACT/ACT ICMA requires period_start, period_end and frequency")
    if period_end <= period_start:
        raise ValueError(f"period_end {period_end} must fall after period_start {period_start}")
    _validate_frequency(frequency)
    return _act_act_icma(start, end, period_start, period_end, frequency)


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


class BusinessDayConvention(StrEnum):
    """Rules for moving a date that lands on a non-business day."""

    FOLLOWING = "Following"
    MODIFIED_FOLLOWING = "Modified Following"
    PRECEDING = "Preceding"
    UNADJUSTED = "Unadjusted"


@dataclass(frozen=True)
class SchedulePeriod:
    """One coupon period with unadjusted accrual and adjusted payment dates."""

    accrual_start: date
    accrual_end: date
    payment_date: date
    reference_start: date
    reference_end: date

    def __post_init__(self) -> None:
        if self.accrual_end <= self.accrual_start:
            raise ValueError("accrual_end must fall after accrual_start")
        if self.reference_end <= self.reference_start:
            raise ValueError("reference_end must fall after reference_start")


def adjust(d: date, calendar: Calendar, bdc: BusinessDayConvention) -> date:
    """Move ``d`` to a business day under ``bdc``."""
    if bdc is BusinessDayConvention.UNADJUSTED or calendar.is_business_day(d):
        return d
    if bdc is BusinessDayConvention.PRECEDING:
        return _roll(d, calendar, -1)
    forward = _roll(d, calendar, +1)
    if bdc is BusinessDayConvention.MODIFIED_FOLLOWING and forward.month != d.month:
        return _roll(d, calendar, -1)
    return forward


def _roll(d: date, calendar: Calendar, step: int) -> date:
    while not calendar.is_business_day(d):
        d += timedelta(days=step)
    return d


def add_months(d: date, months: int) -> date:
    """Add ``months`` calendar months, clamping to the end of a shorter month."""
    return _anchored_date(d, months, anchor_day=d.day, end_of_month=False)


def _schedule_boundaries(start: date, end: date, frequency: int) -> tuple[tuple[date, ...], date]:
    _validate_interval(start, end)
    _validate_frequency(frequency)
    step = 12 // frequency
    anchored = [end]
    offset = 1
    while True:
        previous = _anchored_date(
            end,
            -offset * step,
            anchor_day=end.day,
            end_of_month=_is_month_end(end),
        )
        if previous <= start:
            return (start, *reversed(anchored)), previous
        anchored.append(previous)
        offset += 1


def schedule_periods(
    start: date,
    end: date,
    frequency: int,
    calendar: Calendar,
    bdc: BusinessDayConvention,
) -> tuple[SchedulePeriod, ...]:
    """Generate canonical coupon periods backwards from the maturity anchor."""
    boundaries, first_reference_start = _schedule_boundaries(start, end, frequency)
    periods = []
    for index, (accrual_start, accrual_end) in enumerate(pairwise(boundaries)):
        reference_start = first_reference_start if index == 0 else accrual_start
        periods.append(
            SchedulePeriod(
                accrual_start,
                accrual_end,
                adjust(accrual_end, calendar, bdc),
                reference_start,
                accrual_end,
            )
        )
    return tuple(periods)


def schedule(
    start: date,
    end: date,
    frequency: int,
    calendar: Calendar,
    bdc: BusinessDayConvention,
) -> tuple[date, ...]:
    """Return the unadjusted start followed by adjusted payment dates."""
    periods = schedule_periods(start, end, frequency, calendar, bdc)
    return (periods[0].accrual_start, *(period.payment_date for period in periods))
