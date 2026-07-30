"""Instruments as cash-flow generators.

Nothing here discounts anything. An instrument knows its own dates, day count
and coupon; turning that into a price requires a curve, and that happens in
``curveengine.pricing``. Keeping the split means the floating-rate note needs no
special pricer of its own, and a shocked curve reprices every instrument without
any instrument knowing a shock occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from curveengine.calendars import Calendar
from curveengine.conventions import (
    BusinessDayConvention,
    DayCount,
    schedule,
    year_fraction,
)

_TENOR_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12, "1Y": 12}


def tenor_to_frequency(tenor: str) -> int:
    """Payments per year implied by a tenor label such as ``"3M"``."""
    if tenor == "ON":
        raise ValueError("Overnight legs compound daily; use OIS, not a fixed frequency")
    try:
        return 12 // _TENOR_MONTHS[tenor]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tenor {tenor!r}; expected one of {sorted(_TENOR_MONTHS)}"
        ) from exc


@dataclass(frozen=True)
class CashFlow:
    """A dated payment."""

    date: date
    amount: float


@dataclass(frozen=True)
class Bill:
    """A zero-coupon discount instrument paying face at maturity."""

    maturity: date
    day_count: DayCount = DayCount.ACT_360
    face: float = 100.0

    def cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        if self.maturity <= asof:
            return ()
        return (CashFlow(self.maturity, self.face),)


@dataclass(frozen=True)
class FixedCouponBond:
    """A bullet fixed-coupon bond. ``coupon`` is the annual rate as a decimal."""

    issue: date
    maturity: date
    coupon: float
    frequency: int
    day_count: DayCount
    calendar: Calendar
    bdc: BusinessDayConvention
    face: float = 100.0

    def coupon_dates(self) -> tuple[date, ...]:
        return schedule(self.issue, self.maturity, self.frequency, self.calendar, self.bdc)

    def cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        dates = self.coupon_dates()
        coupon_amount = self.face * self.coupon / self.frequency
        flows = [CashFlow(d, coupon_amount) for d in dates[1:] if d > asof]
        if flows:
            last = flows[-1]
            flows[-1] = CashFlow(last.date, last.amount + self.face)
        return tuple(flows)

    def accrual_period(self, asof: date) -> tuple[date, date]:
        """The coupon period containing ``asof``, as (period start, period end)."""
        dates = self.coupon_dates()
        if not dates[0] <= asof <= dates[-1]:
            raise ValueError(f"{asof} lies outside the bond's life {dates[0]}..{dates[-1]}")
        for previous, following in pairwise(dates):
            if previous <= asof < following:
                return previous, following
        return dates[-2], dates[-1]

    def accrued(self, asof: date) -> float:
        """Accrued interest per ``face``, on the bond's own day count."""
        period_start, period_end = self.accrual_period(asof)
        fraction = year_fraction(
            period_start,
            asof,
            self.day_count,
            period_start=period_start,
            period_end=period_end,
            frequency=self.frequency,
        )
        return self.face * self.coupon * fraction


@dataclass(frozen=True)
class FRN:
    """A floating-rate note paying an index plus a fixed spread."""

    issue: date
    maturity: date
    frequency: int
    day_count: DayCount
    calendar: Calendar
    bdc: BusinessDayConvention
    index_tenor: str
    spread: float
    face: float = 100.0

    def coupon_dates(self) -> tuple[date, ...]:
        return schedule(self.issue, self.maturity, self.frequency, self.calendar, self.bdc)

    def cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        raise NotImplementedError(
            "An FRN's coupons are unknown without a forecast curve. "
            "Use curveengine.pricing.price(frn, curves, asof)."
        )


@dataclass(frozen=True)
class VanillaSwap:
    """A fixed-for-floating interest rate swap."""

    start: date
    maturity: date
    fixed_rate: float
    fixed_frequency: int
    fixed_day_count: DayCount
    float_tenor: str
    float_day_count: DayCount
    calendar: Calendar
    bdc: BusinessDayConvention
    notional: float = 1_000_000.0
    pay_fixed: bool = True

    def fixed_schedule(self) -> tuple[date, ...]:
        return schedule(self.start, self.maturity, self.fixed_frequency, self.calendar, self.bdc)

    def fixed_cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        dates = self.fixed_schedule()
        flows = []
        for previous, payment_date in pairwise(dates):
            if payment_date <= asof:
                continue
            tau = year_fraction(
                previous,
                payment_date,
                self.fixed_day_count,
                period_start=previous,
                period_end=payment_date,
                frequency=self.fixed_frequency,
            )
            amount = self.notional * self.fixed_rate * tau
            flows.append(CashFlow(payment_date, amount))
        return tuple(flows)

    def float_schedule(self) -> tuple[date, ...]:
        return schedule(
            self.start,
            self.maturity,
            tenor_to_frequency(self.float_tenor),
            self.calendar,
            self.bdc,
        )


@dataclass(frozen=True)
class OIS:
    """An overnight-index swap. The floating leg compounds the overnight rate."""

    start: date
    maturity: date
    fixed_rate: float
    fixed_frequency: int
    fixed_day_count: DayCount
    calendar: Calendar
    bdc: BusinessDayConvention
    float_day_count: DayCount = DayCount.ACT_360
    notional: float = 1_000_000.0
    pay_fixed: bool = True
    compounded: bool = True

    def fixed_schedule(self) -> tuple[date, ...]:
        return schedule(self.start, self.maturity, self.fixed_frequency, self.calendar, self.bdc)

    def float_schedule(self) -> tuple[date, ...]:
        return self.fixed_schedule()


@dataclass(frozen=True)
class Swaption:
    """A European option to enter ``swap`` at ``expiry``."""

    expiry: date
    swap: VanillaSwap
    strike: float
    pay_fixed: bool = True


Instrument = Bill | FixedCouponBond | FRN | VanillaSwap | OIS
