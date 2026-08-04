"""Instruments as cash-flow generators.

Nothing here discounts anything. An instrument knows its own dates, day count
and coupon; turning that into a price requires a curve, and that happens in
``yieldcurve.curves.pricing``. Keeping the split means the floating-rate note needs no
special pricer of its own, and a shocked curve reprices every instrument without
any instrument knowing a shock occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from yieldcurve.calendars import Calendar
from yieldcurve.conventions import (
    BusinessDayConvention,
    DayCount,
    SchedulePeriod,
    schedule_periods,
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


def _payment_dates(periods: tuple[SchedulePeriod, ...]) -> tuple[date, ...]:
    return (periods[0].accrual_start, *(period.payment_date for period in periods))


def _period_year_fraction(period: SchedulePeriod, day_count: DayCount, frequency: int) -> float:
    return year_fraction(
        period.accrual_start,
        period.accrual_end,
        day_count,
        period_start=period.reference_start,
        period_end=period.reference_end,
        frequency=frequency,
    )


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

    def coupon_periods(self) -> tuple[SchedulePeriod, ...]:
        return schedule_periods(self.issue, self.maturity, self.frequency, self.calendar, self.bdc)

    def coupon_dates(self) -> tuple[date, ...]:
        return _payment_dates(self.coupon_periods())

    def cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        flows = []
        for period in self.coupon_periods():
            if period.payment_date <= asof:
                continue
            amount = (
                self.face
                * self.coupon
                * _period_year_fraction(period, self.day_count, self.frequency)
            )
            flows.append(CashFlow(period.payment_date, amount))
        if flows:
            flows[-1] = CashFlow(flows[-1].date, flows[-1].amount + self.face)
        return tuple(flows)

    def _period_at(self, asof: date) -> SchedulePeriod:
        periods = self.coupon_periods()
        if not periods[0].accrual_start <= asof <= periods[-1].accrual_end:
            raise ValueError(
                f"{asof} lies outside the bond's life "
                f"{periods[0].accrual_start}..{periods[-1].accrual_end}"
            )
        for period in periods:
            if period.accrual_start <= asof < period.accrual_end:
                return period
        return periods[-1]

    def accrual_period(self, asof: date) -> tuple[date, date]:
        """The unadjusted coupon period containing ``asof``."""
        period = self._period_at(asof)
        return period.accrual_start, period.accrual_end

    def accrued(self, asof: date) -> float:
        """Accrued interest per ``face``, on the bond's own day count."""
        period = self._period_at(asof)
        if asof == period.accrual_start:
            return 0.0
        fraction = year_fraction(
            period.accrual_start,
            asof,
            self.day_count,
            period_start=period.reference_start,
            period_end=period.reference_end,
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

    def coupon_periods(self) -> tuple[SchedulePeriod, ...]:
        return schedule_periods(self.issue, self.maturity, self.frequency, self.calendar, self.bdc)

    def coupon_dates(self) -> tuple[date, ...]:
        return _payment_dates(self.coupon_periods())

    def cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        raise NotImplementedError(
            "An FRN's coupons are unknown without a forecast curve. "
            "Use yieldcurve.curves.pricing.price(frn, curves, asof)."
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

    def fixed_periods(self) -> tuple[SchedulePeriod, ...]:
        return schedule_periods(
            self.start, self.maturity, self.fixed_frequency, self.calendar, self.bdc
        )

    def fixed_schedule(self) -> tuple[date, ...]:
        return _payment_dates(self.fixed_periods())

    def fixed_cashflows(self, asof: date) -> tuple[CashFlow, ...]:
        flows = []
        for period in self.fixed_periods():
            if period.payment_date <= asof:
                continue
            tau = _period_year_fraction(period, self.fixed_day_count, self.fixed_frequency)
            amount = self.notional * self.fixed_rate * tau
            flows.append(CashFlow(period.payment_date, amount))
        return tuple(flows)

    def float_periods(self) -> tuple[SchedulePeriod, ...]:
        return schedule_periods(
            self.start,
            self.maturity,
            tenor_to_frequency(self.float_tenor),
            self.calendar,
            self.bdc,
        )

    def float_schedule(self) -> tuple[date, ...]:
        return _payment_dates(self.float_periods())


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

    def fixed_periods(self) -> tuple[SchedulePeriod, ...]:
        return schedule_periods(
            self.start, self.maturity, self.fixed_frequency, self.calendar, self.bdc
        )

    def fixed_schedule(self) -> tuple[date, ...]:
        return _payment_dates(self.fixed_periods())

    def float_periods(self) -> tuple[SchedulePeriod, ...]:
        return self.fixed_periods()

    def float_schedule(self) -> tuple[date, ...]:
        return _payment_dates(self.float_periods())


@dataclass(frozen=True)
class Swaption:
    """A European option to enter ``swap`` at ``expiry``."""

    expiry: date
    swap: VanillaSwap
    strike: float
    pay_fixed: bool = True


Instrument = Bill | FixedCouponBond | FRN | VanillaSwap | OIS
