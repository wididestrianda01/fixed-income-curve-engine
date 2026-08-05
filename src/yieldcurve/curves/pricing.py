"""Valuation. The only module that combines instruments with curves.

Dispatch is an explicit ``match`` rather than a method on each instrument. That
keeps instruments free of curve knowledge, and it puts every valuation formula in
the library on one screen where they can be compared.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from scipy.optimize import brentq

from yieldcurve.calendars import Calendar
from yieldcurve.conventions import (
    BusinessDayConvention,
    DayCount,
    adjust,
    year_fraction,
)
from yieldcurve.curves.protocol import (
    CurveSet,
    DiscountCurve,
    Fixings,
    MissingFixingError,
    curve_time,
)
from yieldcurve.instruments import (
    FRN,
    OIS,
    Bill,
    CashFlow,
    FixedCouponBond,
    Instrument,
    VanillaSwap,
)

_EMPTY = Fixings()
_YTM_BRACKET = (-0.5, 2.0)
_YTM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PricingResult:
    """Dirty price, clean price and accrued interest, per unit of face."""

    dirty: float
    clean: float
    accrued: float


def _df(curves: CurveSet, asof: date, d: date) -> float:
    """Discount from ``asof`` to ``d`` in absolute curve time.

    Every path through this helper divides by D_r(asof) so non-flat curves
    valued on dates other than their reference date stay correct.
    """
    disc = curves.discount
    ref = disc.reference_date
    return disc.df(curve_time(ref, d)) / disc.df(curve_time(ref, asof))


def _pv(flows: tuple[CashFlow, ...], curves: CurveSet, asof: date) -> float:
    return sum(flow.amount * _df(curves, asof, flow.date) for flow in flows)


def price(
    instrument: Instrument, curves: CurveSet, asof: date, *, fixings: Fixings = _EMPTY
) -> PricingResult:
    """Value ``instrument`` on ``curves`` as of ``asof``."""
    match instrument:
        case Bill():
            dirty = _pv(instrument.cashflows(asof), curves, asof)
            return PricingResult(dirty=dirty, clean=dirty, accrued=0.0)
        case FixedCouponBond():
            dirty = _pv(instrument.cashflows(asof), curves, asof)
            accrued = instrument.accrued(asof)
            return PricingResult(dirty=dirty, clean=dirty - accrued, accrued=accrued)
        case FRN():
            return _price_frn(instrument, curves, asof, fixings)
        case VanillaSwap():
            dirty = _price_swap(instrument, curves, asof, fixings)
            return PricingResult(dirty=dirty, clean=dirty, accrued=0.0)
        case OIS():
            dirty = _price_ois(instrument, curves, asof, fixings)
            return PricingResult(dirty=dirty, clean=dirty, accrued=0.0)
        case _:
            raise TypeError(
                f"Cannot price {type(instrument).__name__}; "
                "supported: Bill, FixedCouponBond, FRN, VanillaSwap, OIS"
            )


# -- fixed-coupon bond helpers ------------------------------------------------


def ytm(bond: FixedCouponBond, dirty_price: float, asof: date) -> float:
    """Yield to maturity on the street convention, solved by Brent's method.

    The discounting exponents are ``w + k`` where ``w`` is the fraction of the
    current coupon period still to run — the market convention, and the reason
    yield is a quoting device rather than a term structure.
    """
    period_start, period_end = bond.accrual_period(asof)
    w = (
        year_fraction(
            asof,
            period_end,
            bond.day_count,
            period_start=period_start,
            period_end=period_end,
            frequency=bond.frequency,
        )
        * bond.frequency
    )
    flows = bond.cashflows(asof)
    if not flows:
        raise ValueError(f"{bond} has no cash flows remaining after {asof}")
    f = bond.frequency

    def residual(y: float) -> float:
        return float(
            sum(flow.amount / (1.0 + y / f) ** (w + k) for k, flow in enumerate(flows))
            - dirty_price
        )

    low, high = _YTM_BRACKET
    if residual(low) * residual(high) > 0.0:
        raise ValueError(f"No yield in [{low}, {high}] reproduces a dirty price of {dirty_price}")
    return _brentq(residual, low, high, xtol=_YTM_TOLERANCE)


# -- FRN pricing --------------------------------------------------------------


def _price_frn(frn: FRN, curves: CurveSet, asof: date, fixings: Fixings) -> PricingResult:
    """Project coupons off the forecast curve, discount off the discount curve."""
    forecast = curves.forecast_for(frn.index_tenor)
    dates = frn.coupon_dates()
    periods = [(start, end) for start, end in pairwise(dates) if end > asof]
    if not periods:
        return PricingResult(dirty=0.0, clean=0.0, accrued=0.0)

    dirty = 0.0
    accrued = 0.0
    for period_start, payment_date in periods:
        tau = year_fraction(period_start, payment_date, frn.day_count)
        if period_start < asof:
            rate = fixings.term_rate(frn.index_tenor, period_start)
        else:
            rate = _forward_rate(forecast, period_start, payment_date, frn.day_count)
        coupon = frn.face * (rate + frn.spread)
        dirty += coupon * tau * _df(curves, asof, payment_date)
        if period_start < asof:
            accrued = coupon * year_fraction(period_start, asof, frn.day_count)
    dirty += frn.face * _df(curves, asof, dates[-1])
    return PricingResult(dirty=dirty, clean=dirty - accrued, accrued=accrued)


# -- forward rate projection ---------------------------------------------------


def _forward_rate(curve: DiscountCurve, start: date, end: date, day_count: DayCount) -> float:
    """The simple-compounded forward over [start, end] in absolute curve time."""
    ref = curve.reference_date
    t1 = curve_time(ref, start)
    t2 = curve_time(ref, end)
    tau = year_fraction(start, end, day_count)
    return _simple_forward(curve, t1, t2, tau)


def _simple_forward(curve: DiscountCurve, t1: float, t2: float, tau: float) -> float:
    if tau == 0.0:
        return 0.0
    df1 = curve.df(t1)
    df2 = curve.df(t2)
    return (df1 / df2 - 1.0) / tau


# -- vanilla swap pricing -----------------------------------------------------


def _price_swap(swap: VanillaSwap, curves: CurveSet, asof: date, fixings: Fixings) -> float:
    """Value from the perspective of the fixed-rate payer when ``pay_fixed``."""
    fixed_leg = _fixed_leg_pv(swap, curves, asof)
    floating_leg = _term_floating_leg_pv(swap, curves, asof, fixings)
    net = floating_leg - fixed_leg
    return net if swap.pay_fixed else -net


def _fixed_leg_pv(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    return swap.fixed_rate * swap.notional * annuity(swap, curves, asof)


def _term_floating_leg_pv(
    swap: VanillaSwap, curves: CurveSet, asof: date, fixings: Fixings | None = None
) -> float:
    forecast = curves.forecast_for(swap.float_tenor)
    day_count = swap.float_day_count
    dates = swap.float_schedule()
    total = 0.0
    for previous, payment_date in pairwise(dates):
        if payment_date <= asof:
            continue
        tau = year_fraction(previous, payment_date, day_count)
        if previous < asof:
            # An active period that has already reset must look its rate up in
            # the fixings; the library never replaces it with a shortened
            # forward over the stub.
            if fixings is None:
                raise MissingFixingError(f"missing term fixing for {swap.float_tenor} @ {previous}")
            rate = fixings.term_rate(swap.float_tenor, previous)
        else:
            rate = _forward_rate(forecast, previous, payment_date, day_count)
        total += swap.notional * rate * tau * _df(curves, asof, payment_date)
    return total


# -- OIS pricing --------------------------------------------------------------


def _price_ois(ois: OIS, curves: CurveSet, asof: date, fixings: Fixings) -> float:
    fixed_leg = _fixed_leg_pv(ois, curves, asof)
    floating_leg = _ois_floating_leg_pv(ois, curves, asof, fixings)
    net = floating_leg - fixed_leg
    return net if ois.pay_fixed else -net


def _ois_floating_leg_pv(
    ois: OIS, curves: CurveSet, asof: date, fixings: Fixings | None = None
) -> float:
    forecast = curves.discount
    day_count = ois.float_day_count
    calendar = ois.calendar
    periods = ois.float_periods()
    # date-only float schedule is _payment_dates(…); periods expose accrual_start/end.
    # The leg compounds each business-day overnight fixing inside each period.
    total = 0.0
    for period in periods:
        if period.payment_date <= asof:
            continue
        p_start, p_end = period.accrual_start, period.accrual_end
        tau = year_fraction(p_start, p_end, day_count)
        rate = _ois_period_rate(p_start, p_end, asof, calendar, forecast, day_count, fixings)
        total += ois.notional * rate * tau * _df(curves, asof, period.payment_date)
    return total


def _ois_period_rate(
    p_start: date,
    p_end: date,
    asof: date,
    calendar: Calendar,
    forecast: DiscountCurve,
    day_count: DayCount,
    fixings: Fixings | None = None,
) -> float:
    """Overnight-indexed rate: product of realised fixings * forward remainder.

    Realised overnight observations compound from *p_start* through *asof*.
    The remaining stub from *asof* to *p_end* is the curve forward factor.
    For a fully future period the realised product is 1 and the forward runs
    over the whole period.
    """
    # compound realised overnight fixings [p_start, asof)
    realized = 1.0
    d = _next_business_day(p_start, calendar, BusinessDayConvention.FOLLOWING)
    prev = p_start
    if fixings is not None:
        while d <= asof:
            dt = year_fraction(prev, d, day_count)
            realized *= 1.0 + fixings.overnight_rate(prev) * dt
            prev = d
            d = _next_business_day(d + timedelta(days=1), calendar, BusinessDayConvention.FOLLOWING)
    if prev < asof:
        # asof landed mid-period (non-business-day gap); no more observations,
        # but the forward stub starts from asof, not prev.
        pass
    ref = forecast.reference_date
    if fixings is None:
        forward_factor = forecast.df(curve_time(ref, p_start)) / forecast.df(curve_time(ref, p_end))
    elif prev > p_start:
        forward_factor = forecast.df(curve_time(ref, asof)) / forecast.df(curve_time(ref, p_end))
    else:
        forward_factor = forecast.df(curve_time(ref, p_start)) / forecast.df(curve_time(ref, p_end))
    tau = year_fraction(p_start, p_end, day_count)
    return (realized * forward_factor - 1.0) / tau


# -- annuity, par rate --------------------------------------------------------


def annuity(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    """Sum of accrual-weighted discount factors on the fixed leg, per unit notional."""
    dates = swap.fixed_schedule()
    total = 0.0
    for previous, payment_date in pairwise(dates):
        if payment_date <= asof:
            continue
        tau = year_fraction(
            previous,
            payment_date,
            swap.fixed_day_count,
            period_start=previous,
            period_end=payment_date,
            frequency=swap.fixed_frequency,
        )
        total += tau * _df(curves, asof, payment_date)
    return total


def par_rate(
    swap: VanillaSwap | OIS, curves: CurveSet, asof: date, *, fixings: Fixings = _EMPTY
) -> float:
    """The fixed rate making ``swap`` worth zero.

    ``fixings`` supplies observed reset rates for floating periods that have
    already started as of ``asof``; a mid-life valuation without them raises
    ``MissingFixingError`` (the library never substitutes a shortened forward).
    Spot- and forward-starting swaps have no active periods and need no
    fixings.
    """
    denominator = annuity(swap, curves, asof) * swap.notional
    if denominator == 0.0:
        raise ValueError("Swap has no remaining fixed payments; par rate is undefined")
    match swap:
        case VanillaSwap():
            return _term_floating_leg_pv(swap, curves, asof, fixings) / denominator
        case _:
            return _ois_floating_leg_pv(swap, curves, asof, fixings) / denominator


# -- internal helpers ---------------------------------------------------------


def _next_business_day(d: date, calendar: Calendar, bdc: BusinessDayConvention) -> date:
    """Return the next business day, strictly after ``d``."""
    nxt = d + timedelta(days=1)
    return adjust(nxt, calendar, bdc)


def _brentq(f: Callable[[float], float], a: float, b: float, **kwargs: object) -> float:
    return float(brentq(f, a, b, **kwargs))
