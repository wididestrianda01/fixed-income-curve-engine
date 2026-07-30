"""Valuation. The only module that combines instruments with curves.

Dispatch is an explicit ``match`` rather than a method on each instrument. That
keeps instruments free of curve knowledge, and it puts every valuation formula in
the library on one screen where they can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import cast

from scipy.optimize import brentq

from curveengine.conventions import DayCount, year_fraction
from curveengine.curves.protocol import CurveSet, DiscountCurve, curve_time
from curveengine.instruments import (
    FRN,
    OIS,
    Bill,
    CashFlow,
    FixedCouponBond,
    VanillaSwap,
)

_YTM_BRACKET = (-0.5, 2.0)
_YTM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PricingResult:
    """Dirty price, clean price and accrued interest, per unit of face."""

    dirty: float
    clean: float
    accrued: float


def _pv(flows: tuple[CashFlow, ...], curves: CurveSet, asof: date) -> float:
    curve = curves.discount
    return sum(flow.amount * curve.df(curve_time(asof, flow.date)) for flow in flows)


def price(instrument: object, curves: CurveSet, asof: date) -> PricingResult:
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
            return _price_frn(instrument, curves, asof)
        case VanillaSwap() | OIS():
            dirty = _price_swap(instrument, curves, asof)
            return PricingResult(dirty=dirty, clean=dirty, accrued=0.0)
        case _:
            raise TypeError(
                f"Cannot price {type(instrument).__name__}; "
                "supported: Bill, FixedCouponBond, FRN, VanillaSwap, OIS"
            )


def _price_frn(frn: FRN, curves: CurveSet, asof: date) -> PricingResult:
    """Project coupons off the forecast curve, discount them off the discount curve.

    Those being two different curves is the whole content of multi-curve pricing.
    """
    forecast = curves.forecast_for(frn.index_tenor)
    discount = curves.discount
    dates = [d for d in frn.coupon_dates() if d > asof]
    if not dates:
        return PricingResult(dirty=0.0, clean=0.0, accrued=0.0)

    period_start = max(d for d in frn.coupon_dates() if d <= asof) if dates[0] != asof else asof
    dirty = 0.0
    previous = period_start
    for payment_date in dates:
        tau = year_fraction(previous, payment_date, frn.day_count)
        t1, t2 = curve_time(asof, previous), curve_time(asof, payment_date)
        projected = _simple_forward(forecast, max(t1, 0.0), t2, tau)
        dirty += frn.face * (projected + frn.spread) * tau * discount.df(t2)
        previous = payment_date
    dirty += frn.face * discount.df(curve_time(asof, dates[-1]))
    return PricingResult(dirty=dirty, clean=dirty, accrued=0.0)


def _simple_forward(curve: DiscountCurve, t1: float, t2: float, tau: float) -> float:
    """The simple-compounded forward implied by a curve over [t1, t2].

    Coupons accrue simply, so the projected rate must be the simple forward
    ``(df1/df2 - 1)/tau``, not the continuously compounded ``fwd``.
    """
    df1 = curve.df(t1)
    df2 = curve.df(t2)
    return (df1 / df2 - 1.0) / tau


def _price_swap(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    """Value from the perspective of the fixed-rate payer when ``pay_fixed``."""
    fixed_leg = _fixed_leg_pv(swap, curves, asof)
    floating_leg = _floating_leg_pv(swap, curves, asof)
    net = floating_leg - fixed_leg
    return net if swap.pay_fixed else -net


def _fixed_leg_pv(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    return swap.fixed_rate * swap.notional * annuity(swap, curves, asof)


def _floating_leg_pv(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    discount = curves.discount
    tenor = "ON" if isinstance(swap, OIS) else swap.float_tenor
    forecast = discount if isinstance(swap, OIS) else curves.forecast_for(tenor)
    day_count = DayCount.ACT_360 if isinstance(swap, OIS) else swap.float_day_count
    dates = swap.float_schedule()
    total = 0.0
    for previous, payment_date in pairwise(dates):
        if payment_date <= asof:
            continue
        tau = year_fraction(previous, payment_date, day_count)
        t1 = max(curve_time(asof, previous), 0.0)
        t2 = curve_time(asof, payment_date)
        projected = _simple_forward(forecast, t1, t2, tau)
        total += swap.notional * projected * tau * discount.df(t2)
    return total


def annuity(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    """Sum of accrual-weighted discount factors on the fixed leg, per unit notional."""
    discount = curves.discount
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
        total += tau * discount.df(curve_time(asof, payment_date))
    return total


def par_rate(swap: VanillaSwap | OIS, curves: CurveSet, asof: date) -> float:
    """The fixed rate making ``swap`` worth zero."""
    denominator = annuity(swap, curves, asof) * swap.notional
    if denominator == 0.0:
        raise ValueError("Swap has no remaining fixed payments; par rate is undefined")
    return _floating_leg_pv(swap, curves, asof) / denominator


def ytm(bond: FixedCouponBond, dirty_price: float, asof: date) -> float:
    """Yield to maturity on the street convention, solved by Brent's method.

    The discounting exponents are ``w + k`` where ``w`` is the fraction of the
    current coupon period still to run — the market convention, and the reason
    yield is a quoting device rather than a term structure.
    """
    period_start, period_end = bond.accrual_period(asof)
    w = (period_end - asof).days / (period_end - period_start).days
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
    return cast(float, brentq(residual, low, high, xtol=_YTM_TOLERANCE))
