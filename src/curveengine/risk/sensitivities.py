"""Interest-rate sensitivities.

Two families that are easy to confuse and must not be:

* **Analytic (yield-space)** — Macaulay, modified, dollar duration, convexity.
  Differentiate the price/yield relation of one instrument. Defined only where
  a yield is defined, which excludes a par swap.
* **Effective (curve-space)** — effective duration and convexity. Reprice under
  a shifted curve. Defined for anything the pricer prices, and the only correct
  choice once cash flows depend on the curve, as an FRN's do.

On a flat curve they agree. On a sloped curve they differ for a real reason,
and reporting one while calling it the other is a live source of wrong numbers.
"""

from __future__ import annotations

from datetime import date

from curveengine.conventions import year_fraction
from curveengine.curves.protocol import CurveSet, curve_time
from curveengine.instruments import Bill, FixedCouponBond, Instrument
from curveengine.pricing import price, ytm
from curveengine.risk.scenarios import parallel, shift_curveset

_BASIS_POINT = 1e-4

_YieldInstrument = Bill | FixedCouponBond


def _require_yield_instrument(instrument: Instrument) -> None:
    if not isinstance(instrument, (Bill, FixedCouponBond)):
        raise TypeError(
            f"{type(instrument).__name__} has no well-defined yield; "
            "use effective_duration, which reprices under a shifted curve"
        )


def macaulay_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Present-value-weighted mean time to cash flow, in years.

    Weighted with discount factors from the curve, not from the yield. The two
    give the same answer on a flat curve and the curve version is the more
    defensible one when the curve is sloped.
    """
    _require_yield_instrument(instrument)
    flows = instrument.cashflows(asof)  # type: ignore[union-attr]
    if not flows:
        raise ValueError(f"{type(instrument).__name__} has no cash flows after {asof}")
    discount = curves.discount
    weighted = 0.0
    total = 0.0
    for flow in flows:
        t = curve_time(asof, flow.date)
        pv = flow.amount * discount.df(t)
        weighted += t * pv
        total += pv
    return weighted / total


def modified_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Macaulay duration discounted once more at the instrument's own yield."""
    _require_yield_instrument(instrument)
    y = ytm(instrument, price(instrument, curves, asof).dirty, asof)  # type: ignore[arg-type]
    frequency = getattr(instrument, "frequency", 1)
    flows = instrument.cashflows(asof)  # type: ignore[union-attr]
    if not flows:
        raise ValueError(f"{type(instrument).__name__} has no cash flows after {asof}")

    if isinstance(instrument, FixedCouponBond):
        period_start, period_end = instrument.accrual_period(asof)
        w = (
            year_fraction(
                asof,
                period_end,
                instrument.day_count,
                period_start=period_start,
                period_end=period_end,
                frequency=frequency,
            )
            * frequency
        )
        weighted = 0.0
        total = 0.0
        for k, flow in enumerate(flows):
            periods = w + k
            t = periods / frequency
            df = 1.0 / (1.0 + y / frequency) ** periods
            pv = flow.amount * df
            weighted += t * pv
            total += pv
    else:
        weighted = 0.0
        total = 0.0
        for flow in flows:
            t = curve_time(asof, flow.date)
            periods = t * frequency
            df = 1.0 / (1.0 + y / frequency) ** periods
            pv = flow.amount * df
            weighted += t * pv
            total += pv

    mac_yield = weighted / total
    return mac_yield / (1.0 + y / frequency)


def dollar_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Modified duration in price units rather than percent."""
    return modified_duration(instrument, curves, asof) * price(instrument, curves, asof).dirty


def dv01(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Price change for a one-basis-point rise, reported positive when long.

    Computed by repricing rather than from dollar duration, so it stays correct
    for instruments where the analytic route is unavailable. The equality with
    ``dollar_duration * 1e-4`` is then a test, not a definition.
    """
    base = price(instrument, curves, asof).dirty
    up = price(instrument, shift_curveset(curves, parallel(_BASIS_POINT)), asof).dirty
    return base - up


def convexity(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Second derivative of price with respect to yield, divided by price.

    Do not compare this number directly against another library's convexity:
    the conventions differ. Compare the price change it predicts. See spec
    section 4.3 and ``tests/parity/test_quantlib_risk.py``.
    """
    _require_yield_instrument(instrument)
    y = ytm(instrument, price(instrument, curves, asof).dirty, asof)  # type: ignore[arg-type]
    frequency = getattr(instrument, "frequency", 1)
    flows = instrument.cashflows(asof)  # type: ignore[union-attr]
    dirty = price(instrument, curves, asof).dirty

    if isinstance(instrument, FixedCouponBond):
        period_start, period_end = instrument.accrual_period(asof)
        w = (
            year_fraction(
                asof,
                period_end,
                instrument.day_count,
                period_start=period_start,
                period_end=period_end,
                frequency=frequency,
            )
            * frequency
        )
        total = 0.0
        for k, flow in enumerate(flows):
            periods = w + k
            discounted = flow.amount / (1.0 + y / frequency) ** periods
            total += discounted * periods * (periods + 1)
    else:
        total = 0.0
        for flow in flows:
            t = curve_time(asof, flow.date)
            periods = t * frequency
            discounted = flow.amount / (1.0 + y / frequency) ** periods
            total += discounted * periods * (periods + 1)

    return total / (dirty * frequency**2 * (1.0 + y / frequency) ** 2)


def effective_duration(
    instrument: Instrument, curves: CurveSet, asof: date, *, bump: float = _BASIS_POINT
) -> float:
    """Central difference of price under a parallel zero shift.

    Central rather than one-sided: the one-sided error is O(bump) and, for a
    bond with any convexity at all, biases duration systematically.
    """
    base = price(instrument, curves, asof).dirty
    up = price(instrument, shift_curveset(curves, parallel(bump)), asof).dirty
    down = price(instrument, shift_curveset(curves, parallel(-bump)), asof).dirty
    return (down - up) / (2.0 * bump * base)


def effective_convexity(
    instrument: Instrument, curves: CurveSet, asof: date, *, bump: float = _BASIS_POINT
) -> float:
    """Central second difference under a parallel zero shift.

    The bump default is 1bp. Second differences divide by ``bump**2``, so at
    1e-6 the floating-point noise in the prices dominates the signal; 1e-4 is
    comfortably in the stable region for double precision, which
    ``test_effective_convexity_is_independent_of_bump_size`` verifies.
    """
    base = price(instrument, curves, asof).dirty
    up = price(instrument, shift_curveset(curves, parallel(bump)), asof).dirty
    down = price(instrument, shift_curveset(curves, parallel(-bump)), asof).dirty
    return (up + down - 2.0 * base) / (base * bump**2)
