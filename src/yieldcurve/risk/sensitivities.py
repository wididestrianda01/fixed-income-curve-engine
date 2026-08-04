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

Naming conventions (design specification section 4):

* ``fisher_weil_duration`` is the spot-curve-weighted mean time to cash flow;
  it was previously mislabelled "Macaulay".
* ``macaulay_duration`` is the classical YTM-weighted mean time.
* ``dv01`` is the positive loss a long position takes when rates rise 1bp —
  ``base - price(+1bp)`` — not a signed price change.

Normalized measures (anything that divides by the base price) reject a
materially zero present value instead of returning noise; the monetary
alternatives (``dv01``, ``dollar_duration``, portfolio ``bucket_exposure``)
do not divide and stay defined.
"""

from __future__ import annotations

import math
from datetime import date

from yieldcurve.conventions import year_fraction
from yieldcurve.curves.pricing import price, ytm
from yieldcurve.curves.protocol import CurveSet, curve_time
from yieldcurve.instruments import FRN, OIS, Bill, FixedCouponBond, Instrument, VanillaSwap
from yieldcurve.risk.scenarios import parallel, shift_curveset

_BASIS_POINT = 1e-4

MIN_UNIT_PRICE = 1e-8
"""Below this unit price (per unit of the instrument's own face/notional) a
normalized measure is floating-point noise and is rejected."""


def instrument_scale(instrument: Instrument) -> float:
    """The face/notional amount ``price()`` quotes this instrument against.

    One scale contract per instrument family: bonds and bills quote per unit of
    ``face``; swaps quote per unit of ``notional``. The portfolio divides a
    position's notional by this scale to convert a unit price into a book
    value, and the near-zero-PV guards use it to make "materially zero"
    scale-invariant.
    """
    if isinstance(instrument, (Bill, FixedCouponBond, FRN)):
        return instrument.face
    if isinstance(instrument, (VanillaSwap, OIS)):
        return instrument.notional
    raise TypeError(f"{type(instrument).__name__} has no face or notional scale")


def _require_yield_instrument(instrument: Instrument) -> None:
    if not isinstance(instrument, (Bill, FixedCouponBond)):
        raise TypeError(
            f"{type(instrument).__name__} has no well-defined yield; "
            "use effective_duration, which reprices under a shifted curve"
        )


def _require_bond(instrument: Instrument) -> None:
    if not isinstance(instrument, FixedCouponBond):
        raise TypeError(
            f"{type(instrument).__name__} has no well-defined yield "
            "for analytic duration/convexity; use effective_duration"
        )


def _require_bump(bump: float, measure: str) -> None:
    if not math.isfinite(bump) or bump <= 0.0:
        raise ValueError(f"{measure} bump must be a positive finite rate, got {bump}")


def _require_unit_price(base: float, instrument: Instrument, measure: str) -> None:
    """Reject normalizing by a materially zero present value (error policy:
    the code must not normalize by near-zero PV)."""
    if abs(base / instrument_scale(instrument)) < MIN_UNIT_PRICE:
        raise ValueError(
            f"{measure} is undefined for {type(instrument).__name__} with base price "
            f"{base:.6g}: materially zero present value; "
            "use bucket_exposure for the monetary sensitivity"
        )


def fisher_weil_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Spot-curve-weighted mean time to cash flow, in years.

    Weighted with discount factors from the curve, not from the yield. Formerly
    mislabelled ``macaulay_duration``: weighting by the spot curve is the
    Fisher-Weil convention, and differs from classical YTM-weighted Macaulay on
    any sloped curve.
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
    _require_unit_price(total, instrument, "Fisher-Weil duration")
    return weighted / total


def macaulay_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Classical Macaulay duration: present-value-weighted mean time to cash
    flow, in years, weighted at the instrument's own yield to maturity.

    For a zero-coupon Bill the single cash flow makes every weighting scheme
    agree, so the result is the flow's time and no yield is needed.
    """
    _require_yield_instrument(instrument)
    flows = instrument.cashflows(asof)  # type: ignore[union-attr]
    if not flows:
        raise ValueError(f"{type(instrument).__name__} has no cash flows after {asof}")
    frequency = getattr(instrument, "frequency", 1)
    if isinstance(instrument, FixedCouponBond):
        y = ytm(instrument, price(instrument, curves, asof).dirty, asof)
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
    else:  # Bill: a single cash flow has the same mean time under any weights
        weighted = curve_time(asof, flows[0].date) * flows[0].amount
        total = flows[0].amount
    _require_unit_price(total, instrument, "Macaulay duration")
    return weighted / total


def modified_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Macaulay duration discounted once more at the instrument's own yield."""
    _require_bond(instrument)
    y = ytm(instrument, price(instrument, curves, asof).dirty, asof)  # type: ignore[arg-type]
    frequency = getattr(instrument, "frequency", 1)
    return macaulay_duration(instrument, curves, asof) / (1.0 + y / frequency)


def dollar_duration(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Price sensitivity to a one-unit (100%) yield move, in currency per unit
    of the instrument's own face scale."""
    return modified_duration(instrument, curves, asof) * price(instrument, curves, asof).dirty


def dv01(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """Positive loss per 1bp rise in rates for a long position.

    Package convention (design specification section 4): DV01 is ``base - up``
    — the price falls when rates rise, so the number is positive for a long
    position. It is a positive loss, not a signed price change. Computed by
    repricing rather than from dollar duration, so it stays correct for
    instruments where the analytic route is unavailable. The equality with
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
    _require_bond(instrument)
    assert isinstance(instrument, FixedCouponBond)  # _require_bond guarantees this
    y = ytm(instrument, price(instrument, curves, asof).dirty, asof)
    frequency = getattr(instrument, "frequency", 1)
    flows = instrument.cashflows(asof)
    dirty = price(instrument, curves, asof).dirty
    _require_unit_price(dirty, instrument, "Convexity")

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

    return total / (dirty * frequency**2 * (1.0 + y / frequency) ** 2)


def _parallel_prices(
    instrument: Instrument, curves: CurveSet, asof: date, bump: float
) -> tuple[float, float, float]:
    """Base, +bump and -bump dirty prices under a parallel zero shift."""
    base = price(instrument, curves, asof).dirty
    up = price(instrument, shift_curveset(curves, parallel(bump)), asof).dirty
    down = price(instrument, shift_curveset(curves, parallel(-bump)), asof).dirty
    return base, up, down


def effective_duration(
    instrument: Instrument, curves: CurveSet, asof: date, *, bump: float = _BASIS_POINT
) -> float:
    """Central difference of price under a parallel zero shift.

    Central rather than one-sided: the one-sided error is O(bump) and, for a
    bond with any convexity at all, biases duration systematically.
    """
    _require_bump(bump, "effective duration")
    base, up, down = _parallel_prices(instrument, curves, asof, bump)
    _require_unit_price(base, instrument, "Effective duration")
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
    _require_bump(bump, "effective convexity")
    base, up, down = _parallel_prices(instrument, curves, asof, bump)
    _require_unit_price(base, instrument, "Effective convexity")
    return (up + down - 2.0 * base) / (base * bump**2)
