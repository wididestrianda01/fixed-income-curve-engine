"""Sequential bootstrap: quoted instruments to a discount curve.

The algorithm is the same for every instrument type. Sort by maturity; for each
quote in turn, every discount factor it depends on is already known except the one
at its own maturity, so solve for that one by root-finding on the instrument's own
pricing function. Instruments are priced by ``yieldcurve.curves.pricing``, not by
bespoke formulas here, which is what keeps the bootstrap consistent with
valuation by construction rather than by coincidence.

The bootstrap fits its inputs exactly. That is its strength as a pricing curve
and its weakness as a description of the market: one stale quote bends the curve
in its own neighbourhood and nothing smooths it away. Task 2.3's parametric fit
is the counterpart, and Notebook 03 shows the same data under both.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import brentq

from yieldcurve.conventions import year_fraction
from yieldcurve.curves.interpolation import (
    CurveConstructionError,
    InterpMethod,
    InterpolatedDiscountCurve,
)
from yieldcurve.curves.pricing import par_rate, price
from yieldcurve.curves.protocol import CurveSet, DiscountCurve, curve_time
from yieldcurve.instruments import OIS, Bill, FixedCouponBond, VanillaSwap

_DF_BRACKET = (1e-8, 5.0)
_SOLVER_TOLERANCE = 1e-15


@dataclass(frozen=True)
class Quote:
    """A market quote on an instrument.

    ``rate`` is read according to the instrument: a simple money-market rate for a
    ``Bill``, the par coupon for a ``FixedCouponBond`` (which therefore prices to
    100), and the par swap rate for a ``VanillaSwap`` or ``OIS``.
    """

    instrument: object
    rate: float


def _maturity(instrument: object) -> date:
    match instrument:
        case Bill() | FixedCouponBond() | VanillaSwap() | OIS():
            return instrument.maturity
        case _:
            raise CurveConstructionError(
                f"Cannot bootstrap from {type(instrument).__name__}; "
                "supported: Bill, FixedCouponBond, VanillaSwap, OIS"
            )


def bootstrap(
    quotes: Sequence[Quote],
    asof: date,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
    discount_curve: DiscountCurve | None = None,
) -> InterpolatedDiscountCurve:
    """Build a curve that reprices every quote exactly.

    With ``discount_curve`` omitted the result is a discount curve in its own
    right — the single-curve case, and the right call when bootstrapping OIS.
    Pass an already-built OIS curve to bootstrap a *forecast* curve instead: the
    solved curve then projects the floating leg while the quoted swap is
    discounted off OIS, which is how a dual-curve build is done post-2008.
    Discounting a swap off its own projection curve is the pre-crisis convention
    and misprices the basis.
    """
    if not quotes:
        raise CurveConstructionError("Cannot bootstrap a curve from no quotes")

    ordered = sorted(quotes, key=lambda q: _maturity(q.instrument))
    maturities = [_maturity(q.instrument) for q in ordered]
    if any(m <= asof for m in maturities):
        raise CurveConstructionError(
            f"Every instrument must mature after {asof}; the earliest matures on {maturities[0]}"
        )
    if len(set(maturities)) != len(maturities):
        duplicates = sorted(m for m, count in Counter(maturities).items() if count > 1)
        raise CurveConstructionError(f"Two instruments share the same maturity: {duplicates}")

    times: list[float] = []
    dfs: list[float] = []
    for quote in ordered:
        t = curve_time(asof, _maturity(quote.instrument))
        next_df = _solve_next_df(quote, asof, times, dfs, t, method, discount_curve)
        times.append(t)
        dfs.append(next_df)

    return InterpolatedDiscountCurve(
        reference_date=asof, times=tuple(times), dfs=tuple(dfs), method=method
    )


def _solve_next_df(
    quote: Quote,
    asof: date,
    times: list[float],
    dfs: list[float],
    t: float,
    method: InterpMethod,
    discount_curve: DiscountCurve | None,
) -> float:
    """The discount factor at ``t`` that makes ``quote`` price to its quoted level."""
    instrument = quote.instrument

    if isinstance(instrument, Bill):
        # Closed form: no earlier discount factor enters, so no solve is needed.
        tau = year_fraction(asof, instrument.maturity, instrument.day_count)
        return 1.0 / (1.0 + quote.rate * tau)

    def residual(candidate: float) -> float:
        trial = InterpolatedDiscountCurve(
            reference_date=asof,
            times=(*times, t),
            dfs=(*dfs, candidate),
            method=method,
        )
        curves = (
            CurveSet.single(trial)
            if discount_curve is None
            else CurveSet(discount=discount_curve, forecast=defaultdict(lambda: trial))
        )
        if isinstance(instrument, FixedCouponBond):
            return price(instrument, curves, asof).dirty - 100.0
        if not isinstance(instrument, VanillaSwap | OIS):
            raise TypeError(
                f"_solve_next_df expects Bill, FixedCouponBond, VanillaSwap, or OIS; "
                f"got {type(instrument).__name__}"
            )
        return par_rate(instrument, curves, asof) - quote.rate

    low, high = _DF_BRACKET
    high = min(high, dfs[-1] * 1.5) if dfs else high
    if residual(low) * residual(high) > 0.0:
        raise CurveConstructionError(
            f"No discount factor in [{low}, {high}] reprices {instrument} at {quote.rate}. "
            "The quote is inconsistent with the shorter instruments already bootstrapped."
        )
    return float(brentq(residual, low, high, xtol=_SOLVER_TOLERANCE))


def discount_factors_from_cashflow_matrix(cashflows: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Solve ``CF d = P`` for the discount factor vector ``d``.

    The linear-algebra statement of bootstrapping: when a set of bonds shares its
    payment dates, ``CF`` is lower triangular and the whole curve falls out of one
    solve. Real quote sets rarely align, which is why ``bootstrap`` above is
    sequential — but the equivalence is worth being able to demonstrate.
    """
    rows, columns = cashflows.shape
    if rows != columns:
        raise CurveConstructionError(
            f"Cash-flow matrix must be square to determine one discount factor per "
            f"instrument, got {rows}x{columns}"
        )
    if prices.shape != (rows,):
        raise CurveConstructionError(f"Expected {rows} prices, got {prices.shape}")
    return np.linalg.solve(cashflows, prices)
