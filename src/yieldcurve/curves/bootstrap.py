"""Sequential bootstrap: quoted instruments to a discount curve.

The algorithm is the same for every instrument type. With quotes sorted by
maturity, every discount factor a quote depends on is already known except the
one at its own maturity, so solve for that one by root-finding on the
instrument's own pricing function. Instruments are priced by
``yieldcurve.curves.pricing``, not by bespoke formulas here, which is what
keeps the bootstrap consistent with valuation by construction rather than by
coincidence.

The canonical method is log-linear discount-factor interpolation, and only the
canonical method is claimed to reprice every input quote exactly: its
interpolant on an interval depends only on the knots bounding that interval, so
adding a later pillar never changes an earlier solve. Global interpolators
(cubic log-DF, monotone convex) do not share that property — a later pillar
reshapes the interpolant behind it, so their sequential solves drift. Their
final quote residuals are therefore measured, never asserted to vanish: use
``repricing_report`` on the finished curve, or pass ``tolerance`` to
``bootstrap`` to enforce the exactness contract mechanically. Notebook 02
builds the same quotes under all three rules to show the difference.

One stale quote bends the curve in its own neighbourhood and nothing smooths it
away. The parametric fits in ``yieldcurve.curves.parametric`` are the
counterpart, and Notebook 03 shows the same data under both.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

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
from yieldcurve.instruments import OIS, Bill, FixedCouponBond, Instrument, VanillaSwap

_DF_BRACKET = (1e-8, 5.0)
_SOLVER_TOLERANCE = 1e-15
_REPORT_TOLERANCE = 1e-10
_MAX_CONDITION = 1e12
_MATRIX_RESIDUAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Quote:
    """A market quote on an instrument.

    ``rate`` is read according to the instrument: a simple money-market rate for a
    ``Bill``, the par coupon for a ``FixedCouponBond`` (which therefore prices to
    100), and the par swap rate for a ``VanillaSwap`` or ``OIS``. The bootstrap
    supports exactly those four instrument types and rejects any other.
    """

    instrument: Instrument
    rate: float


@dataclass(frozen=True)
class QuoteRepricing:
    """One quote's final repricing result on a finished curve.

    ``model_rate`` is the quote the curve implies for the instrument — the
    implied simple rate for a ``Bill``, the par coupon for a ``FixedCouponBond``,
    and the par rate for a ``VanillaSwap`` or ``OIS``. ``residual`` is
    ``model_rate - target_rate`` and ``ok`` is the tolerance verdict.
    """

    instrument: Instrument
    target_rate: float
    model_rate: float
    residual: float
    tolerance: float
    ok: bool


def _maturity(instrument: Instrument) -> date:
    match instrument:
        case Bill() | FixedCouponBond() | VanillaSwap() | OIS():
            return instrument.maturity
        case _:
            raise CurveConstructionError(
                f"Cannot bootstrap from {type(instrument).__name__}; "
                "supported: Bill, FixedCouponBond, VanillaSwap, OIS"
            )


def _model_quote(instrument: Instrument, curves: CurveSet, asof: date) -> float:
    """The quote ``curves`` implies for ``instrument``, in the quote's own units."""
    ref = curves.discount.reference_date
    match instrument:
        case Bill():
            tau = year_fraction(asof, instrument.maturity, instrument.day_count)
            df = curves.discount.df(curve_time(ref, instrument.maturity))
            return (1.0 / df - 1.0) / tau
        case FixedCouponBond():
            if instrument.coupon == 0.0:
                raise CurveConstructionError(
                    f"Cannot derive a par-coupon model quote for a zero-coupon bond {instrument}"
                )
            df_terminal = curves.discount.df(curve_time(ref, instrument.maturity))
            tau_df_sum = 0.0
            for flow in instrument.cashflows(asof):
                # The maturity flow bundles the final coupon with the face.
                coupon_amount = (
                    flow.amount - instrument.face
                    if flow.date == instrument.maturity
                    else flow.amount
                )
                if coupon_amount == 0.0:
                    continue
                tau = coupon_amount / (instrument.face * instrument.coupon)
                tau_df_sum += tau * curves.discount.df(curve_time(ref, flow.date))
            if tau_df_sum == 0.0:
                raise CurveConstructionError(
                    f"{instrument} has no remaining coupon flows; par-coupon quote is undefined"
                )
            # par coupon c* solves face = face*df(T) + c*face*sum(tau_i df_i).
            return (1.0 - df_terminal) / tau_df_sum
        case VanillaSwap() | OIS():
            return par_rate(instrument, curves, asof)
        case _:
            raise CurveConstructionError(
                f"Cannot derive a model quote for {type(instrument).__name__}; "
                "supported: Bill, FixedCouponBond, VanillaSwap, OIS"
            )


def _curves_for_report(curve: DiscountCurve, discount_curve: DiscountCurve | None) -> CurveSet:
    """The CurveSet the report prices against: the reported curve as forecast,
    with ``discount_curve`` (or the curve itself) as the discounting curve."""
    if discount_curve is None:
        return CurveSet.single(curve)
    return CurveSet(discount=discount_curve, forecast=defaultdict(lambda: curve))


def repricing_report(
    curve: DiscountCurve,
    quotes: Sequence[Quote],
    asof: date,
    *,
    discount_curve: DiscountCurve | None = None,
    tolerance: float = _REPORT_TOLERANCE,
) -> tuple[QuoteRepricing, ...]:
    """Reprice every quote on the finished ``curve`` and verdict each against ``tolerance``.

    A canonical log-linear build passes ``tolerance`` for every quote; a
    comparative overlay (cubic log-DF or monotone convex) leaves measured,
    typically nonzero residuals wherever a payment falls between knots. The
    report is the single place the final residuals of any build are read.
    """
    curves = _curves_for_report(curve, discount_curve)
    rows = []
    for quote in quotes:
        model_rate = _model_quote(quote.instrument, curves, asof)
        rows.append(
            QuoteRepricing(
                instrument=quote.instrument,
                target_rate=quote.rate,
                model_rate=model_rate,
                residual=model_rate - quote.rate,
                tolerance=tolerance,
                ok=abs(model_rate - quote.rate) <= tolerance,
            )
        )
    return tuple(rows)


def bootstrap(
    quotes: Sequence[Quote],
    asof: date,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
    discount_curve: DiscountCurve | None = None,
    *,
    tolerance: float | None = None,
) -> InterpolatedDiscountCurve:
    """Build a curve from quoted instruments.

    With ``discount_curve`` omitted the result is a discount curve in its own
    right — the single-curve case, and the right call when bootstrapping OIS.
    Pass an already-built OIS curve to bootstrap a *forecast* curve instead: the
    solved curve then projects the floating leg while the quoted swap is
    discounted off OIS, which is how a dual-curve build is done post-2008.
    Discounting a swap off its own projection curve is the pre-crisis convention
    and misprices the basis.

    The canonical method (log-linear DF, the default) reprices every quote
    exactly; that is its exactness contract. With a comparative method the final
    residuals are measured with ``repricing_report`` and are expected to be
    nonzero. Passing ``tolerance`` enforces the contract mechanically: the build
    raises if any final quote residual exceeds it.
    """
    if not quotes:
        raise CurveConstructionError("Cannot bootstrap a curve from no quotes")
    for quote in quotes:
        _maturity(quote.instrument)  # rejects unsupported instrument types
        if not math.isfinite(quote.rate):
            raise CurveConstructionError(
                f"Non-finite quote rate {quote.rate!r} on {_maturity(quote.instrument)}"
            )

    maturities = [_maturity(quote.instrument) for quote in quotes]
    for earlier, later in pairwise(maturities):
        if later == earlier:
            raise CurveConstructionError(f"Two instruments share the same maturity: {earlier}")
        if later < earlier:
            raise CurveConstructionError(
                f"Quotes are not in increasing maturity order: {earlier} before {later}; "
                "pass them sorted by maturity"
            )
    if any(m <= asof for m in maturities):
        raise CurveConstructionError(
            f"Every instrument must mature after {asof}; the earliest matures on {maturities[0]}"
        )

    times: list[float] = []
    dfs: list[float] = []
    for quote in quotes:
        t = curve_time(asof, _maturity(quote.instrument))
        next_df = _solve_next_df(quote, asof, times, dfs, t, method, discount_curve)
        times.append(t)
        dfs.append(next_df)

    curve = InterpolatedDiscountCurve(
        reference_date=asof, times=tuple(times), dfs=tuple(dfs), method=method
    )
    if tolerance is not None:
        failures = [
            row
            for row in repricing_report(
                curve, quotes, asof, discount_curve=discount_curve, tolerance=tolerance
            )
            if not row.ok
        ]
        if failures:
            detail = "; ".join(
                f"{type(row.instrument).__name__} {row.instrument.maturity} "
                f"residual {row.residual:g}"
                for row in failures
            )
            raise CurveConstructionError(
                f"Curve does not reprice {len(failures)} of {len(quotes)} quotes within "
                f"tolerance {tolerance:g}: {detail}"
            )
    return curve


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

    The solve is only accepted when the system is well posed: the cash-flow
    matrix must be square and full rank with a bounded condition number, the
    price vector must have nonzero norm, and the normalized residual
    ``||CF d - P|| / ||P||`` of the solution must stay within tolerance.
    """
    rows, columns = cashflows.shape
    if rows != columns:
        raise CurveConstructionError(
            f"Cash-flow matrix must be square to determine one discount factor per "
            f"instrument, got {rows}x{columns}"
        )
    if prices.shape != (rows,):
        raise CurveConstructionError(f"Expected {rows} prices, got {prices.shape}")
    if not np.all(np.isfinite(cashflows)):
        raise CurveConstructionError("Cash-flow matrix contains non-finite entries")
    if not np.all(np.isfinite(prices)):
        raise CurveConstructionError("Price vector contains non-finite entries")

    rank = np.linalg.matrix_rank(cashflows)
    if rank < rows:
        raise CurveConstructionError(
            f"Cash-flow matrix is rank deficient ({rank} of {rows}); no unique discount "
            "factor vector exists"
        )
    condition = float(np.linalg.cond(cashflows))
    if not np.isfinite(condition) or condition > _MAX_CONDITION:
        raise CurveConstructionError(
            f"Cash-flow matrix is ill-conditioned (condition number {condition:g} "
            f"> {_MAX_CONDITION:g}); the solve would amplify input noise"
        )
    price_norm = float(np.linalg.norm(prices))
    if price_norm == 0.0:
        raise CurveConstructionError(
            "Price vector has zero norm; the normalized residual is undefined"
        )

    df = np.linalg.solve(cashflows, prices)
    if not np.all(np.isfinite(df)):
        raise CurveConstructionError("Matrix solve produced non-finite discount factors")
    residual = float(np.linalg.norm(cashflows @ df - prices) / price_norm)
    if not np.isfinite(residual) or residual > _MATRIX_RESIDUAL_TOLERANCE:
        raise CurveConstructionError(
            f"Matrix solve leaves a normalized residual of {residual:g} "
            f"(tolerance {_MATRIX_RESIDUAL_TOLERANCE:g})"
        )
    return df
