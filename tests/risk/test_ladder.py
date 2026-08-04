"""Par-rate delta ladder against DV01 and against a known-zero position."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from yieldcurve.calendars import NullCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.bootstrap import Quote, bootstrap
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import OIS, Bill, FixedCouponBond
from yieldcurve.risk.ladder import bump_quote, par_delta_ladder
from yieldcurve.risk.sensitivities import dv01

ASOF = date(2026, 7, 24)


def _ois(years: float, rate: float) -> Quote:
    swap = OIS(
        start=ASOF,
        maturity=ASOF + timedelta(days=round(years * 365)),
        fixed_rate=rate,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    return Quote(instrument=swap, rate=rate)


QUOTES = (_ois(1, 0.042), _ois(2, 0.043), _ois(5, 0.044), _ois(10, 0.046))


def _bond(years: float) -> FixedCouponBond:
    return FixedCouponBond(
        issue=ASOF,
        maturity=ASOF + timedelta(days=round(years * 365)),
        coupon=0.045,
        frequency=2,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )


def test_ladder_has_one_entry_per_quote_keyed_by_maturity() -> None:
    ladder = par_delta_ladder(_bond(5), QUOTES, ASOF)

    assert list(ladder) == [q.instrument.maturity for q in QUOTES]  # type: ignore[attr-defined]


def _bump_everything(bond: FixedCouponBond, method: InterpMethod) -> float:
    """Price change from raising every quote 1bp at once, the ladder's total by definition."""
    base = bootstrap(QUOTES, asof=ASOF, method=method)
    shocked = bootstrap([bump_quote(q, 1e-4) for q in QUOTES], asof=ASOF, method=method)
    return (
        price(bond, CurveSet.single(base), ASOF).dirty
        - price(bond, CurveSet.single(shocked), ASOF).dirty
    )


@pytest.mark.parametrize("method", [InterpMethod.LOG_LINEAR_DF, InterpMethod.CUBIC_LOG_DF])
def test_ladder_is_additive_on_a_smooth_interpolator(method: InterpMethod) -> None:
    """One-at-a-time bumps sum to the simultaneous bump when the curve is smooth in the quotes."""
    bond = _bond(5)

    total = sum(par_delta_ladder(bond, QUOTES, ASOF, method=method).values())

    assert total == pytest.approx(_bump_everything(bond, method), rel=1e-3)


def test_monotone_convex_breaks_additivity() -> None:
    """Hagan-West is only C0 in its inputs, so its ladder does not sum to the joint bump.

    The amendment tests are branches on which region a forward falls into. A 1bp
    quote bump can flip a region and reshape the curve between knots, which the
    smooth interpolators above never do. Pinned rather than fixed: it is a
    property of the interpolation, and the reason ``ladder`` documents passing a
    smooth method when the numbers are going to be hedged on.
    """
    bond = _bond(5)

    total = sum(par_delta_ladder(bond, QUOTES, ASOF, method=InterpMethod.MONOTONE_CONVEX).values())

    assert total / _bump_everything(bond, InterpMethod.MONOTONE_CONVEX) == pytest.approx(
        1.014, abs=5e-3
    )


def test_ladder_is_close_to_dv01_without_matching_it() -> None:
    """A uniform par bump is not a uniform zero bump — same order, not the same number.

    The gap is the par-to-zero Jacobian plus the ACT/360 quote versus ACT/365F
    curve basis. It runs a few percent here and is maturity-dependent, so a
    tighter tolerance would be asserting a coincidence.
    """
    bond = _bond(5)
    curves = CurveSet.single(bootstrap(QUOTES, asof=ASOF))

    total = sum(par_delta_ladder(bond, QUOTES, ASOF).values())

    assert total == pytest.approx(dv01(bond, curves, ASOF), rel=5e-2)


def test_risk_sits_on_the_quotes_that_span_the_maturity() -> None:
    """A 2y bond carries no sensitivity to the 10y quote a bootstrap never uses."""
    ladder = par_delta_ladder(_bond(2), QUOTES, ASOF)
    ten_year = QUOTES[-1].instrument.maturity  # type: ignore[attr-defined]

    assert abs(ladder[ten_year]) < 1e-9


def test_bumping_a_bill_moves_only_the_quote() -> None:
    """A Bill carries no fixed rate of its own, so only Quote.rate can move."""
    bill = Bill(maturity=ASOF + timedelta(days=90), day_count=DayCount.ACT_360)

    bumped = bump_quote(Quote(instrument=bill, rate=0.04), 1e-4)

    assert bumped.rate == pytest.approx(0.0401)
    assert bumped.instrument is bill


def test_empty_quote_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one quote"):
        par_delta_ladder(_bond(5), (), ASOF)


def test_ladder_rejects_duplicate_maturities() -> None:
    """Two quotes on the same maturity would silently overwrite one ladder
    entry with the other; the quote grid must be distinct."""
    same_maturity = (_ois(2, 0.043), _ois(2, 0.044))

    with pytest.raises(ValueError, match="distinct"):
        par_delta_ladder(_bond(5), same_maturity, ASOF)


@pytest.mark.parametrize("bump", [0.0, -1e-4, float("nan"), float("inf")])
def test_ladder_rejects_an_invalid_bump(bump: float) -> None:
    with pytest.raises(ValueError, match="bump"):
        par_delta_ladder(_bond(5), QUOTES, ASOF, bump=bump)
