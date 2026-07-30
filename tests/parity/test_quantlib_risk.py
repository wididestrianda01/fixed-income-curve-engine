"""QuantLib parity for risk measures. QuantLib is a test-only dependency."""

from __future__ import annotations

from datetime import date

import pytest
import QuantLib as ql  # noqa: N813

from yieldcurve.calendars import USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.risk.sensitivities import (
    convexity,
    modified_duration,
)

ASOF = date(2026, 7, 24)
MATURITY = date(2036, 7, 24)
COUPON = 0.04
FACE = 100.0
FLAT_RATE = 0.04


def _our_bond() -> FixedCouponBond:
    return FixedCouponBond(
        issue=ASOF,
        maturity=MATURITY,
        coupon=COUPON,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )


def _our_curves() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=ASOF, rate=FLAT_RATE))


def _ql_bond() -> tuple[ql.FixedRateBond, ql.DayCounter]:
    """The same bond in QuantLib. Settlement is zero days so that the QuantLib
    settlement date is ASOF, matching our pricing date exactly. Returns the
    bond and its day counter, because every BondFunctions call needs both."""
    asof = ql.Date(24, 7, 2026)
    ql.Settings.instance().evaluationDate = asof

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(
        asof,
        ql.Date(24, 7, 2036),
        ql.Period(ql.Semiannual),
        calendar,
        ql.Following,
        ql.Following,
        ql.DateGeneration.Backward,
        False,
    )
    day_count = ql.ActualActual(ql.ActualActual.Bond, schedule)
    bond = ql.FixedRateBond(0, FACE, schedule, [COUPON], day_count)

    return bond, day_count


def test_modified_duration_matches_quantlib() -> None:
    """Duration has one convention, so the number itself is comparable.

    Our clean price is fed into QuantLib's yield solver rather than rebuilding
    our flat curve in QuantLib. That deliberately takes the curve's compounding
    convention out of the comparison: both libraries then differentiate the
    same yield, so a failure here is a duration bug and nothing else.

    Settlement is the issue date, so accrued interest is zero and the dirty
    price the pricer returns is also the clean price.
    """
    ql_bond, day_count = _ql_bond()
    clean = price(_our_bond(), _our_curves(), ASOF).dirty

    y = ql_bond.bondYield(
        ql.BondPrice(clean, ql.BondPrice.Clean),
        day_count,
        ql.Compounded,
        ql.Semiannual,
    )
    rate = ql.InterestRate(y, day_count, ql.Compounded, ql.Semiannual)
    expected = ql.BondFunctions.duration(ql_bond, rate, ql.Duration.Modified)

    assert modified_duration(_our_bond(), _our_curves(), ASOF) == pytest.approx(expected, rel=1e-4)


def test_convexity_matches_quantlib_through_the_price_change() -> None:
    """Convexity does not have one convention. Spec section 4.3: compare the
    resulting price change under a 50bp move, never the raw convexity.

    Build the same bond in both libraries, shock both by 50bp, and assert the
    two predicted price changes agree to 1e-6 of face. If instead you assert
    ``our_convexity == ql.BondFunctions.convexity(...)`` you will chase a
    factor that is a compounding convention and lose an afternoon.
    """
    move = 0.0050

    bond, curves = _our_bond(), _our_curves()
    base = price(bond, curves, ASOF).dirty
    ours = base * (
        -modified_duration(bond, curves, ASOF) * move
        + 0.5 * convexity(bond, curves, ASOF) * move**2
    )

    ql_bond, day_count = _ql_bond()
    y = ql_bond.bondYield(
        ql.BondPrice(base, ql.BondPrice.Clean),
        day_count,
        ql.Compounded,
        ql.Semiannual,
    )
    rate = ql.InterestRate(y, day_count, ql.Compounded, ql.Semiannual)
    theirs = base * (
        -ql.BondFunctions.duration(ql_bond, rate, ql.Duration.Modified) * move
        + 0.5 * ql.BondFunctions.convexity(ql_bond, rate) * move**2
    )

    assert ours == pytest.approx(theirs, abs=1e-6 * FACE)
