"""Parity against QuantLib on dirty price, clean price, accrued and yield.

Two implementations of the same closed form agreeing rules out implementation
bugs. It does not validate the conventions themselves against the market.
"""

from __future__ import annotations

from datetime import date

import pytest
import QuantLib as ql  # noqa: N813

from yieldcurve.calendars import SwedenCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import price, ytm
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import FixedCouponBond

REFERENCE = date(2026, 7, 24)
FLAT_RATE = 0.03

PRICE_TOLERANCE = 1e-8  # per 100 face
ACCRUED_TOLERANCE = 1e-10
YIELD_TOLERANCE = 1e-8  # in absolute rate terms, i.e. well inside a hundredth of a bp


def _ql_date(d: date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def _ql_bond(
    bond: FixedCouponBond, ql_calendar: ql.Calendar, ql_day_count: ql.DayCounter
) -> ql.FixedRateBond:
    ql.Settings.instance().evaluationDate = _ql_date(REFERENCE)
    ql_schedule = ql.Schedule(
        _ql_date(bond.issue),
        _ql_date(bond.maturity),
        ql.Period(12 // bond.frequency, ql.Months),
        ql_calendar,
        ql.Unadjusted,
        ql.Unadjusted,
        ql.DateGeneration.Backward,
        False,
    )
    ql_bond = ql.FixedRateBond(
        0, bond.face, ql_schedule, [bond.coupon], ql_day_count, ql.Unadjusted
    )
    curve = ql.FlatForward(
        _ql_date(REFERENCE), FLAT_RATE, ql.Actual365Fixed(), ql.Continuous, ql.Annual
    )
    ql_bond.setPricingEngine(ql.DiscountingBondEngine(ql.YieldTermStructureHandle(curve)))
    return ql_bond


TREASURY = FixedCouponBond(
    issue=date(2024, 2, 15),
    maturity=date(2034, 2, 15),
    coupon=0.0425,
    frequency=2,
    day_count=DayCount.ACT_ACT_ICMA,
    calendar=USGovernmentBondCalendar(),
    bdc=BusinessDayConvention.UNADJUSTED,
)

SGB = FixedCouponBond(
    issue=date(2024, 6, 12),
    maturity=date(2032, 6, 12),
    coupon=0.025,
    frequency=1,
    day_count=DayCount.THIRTY_360_BOND,
    calendar=SwedenCalendar(),
    bdc=BusinessDayConvention.UNADJUSTED,
)

CASES = [
    pytest.param(
        TREASURY,
        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        ql.ActualActual(ql.ActualActual.ISMA),
        id="us-treasury-act/act-semiannual",
    ),
    pytest.param(
        SGB,
        ql.Sweden(),
        ql.Thirty360(ql.Thirty360.BondBasis),
        id="swedish-government-bond-30/360-annual",
    ),
]


@pytest.mark.parametrize(("bond", "ql_calendar", "ql_day_count"), CASES)
def test_accrued_matches_quantlib(
    bond: FixedCouponBond, ql_calendar: ql.Calendar, ql_day_count: ql.DayCounter
) -> None:
    ql_bond = _ql_bond(bond, ql_calendar, ql_day_count)

    assert bond.accrued(REFERENCE) == pytest.approx(ql_bond.accruedAmount(), abs=ACCRUED_TOLERANCE)


@pytest.mark.parametrize(("bond", "ql_calendar", "ql_day_count"), CASES)
def test_dirty_and_clean_price_match_quantlib(
    bond: FixedCouponBond, ql_calendar: ql.Calendar, ql_day_count: ql.DayCounter
) -> None:
    ql_bond = _ql_bond(bond, ql_calendar, ql_day_count)
    curves = CurveSet.single(FlatCurve(reference_date=REFERENCE, rate=FLAT_RATE))

    result = price(bond, curves, asof=REFERENCE)

    assert result.dirty == pytest.approx(ql_bond.dirtyPrice(), abs=PRICE_TOLERANCE)
    assert result.clean == pytest.approx(ql_bond.cleanPrice(), abs=PRICE_TOLERANCE)


@pytest.mark.parametrize(("bond", "ql_calendar", "ql_day_count"), CASES)
def test_yield_matches_quantlib(
    bond: FixedCouponBond, ql_calendar: ql.Calendar, ql_day_count: ql.DayCounter
) -> None:
    ql_bond = _ql_bond(bond, ql_calendar, ql_day_count)
    clean = ql_bond.cleanPrice()
    ql_yield = ql_bond.bondYield(
        ql.BondPrice(clean, ql.BondPrice.Clean),
        ql_day_count,
        ql.Compounded,
        bond.frequency,
        _ql_date(REFERENCE),
    )

    ours = ytm(bond, clean + bond.accrued(REFERENCE), asof=REFERENCE)

    assert ours == pytest.approx(ql_yield, abs=YIELD_TOLERANCE)
