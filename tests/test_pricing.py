"""Pricing: dirty, clean, accrued, yield, and swap par rates."""

from __future__ import annotations

import math
from datetime import date

import pytest

from curveengine.calendars import NullCalendar, USGovernmentBondCalendar
from curveengine.conventions import BusinessDayConvention, DayCount
from curveengine.curves.protocol import CurveSet, FlatCurve
from curveengine.instruments import FRN, Bill, FixedCouponBond, VanillaSwap
from curveengine.pricing import annuity, par_rate, price, ytm

REFERENCE = date(2026, 7, 24)


@pytest.fixture
def flat() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=REFERENCE, rate=0.03))


def make_treasury() -> FixedCouponBond:
    return FixedCouponBond(
        issue=date(2024, 2, 15),
        maturity=date(2034, 2, 15),
        coupon=0.0425,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )


def test_bill_price_is_the_discount_factor_times_face(flat: CurveSet) -> None:
    bill = Bill(maturity=date(2027, 7, 24))

    result = price(bill, flat, asof=REFERENCE)

    assert result.dirty == pytest.approx(100.0 * math.exp(-0.03 * 365 / 365))
    assert result.accrued == 0.0
    assert result.clean == result.dirty


def test_clean_price_is_dirty_less_accrued(flat: CurveSet) -> None:
    bond = make_treasury()

    result = price(bond, flat, asof=REFERENCE)

    assert result.clean == pytest.approx(result.dirty - result.accrued)
    assert result.accrued == pytest.approx(bond.accrued(REFERENCE))
    assert result.accrued > 0.0


def test_a_bond_priced_at_its_own_coupon_rate_is_worth_about_par() -> None:
    """A par bond discounted at its coupon rate prices near 100. 'Near' rather
    than 'exactly' because the curve compounds continuously while the coupon
    compounds semiannually — the gap is the compounding convention, not an error."""
    bond = FixedCouponBond(
        issue=date(2026, 7, 24),
        maturity=date(2036, 7, 24),
        coupon=0.03,
        frequency=2,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    curves = CurveSet.single(FlatCurve(reference_date=REFERENCE, rate=0.03))

    assert price(bond, curves, asof=REFERENCE).dirty == pytest.approx(100.0, abs=1.0)


def test_ytm_inverts_the_price(flat: CurveSet) -> None:
    """The round trip that matters: yield the price back out of the price."""
    bond = make_treasury()
    dirty = price(bond, flat, asof=REFERENCE).dirty

    y = ytm(bond, dirty, asof=REFERENCE)

    assert 0.0 < y < 0.10
    reconstructed = _street_price(bond, y, REFERENCE)
    assert reconstructed == pytest.approx(dirty, rel=1e-10)


def _street_price(bond: FixedCouponBond, y: float, asof: date) -> float:
    """Independent restatement of the street-convention formula, so the round
    trip is checked against arithmetic written twice rather than once."""
    period_start, period_end = bond.accrual_period(asof)
    w = (period_end - asof).days / (period_end - period_start).days
    flows = bond.cashflows(asof)
    f = bond.frequency
    return float(sum(flow.amount / (1.0 + y / f) ** (w + k) for k, flow in enumerate(flows)))


def test_ytm_rises_when_the_price_falls(flat: CurveSet) -> None:
    bond = make_treasury()
    dirty = price(bond, flat, asof=REFERENCE).dirty

    assert ytm(bond, dirty - 5.0, asof=REFERENCE) > ytm(bond, dirty, asof=REFERENCE)


def test_par_rate_makes_a_swap_worth_nothing(flat: CurveSet) -> None:
    swap = VanillaSwap(
        start=date(2026, 7, 24),
        maturity=date(2031, 7, 24),
        fixed_rate=0.0,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )

    fair = par_rate(swap, flat, asof=REFERENCE)
    at_par = price(VanillaSwap(**{**swap.__dict__, "fixed_rate": fair}), flat, asof=REFERENCE)

    assert at_par.dirty == pytest.approx(0.0, abs=1e-6)
    assert annuity(swap, flat, asof=REFERENCE) > 0.0


def test_frn_priced_off_a_single_curve_at_zero_spread_is_worth_par(flat: CurveSet) -> None:
    """The textbook result: when the forecast and discount curves coincide and
    the spread is zero, an FRN is worth par at each reset. Phase 3 breaks this
    by separating the two curves, and the difference is the point of that phase."""
    frn = FRN(
        issue=REFERENCE,
        maturity=date(2031, 7, 24),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="3M",
        spread=0.0,
    )

    assert price(frn, flat, asof=REFERENCE).dirty == pytest.approx(100.0, abs=0.05)


def test_pricing_an_unsupported_type_names_the_type(flat: CurveSet) -> None:
    with pytest.raises(TypeError, match="str"):
        price("not an instrument", flat, asof=REFERENCE)
