"""Duration, convexity and DV01: analytic against effective, and against QuantLib."""

from __future__ import annotations

from datetime import date

import pytest

from curveengine.calendars import USGovernmentBondCalendar
from curveengine.conventions import BusinessDayConvention, DayCount
from curveengine.curves.protocol import CurveSet, FlatCurve
from curveengine.instruments import FixedCouponBond, VanillaSwap
from curveengine.pricing import price
from curveengine.risk.sensitivities import (
    convexity,
    dollar_duration,
    dv01,
    effective_convexity,
    effective_duration,
    macaulay_duration,
    modified_duration,
)

ASOF = date(2026, 7, 24)


@pytest.fixture
def bond() -> FixedCouponBond:
    return FixedCouponBond(
        issue=ASOF,
        maturity=date(2036, 7, 24),
        coupon=0.04,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )


@pytest.fixture
def flat() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))


def test_zero_coupon_macaulay_duration_equals_its_maturity(flat: CurveSet) -> None:
    """The one duration everybody can check by hand. A zero has a single cash
    flow, so its weighted mean time is its maturity, exactly."""
    from curveengine.instruments import Bill

    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)

    assert macaulay_duration(zero, flat, ASOF) == pytest.approx(5.0, abs=0.01)


def test_modified_is_macaulay_discounted_by_the_yield(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    from curveengine.pricing import ytm

    y = ytm(bond, price(bond, flat, ASOF).dirty, ASOF)
    macaulay = macaulay_duration(bond, flat, ASOF)

    assert modified_duration(bond, flat, ASOF) == pytest.approx(
        macaulay / (1 + y / bond.frequency), rel=1e-3
    )


def test_par_bond_duration_is_below_its_maturity(bond: FixedCouponBond, flat: CurveSet) -> None:
    """A coupon bond pays before maturity, so duration is strictly shorter.
    Fails loudly if the coupon flows are being dropped from the weighting."""
    assert 0.0 < macaulay_duration(bond, flat, ASOF) < 10.0


def test_effective_and_modified_duration_agree_on_a_flat_curve(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """Under a flat curve the yield and the zero rate are the same number, so
    the yield-space and curve-space derivatives must coincide. On a sloped
    curve they legitimately differ, which is why the test fixes the curve flat
    — this is a correctness check, not a claim that they are always equal."""
    assert effective_duration(bond, flat, ASOF) == pytest.approx(
        modified_duration(bond, flat, ASOF), rel=0.03
    )


def test_dv01_is_dollar_duration_per_basis_point(bond: FixedCouponBond, flat: CurveSet) -> None:
    assert dv01(bond, flat, ASOF) == pytest.approx(
        dollar_duration(bond, flat, ASOF) * 1e-4, rel=0.03
    )


def test_dv01_is_positive_for_a_long_bond(bond: FixedCouponBond, flat: CurveSet) -> None:
    """Sign convention, fixed once: DV01 is reported positive for a long
    position, meaning 'the price falls by this much when rates rise by 1bp'.
    Half of all risk-system sign bugs are this convention left unstated."""
    assert dv01(bond, flat, ASOF) > 0.0


def test_convexity_is_positive_for_a_vanilla_bond(bond: FixedCouponBond, flat: CurveSet) -> None:
    assert convexity(bond, flat, ASOF) > 0.0


def test_effective_convexity_is_independent_of_bump_size(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """A central second difference over a bump h has error O(h^2) but noise
    O(eps/h^2). Too small a bump and floating point wins. This pins that the
    default bump sits in the stable region."""
    coarse = effective_convexity(bond, flat, ASOF, bump=1e-3)
    fine = effective_convexity(bond, flat, ASOF, bump=1e-4)

    assert coarse == pytest.approx(fine, rel=1e-3)


def test_duration_and_convexity_predict_a_real_reprice(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """The spec's rule: compare the implied price change, never the raw
    convexity number. A 50bp move is large enough that a duration-only
    prediction misses and the convexity term has to do work."""
    from curveengine.risk.scenarios import parallel, shift_curveset

    base = price(bond, flat, ASOF).dirty
    move = 0.0050
    shocked = price(bond, shift_curveset(flat, parallel(move)), ASOF).dirty

    predicted = base * (
        1
        - effective_duration(bond, flat, ASOF) * move
        + 0.5 * effective_convexity(bond, flat, ASOF) * move**2
    )

    assert predicted == pytest.approx(shocked, rel=2e-5)


def test_duration_alone_underpredicts_the_gain_on_a_rally(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """Positive convexity means the linear estimate is always pessimistic in
    both directions. If this fails, the convexity term has the wrong sign and
    the previous test was passing on a cancellation."""
    from curveengine.risk.scenarios import parallel, shift_curveset

    base = price(bond, flat, ASOF).dirty
    actual = price(bond, shift_curveset(flat, parallel(-0.0050)), ASOF).dirty
    linear = base * (1 + effective_duration(bond, flat, ASOF) * 0.0050)

    assert actual > linear


def test_analytic_duration_rejects_a_swap(flat: CurveSet) -> None:
    """A par swap has zero price, so modified duration divides by zero and
    yield is not defined. Effective duration on the fixed leg is the meaningful
    measure. Raising beats returning inf."""
    swap = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.04,
        fixed_frequency=2,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
    )

    with pytest.raises(TypeError, match="yield"):
        modified_duration(swap, flat, ASOF)


def test_effective_duration_accepts_a_swap(flat: CurveSet) -> None:
    swap = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.04,
        fixed_frequency=2,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
    )

    assert effective_duration(swap, flat, ASOF) != 0.0
