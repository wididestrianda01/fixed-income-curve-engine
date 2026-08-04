"""Duration, convexity and DV01: analytic against effective, and against QuantLib."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import pytest

from yieldcurve.calendars import NullCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.pricing import par_rate, price, ytm
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import Bill, FixedCouponBond, VanillaSwap
from yieldcurve.risk.scenarios import parallel, shift_curveset
from yieldcurve.risk.sensitivities import (
    convexity,
    dollar_duration,
    dv01,
    effective_convexity,
    effective_duration,
    fisher_weil_duration,
    instrument_scale,
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


def _sloped() -> CurveSet:
    """Zero rates 2%, 4%, 6% at 1y, 5y, 10y — steep enough that the two
    duration conventions disagree visibly rather than at the noise floor."""
    curve = InterpolatedDiscountCurve(
        reference_date=ASOF,
        times=(1.0, 5.0, 10.0),
        dfs=(math.exp(-0.02), math.exp(-0.04 * 5.0), math.exp(-0.06 * 10.0)),
        method=InterpMethod.LOG_LINEAR_DF,
    )
    return CurveSet.single(curve)


def _par_swap(curves: CurveSet) -> VanillaSwap:
    """A swap struck at the curve's own par rate: opens at zero present value."""
    base = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.0,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    return replace(base, fixed_rate=par_rate(base, curves, ASOF))


def test_zero_coupon_duration_is_its_maturity_under_both_conventions(
    flat: CurveSet,
) -> None:
    """A zero has a single cash flow, so every weighting scheme gives the same
    mean time: the maturity. Pins both named conventions at once by hand."""
    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)

    assert fisher_weil_duration(zero, flat, ASOF) == pytest.approx(5.0, abs=0.01)
    assert macaulay_duration(zero, flat, ASOF) == pytest.approx(5.0, abs=0.01)


def test_modified_is_macaulay_discounted_by_the_yield(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """Modified duration is defined as YTM-weighted Macaulay divided by
    (1 + y/frequency); both sides use the same yield, so the identity is exact
    rather than approximate."""
    y = ytm(bond, price(bond, flat, ASOF).dirty, ASOF)

    assert modified_duration(bond, flat, ASOF) == pytest.approx(
        macaulay_duration(bond, flat, ASOF) / (1 + y / bond.frequency), rel=1e-9
    )


def test_fisher_weil_duration_sits_close_to_macaulay_on_a_flat_curve(
    bond: FixedCouponBond, flat: CurveSet
) -> None:
    """On a flat curve the spot rate equals the yield, so the spot-curve and
    YTM-weighted means nearly agree (they differ only by the continuous-versus-
    periodic compounding basis, a fraction of a percent)."""
    assert fisher_weil_duration(bond, flat, ASOF) == pytest.approx(
        macaulay_duration(bond, flat, ASOF), rel=2e-2
    )


def test_fisher_weil_and_macaulay_differ_on_a_sloped_curve(
    bond: FixedCouponBond,
) -> None:
    """The two conventions answer different questions: on a steeply sloped
    curve they disagree by far more than numerical noise. A regression that
    aliases the two names to one formula fails this loudly."""
    sloped = _sloped()
    gap = fisher_weil_duration(bond, sloped, ASOF) - macaulay_duration(bond, sloped, ASOF)
    assert abs(gap) > 1e-3


def test_par_bond_duration_is_below_its_maturity(bond: FixedCouponBond, flat: CurveSet) -> None:
    """A coupon bond pays before maturity, so duration is strictly shorter.
    Fails loudly if the coupon flows are being dropped from the weighting."""
    assert 0.0 < macaulay_duration(bond, flat, ASOF) < 10.0
    assert 0.0 < fisher_weil_duration(bond, flat, ASOF) < 10.0


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


def test_dv01_is_positive_loss_per_basis_point(bond: FixedCouponBond, flat: CurveSet) -> None:
    """Package convention (spec section 4): DV01 is the positive loss a long
    position takes when rates rise 1bp — ``base - price(+1bp)`` — not a signed
    price change. Half of all risk-system sign bugs are this convention left
    unstated."""
    base = price(bond, flat, ASOF).dirty
    up = price(bond, shift_curveset(flat, parallel(1e-4)), ASOF).dirty

    assert dv01(bond, flat, ASOF) == pytest.approx(base - up, rel=1e-12)
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


def test_modified_duration_rejects_a_bill(flat: CurveSet) -> None:
    """ytm only accepts FixedCouponBond, so a Bill must be rejected before calling it."""
    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)

    with pytest.raises(TypeError, match="analytic duration"):
        modified_duration(zero, flat, ASOF)


def test_convexity_rejects_a_bill(flat: CurveSet) -> None:
    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)

    with pytest.raises(TypeError, match="analytic duration"):
        convexity(zero, flat, ASOF)


def test_effective_duration_rejects_a_materially_zero_base_pv(flat: CurveSet) -> None:
    """Error policy: normalized risk must not divide by a materially zero
    present value. A par swap prices at zero, so effective duration must raise
    instead of returning inf/NaN."""
    swap = _par_swap(flat)

    with pytest.raises(ValueError, match="materially zero"):
        effective_duration(swap, flat, ASOF)


def test_monetary_sensitivities_stay_available_for_a_par_swap(flat: CurveSet) -> None:
    """DV01 does not divide by the base price, so it stays meaningful exactly
    where normalized duration is not. The par swap prices at zero; the fixed
    receiver loses when rates rise (DV01 positive), the fixed payer gains
    (DV01 negative) — both finite and equal to the direct reprice."""
    payer = _par_swap(flat)
    receiver = replace(payer, pay_fixed=False)

    base = price(receiver, flat, ASOF).dirty
    up = price(receiver, shift_curveset(flat, parallel(1e-4)), ASOF).dirty
    assert base == pytest.approx(0.0, abs=1e-9)
    assert dv01(receiver, flat, ASOF) == pytest.approx(base - up, rel=1e-12)
    assert dv01(receiver, flat, ASOF) > 0.0
    assert math.isfinite(dv01(payer, flat, ASOF))
    assert dv01(payer, flat, ASOF) < 0.0


def test_effective_duration_still_works_for_an_off_par_swap(flat: CurveSet) -> None:
    """The near-zero guard rejects only the degenerate par swap; a swap with
    real present value keeps a well-defined effective duration."""
    swap = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )

    assert math.isfinite(effective_duration(swap, flat, ASOF))
    assert effective_duration(swap, flat, ASOF) != 0.0


def test_instrument_scale_pins_face_and_notional(bond: FixedCouponBond, flat: CurveSet) -> None:
    """One scale contract per instrument family: bonds and bills quote per
    face, swaps per notional. The portfolio converts position notionals with
    this, so pinning it here cross-checks the portfolio scaling tests."""
    assert instrument_scale(Bill(maturity=date(2031, 7, 24))) == 100.0
    assert instrument_scale(bond) == 100.0
    assert instrument_scale(_par_swap(flat)) == 1_000_000.0
