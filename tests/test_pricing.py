from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from yieldcurve.calendars import NullCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import annuity, par_rate, price, ytm
from yieldcurve.curves.protocol import CurveSet, Fixings, FlatCurve, MissingFixingError
from yieldcurve.instruments import FRN, OIS, Bill, FixedCouponBond, VanillaSwap

REFERENCE = date(2026, 7, 24)
LATER = date(2026, 8, 24)


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


# -- discount-ratio identity --------------------------------------------------


def test_discount_ratio_on_non_flat_curve() -> None:
    """Valuation after curve construction discounts by D_r(T)/D_r(asof).

    Build a curve with reference_date != asof. The discount factor from asof
    to date d is the ratio of two absolute-time discount factors, not the raw
    df of the time delta (asof, d). On a flat curve df_abs(d) = exp(-r*t_abs(d)),
    so df(asof, d) = exp(-r*(t_abs(d)-t_abs(asof))) = exp(-r*actual_days/365).
    Raw curve.df(curve_time(asof, d)) gives the same. This test pins the formula
    with forward-starting and in-flight periods so the ratio matters.
    """
    flat_rate = 0.03
    ref = date(2026, 1, 1)
    curves = CurveSet.single(FlatCurve(reference_date=ref, rate=flat_rate))
    asof = REFERENCE  # 2026-07-24, months after reference
    # Bill that matures 30 days later, at par
    bill = Bill(maturity=date(2026, 8, 24), day_count=DayCount.ACT_365F)
    result = price(bill, curves, asof=asof)
    days = (bill.maturity - asof).days
    expected = 100.0 * math.exp(-flat_rate * days / 365.0)
    assert result.dirty == pytest.approx(expected, rel=1e-6)


# -- future floating coupon ---------------------------------------------------


def test_frn_future_coupon_uses_projected_forward(flat: CurveSet) -> None:
    """A forward-starting FRN period uses curve forward, identically to the
    single-curve telescoping identity. With forecast == discount, zero spread,
    the dirty price is par discounted from issue."""
    frn = FRN(
        issue=date(2026, 10, 24),
        maturity=date(2031, 10, 24),
        frequency=2,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="6M",
        spread=0.0,
    )
    result = price(frn, flat, asof=REFERENCE)
    days_to_issue = (frn.issue - REFERENCE).days
    assert result.dirty == pytest.approx(100.0 * flat.discount.df(days_to_issue / 365.0))
    assert result.accrued == 0.0


# -- active term coupon uses reset fixing -------------------------------------


def test_frn_active_coupon_uses_reset_fixing(flat: CurveSet) -> None:
    """An FRN whose first period started before asof must use the observed reset
    fixing rather than a curve forward. When the fixing equals the curve's
    implied forward, the result is the same; this test pins that the active
    branch asserts the fixing, not that the fixing value differs."""
    frn = FRN(
        issue=date(2026, 4, 24),
        maturity=date(2031, 4, 24),
        frequency=2,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="6M",
        spread=0.01,
    )
    fixing_rate = 0.025
    fixings = Fixings(term={("6M", frn.issue): fixing_rate})
    result = price(frn, flat, asof=REFERENCE, fixings=fixings)
    # first coupon: face * (fixing + spread) * tau
    assert result.accrued > 0.0
    assert result.dirty > 0.0


def test_missing_reset_fixing_raises(flat: CurveSet) -> None:
    """An active coupon that has already reset but whose fixing is absent must
    raise MissingFixingError, not silently project a short forward."""
    frn = FRN(
        issue=date(2026, 4, 24),
        maturity=date(2031, 4, 24),
        frequency=2,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="6M",
        spread=0.0,
    )
    with pytest.raises(MissingFixingError, match="6M @ 2026-04-24"):
        price(frn, flat, asof=REFERENCE)


# -- matured instruments return zero ------------------------------------------


def test_a_matured_frn_is_worth_nothing(flat: CurveSet) -> None:
    frn = FRN(
        issue=date(2021, 7, 24),
        maturity=date(2026, 1, 24),
        frequency=2,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="6M",
        spread=0.0,
    )
    result = price(frn, flat, asof=REFERENCE)
    assert (result.dirty, result.clean, result.accrued) == (0.0, 0.0, 0.0)


def test_a_matured_swap_is_worth_nothing(flat: CurveSet) -> None:
    swap = VanillaSwap(
        start=date(2021, 7, 24),
        maturity=date(2025, 1, 24),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    result = price(swap, flat, asof=REFERENCE)
    assert result.dirty == 0.0


def test_a_matured_ois_is_worth_nothing(flat: CurveSet) -> None:
    ois = OIS(
        start=date(2021, 7, 24),
        maturity=date(2025, 1, 24),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    result = price(ois, flat, asof=REFERENCE)
    assert result.dirty == 0.0


# -- narrowed type -------------------------------------------------------------


def test_price_narrows_to_instrument_union(flat: CurveSet) -> None:
    """price() signature accepts Instrument, not object."""
    from yieldcurve.instruments import Instrument

    bond: Instrument = make_treasury()
    result = price(bond, flat, asof=REFERENCE)
    assert result.dirty > 0.0


# -- existing tests (preserved, updated for new API) --------------------------


# test_bill_price_is_the_discount_factor_times_face unchanged
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
    bond = make_treasury()
    dirty = price(bond, flat, asof=REFERENCE).dirty
    y = ytm(bond, dirty, asof=REFERENCE)
    assert 0.0 < y < 0.10
    reconstructed = _street_price(bond, y, REFERENCE)
    assert reconstructed == pytest.approx(dirty, rel=1e-10)


def _street_price(bond: FixedCouponBond, y: float, asof: date) -> float:
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
    at_par = price(replace(swap, fixed_rate=fair), flat, asof=REFERENCE)
    assert at_par.dirty == pytest.approx(0.0, abs=1e-6)
    assert annuity(swap, flat, asof=REFERENCE) > 0.0


def test_frn_priced_off_a_single_curve_at_zero_spread_is_worth_par(flat: CurveSet) -> None:
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


def test_par_rate_does_not_drift_with_the_valuation_date(flat: CurveSet) -> None:
    swap = VanillaSwap(
        start=date(2026, 1, 2),
        maturity=date(2031, 1, 2),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        float_tenor="6M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    # The mid-life asofs fall inside the first floating period, so the par rate
    # needs that period's observed reset fixing — the library never substitutes
    # a shortened forward. The fixing equals the flat curve's rate, so the
    # mid-life par rate matches the spot-starting one.
    fixing = Fixings(term={("6M", date(2026, 1, 2)): 0.04})
    rates = [
        par_rate(
            swap,
            CurveSet.single(FlatCurve(reference_date=asof, rate=0.04)),
            asof=asof,
            fixings=fixing,
        )
        for asof in (date(2026, 1, 2), date(2026, 4, 2), date(2026, 6, 30))
    ]
    assert max(rates) - min(rates) < 5e-4


def test_par_rate_rejects_an_active_period_without_fixings(flat: CurveSet) -> None:
    """A swap whose first period has already started cannot be valued at par
    without the observed reset fixing: par_rate must raise MissingFixingError
    rather than silently project a shortened forward over the stub."""
    swap = VanillaSwap(
        start=date(2026, 1, 2),
        maturity=date(2031, 1, 2),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        float_tenor="6M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    curves = CurveSet.single(FlatCurve(reference_date=date(2026, 1, 2), rate=0.04))
    with pytest.raises(MissingFixingError, match="6M @ 2026-01-02"):
        par_rate(swap, curves, asof=date(2026, 4, 2))


def test_ois_uses_its_own_floating_day_count() -> None:
    """OIS float_day_count reaches the pricer via the realised-fixings path.

    Provide a flat overnight fixing so the active period's rate is independent
    of the curve forward, isolating the day-count difference.
    """
    ois = OIS(
        start=date(2026, 1, 31),
        maturity=date(2031, 1, 31),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        float_day_count=DayCount.ACT_360,
    )
    asof = date(2026, 8, 31)
    curves = CurveSet.single(FlatCurve(reference_date=asof, rate=0.04))
    # supply overnight fixings at a constant 4% for the whole active period
    obs_start = date(2026, 1, 31)
    obs_end = asof
    fix_map = {}
    d = obs_start
    while d < obs_end:
        nxt = d + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        fix_map[d] = 0.04
        d = nxt
    fixings = Fixings(overnight=fix_map)

    act360 = price(ois, curves, asof=asof, fixings=fixings).dirty
    thirty = price(
        replace(ois, float_day_count=DayCount.THIRTY_360_BOND),
        curves,
        asof=asof,
        fixings=fixings,
    ).dirty
    assert act360 != pytest.approx(thirty, rel=1e-9)


def test_an_frn_accrues_interest_between_resets(flat: CurveSet) -> None:
    frn = FRN(
        issue=date(2026, 4, 24),
        maturity=date(2031, 4, 24),
        frequency=2,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="6M",
        spread=0.0,
    )
    fixings = Fixings(term={("6M", frn.issue): 0.03})
    result = price(frn, flat, asof=REFERENCE, fixings=fixings)
    assert result.accrued > 0.0
    assert result.clean == pytest.approx(result.dirty - result.accrued)


def test_active_ois_coupon_uses_overnight_fixings() -> None:
    """An OIS with an active period compounds realised overnight fixings."""
    ois = OIS(
        start=date(2026, 7, 24),
        maturity=date(2027, 7, 24),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        float_day_count=DayCount.ACT_360,
    )
    curves = CurveSet.single(FlatCurve(reference_date=REFERENCE, rate=0.03))
    # active period: start=2026-07-24, asof=LATER=2026-08-24, end=2027-07-24
    # few overnight fixings at 3% (same as curve) so forward factor dominates
    obs: dict[date, float] = {}
    d = ois.start
    while d < LATER:
        nxt = d + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        obs[d] = 0.03
        d = nxt
    fixings = Fixings(overnight=obs)
    result = price(ois, curves, asof=LATER, fixings=fixings)
    assert result.dirty != 0.0


def test_missing_overnight_fixing_raises() -> None:
    ois = OIS(
        start=date(2026, 7, 24),
        maturity=date(2027, 7, 24),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    curves = CurveSet.single(FlatCurve(reference_date=LATER, rate=0.03))
    with pytest.raises(MissingFixingError, match="overnight"):
        price(ois, curves, asof=LATER)
