"""Instruments generate cash flows and never price."""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from yieldcurve import instruments
from yieldcurve.calendars import NullCalendar, SwedenCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.instruments import FRN, OIS, Bill, FixedCouponBond, VanillaSwap, tenor_to_frequency


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


def test_bill_pays_face_once_at_maturity() -> None:
    bill = Bill(maturity=date(2026, 10, 24), day_count=DayCount.ACT_360)

    flows = bill.cashflows(asof=date(2026, 7, 24))

    assert flows == (instruments.CashFlow(date(2026, 10, 24), 100.0),)


def test_bond_final_flow_carries_coupon_plus_principal() -> None:
    flows = make_treasury().cashflows(asof=date(2026, 7, 24))

    assert flows[-1].date == date(2034, 2, 15)
    assert flows[-1].amount == pytest.approx(100.0 + 100.0 * 0.0425 / 2)
    assert flows[0].amount == pytest.approx(100.0 * 0.0425 / 2)


def test_bond_drops_flows_on_or_before_the_valuation_date() -> None:
    """A coupon paid today belongs to the seller, not the buyer."""
    flows = make_treasury().cashflows(asof=date(2026, 8, 15))

    assert all(flow.date > date(2026, 8, 15) for flow in flows)
    assert flows[0].date == date(2027, 2, 15)


def test_accrual_period_brackets_the_valuation_date() -> None:
    previous, following = make_treasury().accrual_period(asof=date(2026, 7, 24))

    assert previous == date(2026, 2, 15)
    assert following == date(2026, 8, 15)


def test_accrued_interest_is_zero_on_a_coupon_date_and_grows_within_a_period() -> None:
    bond = make_treasury()

    assert bond.accrued(asof=date(2026, 2, 15)) == pytest.approx(0.0)
    mid = bond.accrued(asof=date(2026, 5, 15))
    assert 0.0 < mid < 100.0 * 0.0425 / 2


def test_accrued_matches_the_hand_computed_icma_fraction() -> None:
    bond = make_treasury()
    asof = date(2026, 5, 15)
    days_accrued = (asof - date(2026, 2, 15)).days
    days_in_period = (date(2026, 8, 15) - date(2026, 2, 15)).days

    expected = 100.0 * 0.0425 / 2 * days_accrued / days_in_period

    assert bond.accrued(asof) == pytest.approx(expected)


def test_short_stub_coupon_and_accrual_use_regular_icma_references() -> None:
    bond = FixedCouponBond(
        issue=date(2026, 4, 1),
        maturity=date(2026, 8, 31),
        coupon=0.06,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    (period,) = bond.coupon_periods()
    reference_days = (date(2026, 8, 31) - date(2026, 2, 28)).days
    coupon_fraction = (period.accrual_end - period.accrual_start).days / reference_days / 2
    (flow,) = bond.cashflows(asof=date(2026, 3, 31))

    assert (period.reference_start, period.reference_end) == (
        date(2026, 2, 28),
        date(2026, 8, 31),
    )
    assert flow.amount == pytest.approx(bond.face + bond.face * bond.coupon * coupon_fraction)
    accrued_fraction = (date(2026, 6, 1) - bond.issue).days / reference_days / 2
    assert bond.accrued(date(2026, 6, 1)) == pytest.approx(
        bond.face * bond.coupon * accrued_fraction
    )


def test_thirty_360_accrued_uses_the_thirty_day_month() -> None:
    """A SEK government bond accrues on 30/360, so three months is exactly a
    quarter of a year regardless of the actual day count."""
    sgb = FixedCouponBond(
        issue=date(2024, 6, 12),
        maturity=date(2032, 6, 12),
        coupon=0.025,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )

    assert sgb.accrued(asof=date(2026, 9, 12)) == pytest.approx(100.0 * 0.025 * 0.25)


def test_frn_cannot_generate_its_own_cashflows() -> None:
    frn = FRN(
        issue=date(2026, 3, 16),
        maturity=date(2029, 3, 16),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        index_tenor="3M",
        spread=0.0025,
    )

    with pytest.raises(NotImplementedError, match="forecast curve"):
        frn.cashflows(asof=date(2026, 7, 24))


def test_frn_coupon_dates_uses_same_schedule_shape_as_bond() -> None:
    frn = FRN(
        issue=date(2026, 3, 16),
        maturity=date(2029, 3, 16),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        index_tenor="3M",
        spread=0.0025,
    )
    periods = frn.coupon_periods()
    dates = frn.coupon_dates()
    assert dates == (
        periods[0].accrual_start,
        *(period.payment_date for period in periods),
    )
    assert dates[0] == date(2026, 3, 16)
    assert dates[-1] == date(2029, 3, 16)


def test_bill_returns_empty_flows_when_matured() -> None:
    bill = Bill(maturity=date(2026, 1, 1))
    assert bill.cashflows(asof=date(2026, 7, 24)) == ()


def test_accrual_period_raises_when_asof_is_outside_bond_life() -> None:
    bond = make_treasury()
    with pytest.raises(ValueError, match="outside"):
        bond.accrual_period(asof=date(2020, 1, 1))


def test_tenor_to_frequency_rejects_overnight_tenor() -> None:
    with pytest.raises(ValueError, match="Overnight"):
        tenor_to_frequency("ON")


def test_tenor_to_frequency_rejects_unsupported_tenor() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        tenor_to_frequency("9M")


def test_swap_schedules_have_the_expected_period_counts() -> None:
    swap = VanillaSwap(
        start=date(2026, 7, 28),
        maturity=date(2031, 7, 28),
        fixed_rate=0.031,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
    )

    assert len(swap.fixed_periods()) == 5
    assert len(swap.float_periods()) == 20
    assert swap.fixed_schedule() == (
        swap.fixed_periods()[0].accrual_start,
        *(period.payment_date for period in swap.fixed_periods()),
    )
    assert swap.float_schedule() == (
        swap.float_periods()[0].accrual_start,
        *(period.payment_date for period in swap.float_periods()),
    )


def test_swap_cashflows_accrue_unadjusted_and_pay_adjusted() -> None:
    swap = VanillaSwap(
        start=date(2026, 1, 1),
        maturity=date(2027, 1, 1),
        fixed_rate=0.04,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_365F,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=SwedenCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )

    (period,) = swap.fixed_periods()
    (flow,) = swap.fixed_cashflows(asof=date(2025, 12, 31))
    assert (period.accrual_start, period.accrual_end) == (
        date(2026, 1, 1),
        date(2027, 1, 1),
    )
    assert flow.date == date(2027, 1, 4)
    assert flow.amount == pytest.approx(swap.notional * swap.fixed_rate)


def test_ois_matches_vanilla_swap_shape_with_float_day_count() -> None:
    """OIS should expose float_day_count, same as VanillaSwap."""
    ois = OIS(
        start=date(2026, 7, 28),
        maturity=date(2031, 7, 28),
        fixed_rate=0.031,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
    )
    assert ois.float_day_count == DayCount.ACT_360
    assert len(ois.fixed_schedule()) == 6
    assert len(ois.float_schedule()) == 6
    assert len(ois.fixed_periods()) == 5
    assert ois.float_periods() == ois.fixed_periods()


def test_no_instrument_method_accepts_a_curve() -> None:
    """The boundary rule as an executable assertion: if a curve ever appears in
    an instrument signature, discounting has leaked out of pricing.py."""
    forbidden = {"curve", "curves", "curveset", "discount_curve"}
    for _, cls in inspect.getmembers(instruments, inspect.isclass):
        if cls.__module__ != instruments.__name__:
            continue
        for _, method in inspect.getmembers(cls, inspect.isfunction):
            params = set(inspect.signature(method).parameters)
            assert not params & forbidden, f"{cls.__name__}.{method.__name__} takes a curve"
