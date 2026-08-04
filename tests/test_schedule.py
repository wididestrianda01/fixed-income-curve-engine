"""Business-day adjustment and coupon schedule generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from yieldcurve import conventions
from yieldcurve.calendars import NullCalendar, SwedenCalendar
from yieldcurve.conventions import BusinessDayConvention, add_months, adjust, schedule


def test_following_rolls_a_saturday_forward_to_monday() -> None:
    assert adjust(date(2026, 7, 25), NullCalendar(), BusinessDayConvention.FOLLOWING) == date(
        2026, 7, 27
    )


def test_preceding_rolls_a_saturday_back_to_friday() -> None:
    assert adjust(date(2026, 7, 25), NullCalendar(), BusinessDayConvention.PRECEDING) == date(
        2026, 7, 24
    )


def test_modified_following_stays_inside_the_month() -> None:
    """31 May 2026 is a Sunday; Following would cross into June, so Modified
    Following turns around and takes the preceding business day instead."""
    assert adjust(
        date(2026, 5, 31), NullCalendar(), BusinessDayConvention.MODIFIED_FOLLOWING
    ) == date(2026, 5, 29)


def test_unadjusted_returns_the_date_untouched() -> None:
    assert adjust(date(2026, 7, 25), SwedenCalendar(), BusinessDayConvention.UNADJUSTED) == date(
        2026, 7, 25
    )


def test_add_months_clamps_to_the_end_of_a_shorter_month() -> None:
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2026, 3, 15), -3) == date(2025, 12, 15)


def test_schedule_generates_backwards_from_maturity() -> None:
    """Coupon dates hang off maturity, not issue: a bond issued off-cycle has a
    short or long first period, never a stub at the end."""
    dates = schedule(
        date(2026, 3, 10),
        date(2029, 5, 15),
        frequency=2,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )

    assert dates[0] == date(2026, 3, 10)
    assert dates[-1] == date(2029, 5, 15)
    assert dates[1] == date(2026, 5, 15)
    assert len(dates) == 8


def test_schedule_periods_preserve_the_maturity_end_of_month_anchor() -> None:
    periods = conventions.schedule_periods(
        date(2024, 8, 31),
        date(2026, 8, 31),
        frequency=2,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )

    assert tuple(period.accrual_end for period in periods) == (
        date(2025, 2, 28),
        date(2025, 8, 31),
        date(2026, 2, 28),
        date(2026, 8, 31),
    )


def test_schedule_period_keeps_accrual_unadjusted_and_adjusts_only_payment() -> None:
    (period,) = conventions.schedule_periods(
        date(2026, 1, 1),
        date(2027, 1, 1),
        frequency=1,
        calendar=SwedenCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )

    assert (period.accrual_start, period.accrual_end) == (
        date(2026, 1, 1),
        date(2027, 1, 1),
    )
    assert period.payment_date == date(2027, 1, 4)
    assert (period.reference_start, period.reference_end) == (
        date(2026, 1, 1),
        date(2027, 1, 1),
    )


def test_schedule_period_is_immutable() -> None:
    (period,) = conventions.schedule_periods(
        date(2026, 1, 1),
        date(2027, 1, 1),
        frequency=1,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    field_name = "payment_date"
    with pytest.raises(FrozenInstanceError):
        setattr(period, field_name, date(2027, 1, 2))


@pytest.mark.parametrize("accrual_end", [date(2026, 1, 1), date(2025, 12, 31)])
def test_schedule_period_rejects_invalid_accrual_interval(accrual_end: date) -> None:
    with pytest.raises(ValueError, match="accrual_end"):
        conventions.SchedulePeriod(
            accrual_start=date(2026, 1, 1),
            accrual_end=accrual_end,
            payment_date=accrual_end,
            reference_start=date(2026, 1, 1),
            reference_end=date(2027, 1, 1),
        )


def test_schedule_adjusts_period_ends_into_business_days() -> None:
    dates = schedule(
        date(2026, 1, 1),
        date(2027, 1, 1),
        frequency=1,
        calendar=SwedenCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )

    assert dates == (date(2026, 1, 1), date(2027, 1, 4))


def test_schedule_rejects_a_frequency_that_does_not_divide_twelve() -> None:
    with pytest.raises(ValueError, match="frequency"):
        schedule(
            date(2026, 1, 1),
            date(2027, 1, 1),
            frequency=5,
            calendar=NullCalendar(),
            bdc=BusinessDayConvention.FOLLOWING,
        )


@pytest.mark.parametrize("end", [date(2026, 1, 1), date(2025, 12, 31)])
def test_schedule_rejects_end_not_after_start(end: date) -> None:
    with pytest.raises(ValueError, match="after start"):
        schedule(
            date(2026, 1, 1),
            end,
            frequency=1,
            calendar=NullCalendar(),
            bdc=BusinessDayConvention.UNADJUSTED,
        )
