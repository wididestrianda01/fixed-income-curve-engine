"""Day-count fractions and compounding conversions."""

from __future__ import annotations

import math
from datetime import date

import pytest

from curveengine.conventions import (
    Compounding,
    DayCount,
    discount_factor,
    to_continuous,
    year_fraction,
)


def test_act_360_counts_actual_days_over_360() -> None:
    assert year_fraction(date(2026, 1, 1), date(2026, 7, 1), DayCount.ACT_360) == pytest.approx(
        181 / 360
    )


def test_act_365f_ignores_leap_years() -> None:
    """ACT/365 Fixed always divides by 365, even across a leap day."""
    assert year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCount.ACT_365F) == pytest.approx(
        366 / 365
    )


def test_thirty_360_treats_every_month_as_thirty_days() -> None:
    assert year_fraction(
        date(2026, 1, 15), date(2026, 7, 15), DayCount.THIRTY_360_BOND
    ) == pytest.approx(0.5)


def test_thirty_360_clamps_a_31st_start_to_the_30th() -> None:
    """Bond basis: d1 = min(d1, 30). 31 Jan to 28 Feb is 28 days, not 27."""
    assert year_fraction(
        date(2026, 1, 31), date(2026, 2, 28), DayCount.THIRTY_360_BOND
    ) == pytest.approx(28 / 360)


def test_thirty_360_clamps_a_31st_end_only_when_the_start_was_the_30th() -> None:
    """d2 = min(d2, 30) applies only if d1 was already 30 after its own clamp."""
    assert year_fraction(
        date(2026, 1, 30), date(2026, 3, 31), DayCount.THIRTY_360_BOND
    ) == pytest.approx(60 / 360)
    assert year_fraction(
        date(2026, 1, 29), date(2026, 3, 31), DayCount.THIRTY_360_BOND
    ) == pytest.approx(62 / 360)


def test_act_act_icma_is_the_period_fraction_over_the_frequency() -> None:
    """Half of a semiannual period is 1/(2*2) of a year, whatever the day count."""
    result = year_fraction(
        date(2026, 2, 15),
        date(2026, 5, 15),
        DayCount.ACT_ACT_ICMA,
        period_start=date(2026, 2, 15),
        period_end=date(2026, 8, 15),
        frequency=2,
    )
    assert result == pytest.approx((89 / 181) * 0.5)


def test_act_act_icma_without_period_information_raises() -> None:
    with pytest.raises(ValueError, match="period_start"):
        year_fraction(date(2026, 2, 15), date(2026, 5, 15), DayCount.ACT_ACT_ICMA)


def test_year_fraction_is_negative_when_the_dates_are_reversed() -> None:
    """Reversal must not silently produce a positive number; a reversed accrual
    period is a caller bug, and a negative fraction makes it visible."""
    assert year_fraction(date(2026, 7, 1), date(2026, 1, 1), DayCount.ACT_360) < 0


@pytest.mark.parametrize(
    ("compounding", "expected"),
    [
        (Compounding.SIMPLE, 1 / 1.05),
        (Compounding.ANNUAL, 1.05**-1.0),
        (Compounding.SEMIANNUAL, 1.025**-2.0),
        (Compounding.CONTINUOUS, math.exp(-0.05)),
    ],
)
def test_discount_factor_per_compounding_basis(compounding: Compounding, expected: float) -> None:
    assert discount_factor(0.05, 1.0, compounding) == pytest.approx(expected)


def test_act_act_icma_rejects_non_positive_period() -> None:
    with pytest.raises(ValueError, match="period_end"):
        year_fraction(
            date(2026, 5, 15),
            date(2026, 2, 15),
            DayCount.ACT_ACT_ICMA,
            period_start=date(2026, 2, 15),
            period_end=date(2026, 2, 15),
            frequency=2,
        )


def test_to_continuous_rejects_non_positive_t() -> None:
    with pytest.raises(ValueError, match="positive"):
        to_continuous(0.05, 0.0, Compounding.SIMPLE)


def test_to_continuous_inverts_discount_factor() -> None:
    """A quote converted to continuous compounding must discount identically."""
    for compounding in Compounding:
        continuous = to_continuous(0.05, 2.0, compounding)
        assert discount_factor(continuous, 2.0, Compounding.CONTINUOUS) == pytest.approx(
            discount_factor(0.05, 2.0, compounding)
        )
