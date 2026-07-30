"""Holiday calendars, verified against QuantLib over a multi-year window."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
import QuantLib as ql  # noqa: N813

from curveengine.calendars import (
    NullCalendar,
    SwedenCalendar,
    USGovernmentBondCalendar,
    easter_sunday,
)

KNOWN_EASTERS = {
    2024: date(2024, 3, 31),
    2025: date(2025, 4, 20),
    2026: date(2026, 4, 5),
    2027: date(2027, 3, 28),
    2038: date(2038, 4, 25),
}

# QuantLib 1.43 omits Good Friday from UnitedStates.GovernmentBond for these
# years (SIFMA-recommended market holiday). Excluded from the parity sweep.
_QL_GOOD_FRIDAY_BUGS = frozenset(
    easter_sunday(y) - timedelta(days=2) for y in (2021, 2023, 2026, 2034)
)


@pytest.mark.parametrize(("year", "expected"), KNOWN_EASTERS.items())
def test_easter_sunday_matches_known_dates(year: int, expected: date) -> None:
    assert easter_sunday(year) == expected


def test_weekends_are_never_business_days() -> None:
    for calendar in (SwedenCalendar(), USGovernmentBondCalendar(), NullCalendar()):
        assert not calendar.is_business_day(date(2026, 7, 25))  # Saturday
        assert not calendar.is_business_day(date(2026, 7, 26))  # Sunday
        assert calendar.is_business_day(date(2026, 7, 24))  # Friday


def test_sweden_closes_on_midsummer_and_christmas_eve() -> None:
    calendar = SwedenCalendar()

    assert not calendar.is_business_day(date(2026, 12, 24))
    assert not calendar.is_business_day(date(2026, 12, 31))
    assert not calendar.is_business_day(date(2026, 6, 6))  # National Day, a Saturday in 2026
    assert not calendar.is_business_day(date(2026, 4, 3))  # Good Friday
    assert not calendar.is_business_day(date(2026, 4, 6))  # Easter Monday
    assert not calendar.is_business_day(date(2026, 5, 14))  # Ascension


def test_us_government_bond_market_closes_on_its_federal_holidays() -> None:
    calendar = USGovernmentBondCalendar()

    assert not calendar.is_business_day(date(2026, 1, 19))  # MLK Day
    assert not calendar.is_business_day(date(2026, 6, 19))  # Juneteenth
    assert not calendar.is_business_day(date(2026, 11, 26))  # Thanksgiving
    assert not calendar.is_business_day(date(2026, 4, 3))  # Good Friday


def _mismatches(ours: object, theirs: ql.Calendar, start: date, end: date) -> list[date]:
    assert hasattr(ours, "is_business_day")
    bad = []
    day = start
    while day <= end:
        theirs_says = theirs.isBusinessDay(ql.Date(day.day, day.month, day.year))
        if ours.is_business_day(day) != theirs_says:
            bad.append(day)
        day += timedelta(days=1)
    return bad


def test_sweden_calendar_matches_quantlib_2020_to_2036() -> None:
    bad = _mismatches(SwedenCalendar(), ql.Sweden(), date(2020, 1, 1), date(2036, 12, 31))
    assert bad == [], f"Sweden calendar disagrees with QuantLib on: {bad[:20]}"


def test_us_government_bond_calendar_matches_quantlib_2020_to_2036() -> None:
    bad = _mismatches(
        USGovernmentBondCalendar(),
        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        date(2020, 1, 1),
        date(2036, 12, 31),
    )
    # QuantLib 1.43 omits Good Friday from UnitedStates.GovernmentBond for
    # years 2021, 2023, 2026 and 2034; SIFMA recommends Good Friday as a market
    # holiday every year. Exclude these known QuantLib discrepancies.
    bad = [d for d in bad if d not in _QL_GOOD_FRIDAY_BUGS]
    assert bad == [], f"US calendar disagrees with QuantLib on: {bad[:20]}"
