"""Business-day calendars for the markets in scope.

Written by hand rather than imported. QuantLib is this project's parity oracle
and appears in ``tests/`` only; the calendars here are verified against it over
2020-2036 in ``tests/test_calendars.py``.

Known QuantLib 1.43 discrepancies are isolated in tests. Current SIFMA
guidance treats Good Friday 2026 as an early close rather than a full closure;
this date-only calendar therefore keeps it as a business day.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Protocol, runtime_checkable

_SATURDAY = 5


@runtime_checkable
class Calendar(Protocol):
    """Anything that can say whether a date is a business day."""

    name: str

    def is_business_day(self, d: date) -> bool: ...


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday by the anonymous algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Monday=0) of a month; ``n = -1`` means the last."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    if n == -1:
        next_month = date(year + month // 12, month % 12 + 1, 1)
        last = next_month - timedelta(days=1)
        return last - timedelta(days=(last.weekday() - weekday) % 7)
    raise ValueError(f"n must be a positive integer or -1, got {n}")


class SwedenCalendar:
    """Swedish bank and market holidays.

    Midsummer Eve, Christmas Eve and New Year's Eve are not statutory public
    holidays but the market is closed, so they are treated as holidays here.
    """

    name = "Sweden"

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < _SATURDAY and d not in _sweden_holidays(d.year)

    def __repr__(self) -> str:
        return "SwedenCalendar()"


class USGovernmentBondCalendar:
    """SIFMA US government securities market holidays.

    A holiday falling on a Sunday is observed the following Monday; one falling
    on a Saturday is observed the preceding Friday for Independence Day,
    Christmas Day and Juneteenth only. New Year's Day and Veterans Day on a
    Saturday are not given an adjacent weekday closure under the
    government-bond-market rule.
    """

    name = "UnitedStates.GovernmentBond"

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < _SATURDAY and d not in _us_bond_holidays(d.year)

    def __repr__(self) -> str:
        return "USGovernmentBondCalendar()"


class NullCalendar:
    """Every weekday is a business day. Isolates schedule logic in tests."""

    name = "Null"

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < _SATURDAY

    def __repr__(self) -> str:
        return "NullCalendar()"


@lru_cache(maxsize=256)
def _sweden_holidays(year: int) -> frozenset[date]:
    easter = easter_sunday(year)
    midsummer_saturday = _nth_weekday(year, 6, _SATURDAY, 1)
    while not 20 <= midsummer_saturday.day <= 26:
        midsummer_saturday += timedelta(days=7)
    return frozenset(
        {
            date(year, 1, 1),  # New Year's Day
            date(year, 1, 6),  # Epiphany
            easter - timedelta(days=2),  # Good Friday
            easter + timedelta(days=1),  # Easter Monday
            date(year, 5, 1),  # Labour Day
            easter + timedelta(days=39),  # Ascension Day
            date(year, 6, 6),  # National Day
            midsummer_saturday - timedelta(days=1),  # Midsummer Eve
            date(year, 12, 24),  # Christmas Eve
            date(year, 12, 25),
            date(year, 12, 26),
            date(year, 12, 31),  # New Year's Eve
        }
    )


@lru_cache(maxsize=256)
def _us_bond_holidays(year: int) -> frozenset[date]:
    easter = easter_sunday(year)
    # Fixed-date holidays. Saturday->Friday adjustment applies only to
    # Independence Day, Christmas Day and Juneteenth (not New Year's Day or
    # Veterans Day, per QuantLib UnitedStates.GovernmentBond parity).
    _sat_adjusted = {date(year, 7, 4), date(year, 12, 25)}
    fixed_names = [
        date(year, 1, 1),  # New Year's Day
        date(year, 7, 4),  # Independence Day
        date(year, 11, 11),  # Veterans Day
        date(year, 12, 25),  # Christmas Day
    ]
    if year >= 2022:
        _sat_adjusted.add(date(year, 6, 19))
        fixed_names.append(date(year, 6, 19))  # Juneteenth
    observed: set[date] = set()
    for d in fixed_names:
        wd = d.weekday()
        if wd == 6:
            observed.add(d + timedelta(days=1))
        elif wd == _SATURDAY and d in _sat_adjusted:
            observed.add(d - timedelta(days=1))
        else:
            observed.add(d)
    full_day_good_friday: set[date] = set()
    if year != 2026:
        full_day_good_friday.add(easter - timedelta(days=2))
    return frozenset(
        observed
        | full_day_good_friday
        | {
            _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr Day
            _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
            _nth_weekday(year, 5, 0, -1),  # Memorial Day
            _nth_weekday(year, 9, 0, 1),  # Labor Day
            _nth_weekday(year, 10, 0, 2),  # Columbus Day
            _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        }
    )
