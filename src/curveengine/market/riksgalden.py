"""Riksgalden: Swedish government bond reference data.

Static data sourced from the Central Government Debt reports
(https://www.riksgalden.se/en/statistics/statistics-regarding-government-securities/).
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date

import pandas as pd

_BONDS = [
    ("SE0007125927", 0.0100, date(2015, 5, 22), date(2026, 11, 12), 96_414_000_000),
    ("SE0009496367", 0.0075, date(2017, 1, 27), date(2028, 5, 12), 80_513_000_000),
    ("SE0011281922", 0.0075, date(2018, 6, 1), date(2029, 11, 12), 90_339_000_000),
    ("SE0013935319", 0.00125, date(2020, 3, 27), date(2031, 5, 12), 63_390_000_000),
    ("SE0004517290", 0.0225, date(2012, 3, 20), date(2032, 6, 1), 48_597_000_000),
    ("SE0017830730", 0.0175, date(2022, 5, 6), date(2033, 11, 11), 60_960_000_000),
    ("SE0021308541", 0.0225, date(2024, 2, 2), date(2035, 5, 11), 69_250_000_000),
    ("SE0025137862", 0.0250, date(2025, 6, 9), date(2036, 10, 15), 18_800_000_000),
    ("SE0002829192", 0.0350, date(2009, 3, 30), date(2039, 3, 30), 45_466_450_000),
    ("SE0015193313", 0.0050, date(2020, 11, 24), date(2045, 11, 24), 18_972_000_000),
    ("SE0016102115", 0.01375, date(2021, 6, 23), date(2071, 6, 23), 10_250_000_000),
]

_BILL_TENORS: dict[str, int] = {"1M": 1, "3M": 3, "6M": 6}
_BENCHMARK_TENORS: dict[str, int] = {"2Y": 2, "5Y": 5, "7Y": 7, "10Y": 10}


class FetchError(RuntimeError):
    """An upstream source failed or returned something unparseable."""


def _parse_gov_bonds() -> pd.DataFrame:
    return pd.DataFrame(
        _BONDS,
        columns=["isin", "coupon", "issue_date", "maturity_date", "outstanding_nominal"],
    )


def fetch_gov_bonds() -> pd.DataFrame:
    return _parse_gov_bonds()


def _add_months(base: date, months: int) -> date:
    month = base.month + months - 1
    year = base.year + month // 12
    month = month % 12 + 1
    day = min(base.day, monthrange(year, month)[1])
    return date(year, month, day)


def maturities_by_tenor(_bonds: pd.DataFrame, on: date) -> dict[str, date]:
    result: dict[str, date] = {}
    for tenor, months in _BILL_TENORS.items():
        result[tenor] = _add_months(on, months)
    for tenor, years in _BENCHMARK_TENORS.items():
        result[tenor] = on.replace(year=on.year + years)
    return result
