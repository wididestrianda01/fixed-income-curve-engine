"""Riksbank SWEA API: T-bills, government benchmark yields, SWESTR.

Series identifiers were verified against the live API in Phase 0 Task 0.3
Step 2 and recorded in DATA_SOURCES.md. They are constants here, not guesses.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Final

import pandas as pd
import requests

_BASE: Final = "https://api.riksbank.se/swea/v1"
_TIMEOUT: Final = 30.0

BILL_SERIES: Final[dict[str, str]] = {
    "1M": "SETB1MBENCHC",
    "3M": "SETB3MBENCH",
    "6M": "SETB6MBENCH",
}
GOV_BENCHMARK_SERIES: Final[dict[str, str]] = {
    "2Y": "SEGVB2YC",
    "5Y": "SEGVB5YC",
    "7Y": "SEGVB7YC",
    "10Y": "SEGVB10YC",
}
SWESTR_SERIES: Final[dict[str, str]] = {
    "ON": "SWESTR",
    "1W": "SWESTRAVG1W",
    "1M": "SWESTRAVG1M",
    "2M": "SWESTRAVG2M",
    "3M": "SWESTRAVG3M",
    "6M": "SWESTRAVG6M",
}


class FetchError(RuntimeError):
    """An upstream source failed or returned something unparseable."""


def _observation(series_id: str, on: date) -> float:
    url = f"{_BASE}/Observations/{series_id}/{on.isoformat()}/{on.isoformat()}"
    response = requests.get(url, timeout=_TIMEOUT)
    if not response.ok:
        raise FetchError(f"{series_id} returned HTTP {response.status_code} for {on}")
    payload: list[dict[str, Any]] = response.json()
    if not payload:
        raise FetchError(f"{series_id} has no observation on {on}; is it a business day?")
    return float(payload[0]["value"])


def _parse_bills(raw: dict[str, float], maturities: dict[str, date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": list(raw),
            "maturity_date": [maturities[t].isoformat() for t in raw],
            "rate": [v / 100.0 for v in raw.values()],
        }
    )


def _parse_gov_benchmarks(raw: dict[str, float], maturities: dict[str, date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": list(raw),
            "maturity_date": [maturities[t].isoformat() for t in raw],
            "yield": [v / 100.0 for v in raw.values()],
        }
    )


def _parse_swestr(raw: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"tenor": list(raw), "rate": [v / 100.0 for v in raw.values()]})


def fetch_bills(on: date, maturities: dict[str, date]) -> pd.DataFrame:
    return _parse_bills({t: _observation(s, on) for t, s in BILL_SERIES.items()}, maturities)


def fetch_gov_benchmarks(on: date, maturities: dict[str, date]) -> pd.DataFrame:
    return _parse_gov_benchmarks(
        {t: _observation(s, on) for t, s in GOV_BENCHMARK_SERIES.items()}, maturities
    )


def fetch_swestr(on: date) -> pd.DataFrame:
    return _parse_swestr({t: _observation(s, on) for t, s in SWESTR_SERIES.items()})
