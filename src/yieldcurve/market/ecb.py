"""ECB SDMX 2.1: EUR spot curve and Svensson parameters.

The API returns beta parameters as percentages. This adapter stores them as-is
without unit conversion — the conversion to decimals lives in ``parametric.py``.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Final

import pandas as pd
import requests
from requests.exceptions import RequestException

_BASE: Final = "https://data-api.ecb.europa.eu/service/data"
_FLOW: Final = "YC/B.U2.EUR.4F.G_N_A.SV_C_YM"
_TIMEOUT: Final = 30.0

_TENORS: Final = [
    "1Y",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
]
_SVENSSON_PARAMS: Final = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]

_TENOR_MAP: Final = {
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "4Y": 4.0,
    "5Y": 5.0,
    "6Y": 6.0,
    "7Y": 7.0,
    "8Y": 8.0,
    "9Y": 9.0,
    "10Y": 10.0,
    "15Y": 15.0,
    "20Y": 20.0,
    "25Y": 25.0,
    "30Y": 30.0,
}


class FetchError(RuntimeError):
    """An upstream source failed or returned something unparseable."""


def _get(datatype: str, on: date) -> float:
    url = (
        f"{_BASE}/{_FLOW}.{datatype}"
        f"?startPeriod={on.isoformat()}&endPeriod={on.isoformat()}"
        "&format=csvdata"
    )
    try:
        response = requests.get(url, timeout=_TIMEOUT)
    except RequestException as exc:
        raise FetchError(f"ECB {datatype} request failed: {exc}") from exc
    if not response.ok:
        raise FetchError(f"ECB {datatype} returned HTTP {response.status_code}")
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ["key", "date", "value"]
    return float(frame["value"].iloc[-1])


def _parse_spot_curve(raw: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"tenor_years": list(raw), "zero_rate": [v / 100.0 for v in raw.values()]})


def _parse_svensson_parameters(raw: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"parameter": list(raw), "value": list(raw.values())})


def fetch_spot_curve(on: date) -> pd.DataFrame:
    return _parse_spot_curve({_TENOR_MAP[t]: _get(f"SR_{t}", on) for t in _TENORS})


def fetch_svensson_parameters(on: date) -> pd.DataFrame:
    return _parse_svensson_parameters({p: _get(p, on) for p in _SVENSSON_PARAMS})
