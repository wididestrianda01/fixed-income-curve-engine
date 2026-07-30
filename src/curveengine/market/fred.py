"""FRED: ICE SOFR swap rates, H.15 Treasury constant-maturity yields.

FRED serves CSV without a key at ``fredgraph.csv?id=...``. The history endpoint
is the same URL with a date range, which is why one parser covers both the
cross-section and the history: they differ only in how many rows come back.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Final

import pandas as pd
import requests

_CSV: Final = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT: Final = 30.0

CMT_SERIES: Final[dict[float, str]] = {
    0.0833: "DGS1MO",
    0.25: "DGS3MO",
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}
OIS_SERIES: Final[dict[float, str]] = {
    1.0: "ICESOFRSWAP1Y",
    2.0: "ICESOFRSWAP2Y",
    3.0: "ICESOFRSWAP3Y",
    5.0: "ICESOFRSWAP5Y",
    7.0: "ICESOFRSWAP7Y",
    10.0: "ICESOFRSWAP10Y",
    30.0: "ICESOFRSWAP30Y",
}


class FetchError(RuntimeError):
    """An upstream source failed or returned something unparseable."""


def _series(series_id: str, start: date, end: date) -> pd.DataFrame:
    params = {
        "id": series_id,
        "cosd": start.isoformat(),
        "coed": end.isoformat(),
    }
    response = requests.get(_CSV, params=params, timeout=_TIMEOUT)
    if not response.ok:
        raise FetchError(f"FRED {series_id} returned HTTP {response.status_code}")
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ["date", "value"]
    frame = frame[frame["value"] != "."]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def _parse_treasury_cmt(raw: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": [CMT_SERIES[t] for t in raw],
            "tenor_years": list(raw),
            "rate": [v / 100.0 for v in raw.values()],
        }
    )


def _parse_ois_swaps(raw: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"tenor_years": list(raw), "par_rate": [v / 100.0 for v in raw.values()]})


def _parse_cmt_history(frames: dict[float, pd.DataFrame]) -> pd.DataFrame:
    common = set.intersection(*(set(f["date"]) for f in frames.values()))
    rows = [
        {"date": d, "tenor_years": tenor, "rate": value / 100.0}
        for tenor, frame in frames.items()
        for d, value in zip(frame["date"], frame["value"], strict=True)
        if d in common
    ]
    return pd.DataFrame(rows).sort_values(["date", "tenor_years"]).reset_index(drop=True)


def fetch_treasury_cmt(on: date) -> pd.DataFrame:
    raw = {t: _series(s, on, on)["value"].iloc[-1] for t, s in CMT_SERIES.items()}
    return _parse_treasury_cmt(raw)


def fetch_ois_swaps(on: date) -> pd.DataFrame:
    raw = {t: _series(s, on, on)["value"].iloc[-1] for t, s in OIS_SERIES.items()}
    return _parse_ois_swaps(raw)


def fetch_treasury_cmt_history(start: date, end: date) -> pd.DataFrame:
    return _parse_cmt_history({t: _series(s, start, end) for t, s in CMT_SERIES.items()})
