"""CME: swaption settlement normal volatilities.

CME cleared-swaption settlement files require a CME Information License
Agreement (ILA). Volatilities may not be redistributed in this repository.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


class FetchError(RuntimeError):
    """An upstream source failed or returned something unparseable."""


def fetch_swaption_vols(_on: date) -> pd.DataFrame:
    raise FetchError(
        "CME swaption settlement files require a CME Information License Agreement. "
        "See DATA_SOURCES.md § CME Swaption Settlement Vols."
    )
