"""Generate the illustrative ATM swaption normal-vol grid.

These volatilities are CONSTRUCTED, not observed. CME cleared-swaption settlement files
require a CME Information License Agreement and may not be redistributed, which is why
src/yieldcurve/market/cme.py raises rather than caches. The grid below has a
market-plausible shape — declining in expiry, humped around 1-2y, mildly decaying across
swap tenor — produced by a closed form that is stated in full so that anyone can reproduce
or replace it.

    sigma(e, m) = (base + hump * e * exp(1 - e / peak)) * exp(-decay * m)

with e the option expiry in years and m the underlying swap tenor in years. Run:

    python scripts/build_illustrative_vols.py
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ASOF = date(2026, 7, 24)
BASE_BP = 60.0
HUMP_BP = 22.0
PEAK_YEARS = 1.5
TENOR_DECAY = 0.018

EXPIRIES = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
TENORS = (1.0, 2.0, 5.0, 10.0)

OUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "snapshots"
    / "2026-07-24"
    / "illustrative_swaption_vols.csv"
)

HEADER = [
    "# ILLUSTRATIVE, NOT MARKET DATA. These normal volatilities are constructed, not",
    "# observed. Real CME cleared-swaption settlement vols require a CME Information",
    "# License Agreement and may not be redistributed in this repository.",
    "# Construction rule, exactly as implemented in scripts/build_illustrative_vols.py:",
    f"#   sigma_bp(e, m) = ({BASE_BP} + {HUMP_BP} * e * exp(1 - e / {PEAK_YEARS}))"
    f" * exp(-{TENOR_DECAY} * m)",
    "#   e = option expiry in years, m = underlying swap tenor in years.",
    "# Shape is market-plausible: declining in expiry beyond the hump, humped around 1-2y,",
    "# mild decay across swap tenor. It is not a fit to any traded price.",
]


def sigma_bp(expiry: float, tenor: float) -> float:
    hump = HUMP_BP * expiry * np.exp(1.0 - expiry / PEAK_YEARS)
    return float((BASE_BP + hump) * np.exp(-TENOR_DECAY * tenor))


def main() -> None:
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        for line in HEADER:
            handle.write(line + "\n")
        writer = csv.writer(handle)
        writer.writerow(["expiry", "maturity", "vol"])
        for expiry_years in EXPIRIES:
            expiry_date = ASOF + timedelta(days=int(expiry_years * 365.25))
            for tenor_years in TENORS:
                maturity_date = ASOF + timedelta(days=int((expiry_years + tenor_years) * 365.25))
                vol = sigma_bp(expiry_years, tenor_years)
                writer.writerow([expiry_date.isoformat(), maturity_date.isoformat(), f"{vol:.4f}"])


if __name__ == "__main__":
    main()
