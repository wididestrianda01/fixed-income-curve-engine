"""Generate the illustrative swaption normal-volatility *smile* (strike dimension).

The ATM grid (``build_illustrative_vols.py``) carries one normal vol per
(expiry, tenor). A smile adds the strike dimension: for a fixed illustrative
ATM forward F, the normal vol varies across strikes with a documented downward
skew and convexity:

    sigma_N(K; e, m) = sigma_atm(e, m) * (1 + SKEW * u + CURVE * u^2 + QUARTIC * u^4)
    u = (K - F) / F

with ``sigma_atm(e, m)`` the same ATM construction as
``build_illustrative_vols.py`` (redefined here so this script stays
self-contained). The quartic term is small and deliberately kept: a SABR smile
is only approximately quadratic, so calibrating SABR to this shape leaves a
small measured residual rather than a planted exact fit. The values are
CONSTRUCTED, not observed: they are not market data and not a fit to any traded
price. The shape (negative skew, positive convexity) is market-plausible by
construction only.

Regeneration is byte-for-byte reproducible. Bare invocation is a dry run;
``--write-packaged`` regenerates
``src/yieldcurve/data/illustrative_swaption_smile.csv`` in place, after which
the dataset's ``sha256`` in ``snapshot_manifest.toml`` must be updated (guarded
by ``tests/market/test_snapshot_contents.py``).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import tomllib
from datetime import date, timedelta
from importlib import resources
from pathlib import Path

# ATM construction, identical to build_illustrative_vols.py (kept local so this
# script has no cross-module import).
BASE_BP = 60.0
HUMP_BP = 22.0
PEAK_YEARS = 1.5
TENOR_DECAY = 0.018

# The illustrative ATM forward (USD, 3%) around which strikes are quoted.
FORWARD = 0.03
# Smile shape parameters: negative skew (normal vol falls as the strike rises),
# positive convexity, and a small quartic term so the smile is not exactly
# quadratic (SABR leaves a measured residual rather than fitting exactly).
SKEW = -0.40
CURVE = 0.25
QUARTIC = 0.06

# The smile is generated for a small grid: three option expiries on one swap
# tenor, across strikes +-150 bp around the forward in 50 bp steps.
EXPIRIES = (1.0, 2.0, 5.0)
TENOR = 5.0
STRIKE_DELTAS = (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015)

_MANIFEST = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")


def snapshot_asof() -> date:
    """The packaged snapshot's as-of date, from its manifest (never hardcoded)."""
    with _MANIFEST.open("rb") as handle:
        return date.fromisoformat(tomllib.load(handle)["snapshot_date"])


def sigma_atm_bp(expiry: float, tenor: float) -> float:
    """ATM normal vol in basis points (identical to build_illustrative_vols.py)."""
    hump = HUMP_BP * expiry * math.exp(1.0 - expiry / PEAK_YEARS)
    return float((BASE_BP + hump) * math.exp(-TENOR_DECAY * tenor))


def _offset_date(asof: date, years: float) -> date:
    return asof + timedelta(days=int(years * 365.25))


def smile_bp(expiry: float, tenor: float, strike: float) -> float:
    """Normal vol in basis points at ``strike`` from the documented closed form."""
    atm = sigma_atm_bp(expiry, tenor)
    u = (strike - FORWARD) / FORWARD
    return float(atm * (1.0 + SKEW * u + CURVE * u * u + QUARTIC * u * u * u * u))


def generate_smile(asof: date) -> list[tuple[str, str, float, float]]:
    """Rows of (expiry, maturity, strike, vol) with vol in basis points."""
    rows = []
    for expiry_years in EXPIRIES:
        expiry_date = _offset_date(asof, expiry_years)
        maturity_date = _offset_date(asof, expiry_years + TENOR)
        for delta in STRIKE_DELTAS:
            strike = round(FORWARD + delta, 6)
            rows.append(
                (
                    expiry_date.isoformat(),
                    maturity_date.isoformat(),
                    strike,
                    smile_bp(expiry_years, TENOR, strike),
                )
            )
    return rows


def preamble(asof: date) -> list[str]:
    """The provenance preamble written above the CSV header row."""
    return [
        "# ILLUSTRATIVE, NOT MARKET DATA. These normal volatilities are constructed, not",
        "# observed; they are not a fit to any traded price.",
        f"# Snapshot as-of: {asof.isoformat()} (from src/yieldcurve/data/snapshot_manifest.toml).",
        "# Construction rule, exactly as implemented in scripts/build_illustrative_smile.py:",
        f"#   sigma_N(K; e, m) = sigma_atm(e, m) * (1 + {SKEW} * u + {CURVE} * u^2"
        f" + {QUARTIC} * u^4)",
        f"#   u = (K - {FORWARD}) / {FORWARD}; sigma_atm from build_illustrative_vols.py.",
        f"# Expiries {EXPIRIES} years on swap tenor {TENOR} years; strikes the ATM forward",
        f"#   plus deltas {STRIKE_DELTAS}.",
        "# Expiry/maturity dates are asof + floor(years * 365.25) days.",
        "# Shape is market-plausible: negative skew, positive convexity, small quartic.",
        "# Fully deterministic: regeneration is byte-identical.",
    ]


def packaged_csv_bytes(asof: date) -> bytes:
    """The exact bytes the packaged resource should contain (preamble + rows)."""
    buffer = io.StringIO()
    for line in preamble(asof):
        buffer.write(line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["expiry", "maturity", "strike", "vol"])
    for expiry, maturity, strike, vol in generate_smile(asof):
        writer.writerow([expiry, maturity, f"{strike:.6f}", f"{vol:.4f}"])
    return buffer.getvalue().encode("utf-8")


def packaged_path() -> Path:
    """The packaged resource location (src/yieldcurve/data/<name>.csv)."""
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "yieldcurve"
        / "data"
        / "illustrative_swaption_smile.csv"
    )


def write_packaged(asof: date) -> Path:
    """Regenerate the packaged resource in place and return its path."""
    target = packaged_path()
    target.write_bytes(packaged_csv_bytes(asof))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the illustrative swaption normal-vol smile "
            "(constructed, not observed). The as-of date is read from the "
            "packaged snapshot manifest."
        )
    )
    parser.add_argument(
        "--write-packaged",
        action="store_true",
        help=(
            "regenerate src/yieldcurve/data/illustrative_swaption_smile.csv in "
            "place (deliberate maintainer action; update the dataset's sha256 "
            "in snapshot_manifest.toml afterwards). Bare invocation only "
            "prints the generated grid."
        ),
    )
    args = parser.parse_args(argv)

    asof = snapshot_asof()
    if args.write_packaged:
        target = write_packaged(asof)
        print(
            f"Regenerated {target}\n"
            "Update the dataset's sha256 in src/yieldcurve/data/snapshot_manifest.toml "
            "after a deliberate regeneration (tests guard the checksum)."
        )
    else:
        print(packaged_csv_bytes(asof).decode("utf-8"), end="")
        print(
            "Dry run: generated the grid above, nothing was written. Pass "
            "--write-packaged to regenerate the packaged CSV.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
