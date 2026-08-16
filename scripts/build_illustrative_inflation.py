"""Generate the illustrative zero-coupon breakeven-inflation curve (tenor dimension).

The inflation module prices linkers and zero-coupon inflation swaps off a
*real* curve, which is the nominal curve minus a zero-coupon breakeven spread.
This script fabricates that breakeven spread across tenors with a documented,
market-plausible shape:

    breakeven(T) = LONG_RUN + HUMP * (T / PEAK_YEARS) * exp(1 - T / PEAK_YEARS)

with ``LONG_RUN`` the long-run breakeven level, ``HUMP`` the peak *excess* over
the long-run level, and ``PEAK_YEARS`` the tenor of that peak. The breakevens
are continuously compounded zero-coupon rates stored as decimals (0.023 = 2.3%),
matching the repository's continuous-compounding convention: a nominal zero rate
``n(T)`` and a breakeven ``b(T)`` give the real zero rate ``r(T) = n(T) - b(T)``.

The values are ILLUSTRATIVE, not observed: they are not market data, not a
measurement of any traded inflation market, and not a CPI forecast. The shape
(a near-term hump decaying to a long-run anchor) is market-plausible by
construction only.

Regeneration is byte-for-byte reproducible. Bare invocation is a dry run;
``--write-packaged`` regenerates
``src/yieldcurve/data/illustrative_inflation_breakevens.csv`` in place, after
which the dataset's ``sha256`` in ``snapshot_manifest.toml`` must be updated
(guarded by ``tests/market/test_snapshot_contents.py``).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import tomllib
from datetime import date
from importlib import resources
from pathlib import Path

# Closed-form construction constants. LONG_RUN is the breakeven the curve
# decays to; HUMP is the peak near-term excess over LONG_RUN, reached at
# PEAK_YEARS. All rates are continuously compounded decimals.
LONG_RUN = 0.0230
HUMP = 0.0120
PEAK_YEARS = 3.0

# A small illustrative tenor grid: short-dated points resolve the hump, the
# tail anchors the long-run level. No sub-1Y points (the packaged USD nominal
# curve has no sub-1Y OIS pillar and short-dated breakevens are noise).
TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

_MANIFEST = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")


def snapshot_asof() -> date:
    """The packaged snapshot's as-of date, from its manifest (never hardcoded)."""
    with _MANIFEST.open("rb") as handle:
        return date.fromisoformat(tomllib.load(handle)["snapshot_date"])


def breakeven(tenor: float) -> float:
    """Continuously compounded zero-coupon breakeven (decimal) at ``tenor`` years."""
    hump = HUMP * (tenor / PEAK_YEARS) * math.exp(1.0 - tenor / PEAK_YEARS)
    return float(LONG_RUN + hump)


def generate_breakevens() -> list[tuple[float, float]]:
    """Rows of ``(tenor_years, breakeven)`` with breakeven as a decimal."""
    return [(tenor, breakeven(tenor)) for tenor in TENORS]


def preamble(asof: date) -> list[str]:
    """The provenance preamble written above the CSV header row."""
    return [
        "# ILLUSTRATIVE, NOT MARKET DATA. These zero-coupon breakevens are constructed,",
        "# not observed; they are not a fit to any traded price and not a CPI forecast.",
        f"# Snapshot as-of: {asof.isoformat()} (from src/yieldcurve/data/snapshot_manifest.toml).",
        "# Construction rule, exactly as implemented in scripts/build_illustrative_inflation.py:",
        f"#   breakeven(T) = {LONG_RUN} + {HUMP} * (T / {PEAK_YEARS}) * exp(1 - T / {PEAK_YEARS})",
        f"# Tenors {TENORS} years. Breakevens are continuously compounded decimals (0.023 = 2.3%).",
        "# Shape is market-plausible: a near-term hump decaying to a long-run anchor.",
        "# Fully deterministic: regeneration is byte-identical.",
    ]


def packaged_csv_bytes(asof: date) -> bytes:
    """The exact bytes the packaged resource should contain (preamble + rows)."""
    buffer = io.StringIO()
    for line in preamble(asof):
        buffer.write(line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["tenor_years", "breakeven"])
    for tenor, rate in generate_breakevens():
        writer.writerow([f"{tenor:.1f}", f"{rate:.6f}"])
    return buffer.getvalue().encode("utf-8")


def packaged_path() -> Path:
    """The packaged resource location (src/yieldcurve/data/<name>.csv)."""
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "yieldcurve"
        / "data"
        / "illustrative_inflation_breakevens.csv"
    )


def write_packaged(asof: date) -> Path:
    """Regenerate the packaged resource in place and return its path."""
    target = packaged_path()
    target.write_bytes(packaged_csv_bytes(asof))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the illustrative zero-coupon breakeven-inflation curve "
            "(constructed, not observed). The as-of date is read from the "
            "packaged snapshot manifest."
        )
    )
    parser.add_argument(
        "--write-packaged",
        action="store_true",
        help=(
            "regenerate src/yieldcurve/data/illustrative_inflation_breakevens.csv "
            "in place (deliberate maintainer action; update the dataset's sha256 "
            "in snapshot_manifest.toml afterwards). Bare invocation only prints "
            "the generated grid."
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
