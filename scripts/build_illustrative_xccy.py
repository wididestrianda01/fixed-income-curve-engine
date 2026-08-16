"""Generate the illustrative EUR/USD cross-currency basis curve (tenor dimension).

The curve is *constructed*, not observed: it is not market data and not a fit
to any traded price. It carries a market-plausible shape by construction — a
basis that starts near zero at the short end and widens to a negative level at
the long end, the sign and shape of the EUR/USD basis in a USD-funding-premium
regime — so the cross-currency and CSA-discounting machinery in
``src/yieldcurve/curves/xccy.py`` has a documented input to build from.

The closed form is ``basis_bp(t) = LEVEL_BP * (1 - exp(-t / TAU))`` with a
negative ``LEVEL_BP``, evaluated on a fixed tenor grid. Fully deterministic:
regeneration is byte-identical.
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

# The long-horizon basis level (bp) and the decay time constant (years). The
# basis widens toward LEVEL_BP as tenor grows, which is how a dollar-funding
# premium shows up in the cross-currency basis term structure.
LEVEL_BP = -28.0
TAU_YEARS = 3.0

# Tenor grid in years: 3M out to 30Y, dense at the short end where the basis
# moves fastest.
TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

_MANIFEST = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")


def snapshot_asof() -> date:
    """The packaged snapshot's as-of date, from its manifest (never hardcoded)."""
    with _MANIFEST.open("rb") as handle:
        return date.fromisoformat(tomllib.load(handle)["snapshot_date"])


def basis_bp(tenor: float) -> float:
    """The EUR/USD cross-currency basis in basis points at ``tenor`` years."""
    return float(LEVEL_BP * (1.0 - math.exp(-tenor / TAU_YEARS)))


def generate_basis() -> list[tuple[float, float]]:
    """Rows of ``(tenor_years, basis_bp)`` on the fixed tenor grid."""
    return [(tenor, basis_bp(tenor)) for tenor in TENORS]


def preamble(asof: date) -> list[str]:
    """The provenance preamble written above the CSV header row."""
    return [
        "# ILLUSTRATIVE, NOT MARKET DATA. This cross-currency basis is constructed, not",
        "# observed; it is not a fit to any traded price.",
        f"# Snapshot as-of: {asof.isoformat()} (from src/yieldcurve/data/snapshot_manifest.toml).",
        "# Construction rule, exactly as implemented in scripts/build_illustrative_xccy.py:",
        f"#   basis_bp(t) = {LEVEL_BP} * (1 - exp(-t / {TAU_YEARS})), t in years.",
        f"# Tenors {list(TENORS)} years.",
        "# Shape is market-plausible: near zero at the short end, widening to a",
        "# negative level at the long end (the USD-funding-premium sign).",
        "# Fully deterministic: regeneration is byte-identical.",
    ]


def packaged_csv_bytes(asof: date) -> bytes:
    """The exact bytes the packaged resource should contain (preamble + rows)."""
    buffer = io.StringIO()
    for line in preamble(asof):
        buffer.write(line + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["tenor_years", "basis_bp"])
    for tenor, spread in generate_basis():
        writer.writerow([f"{tenor:g}", f"{spread:.4f}"])
    return buffer.getvalue().encode("utf-8")


def packaged_path() -> Path:
    """The packaged resource location (src/yieldcurve/data/<name>.csv)."""
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "yieldcurve"
        / "data"
        / "illustrative_xccy_basis.csv"
    )


def write_packaged(asof: date) -> Path:
    """Regenerate the packaged resource in place and return its path."""
    target = packaged_path()
    target.write_bytes(packaged_csv_bytes(asof))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the illustrative EUR/USD cross-currency basis "
            "(constructed, not observed). The as-of date is read from the "
            "packaged snapshot manifest."
        )
    )
    parser.add_argument(
        "--write-packaged",
        action="store_true",
        help=(
            "regenerate src/yieldcurve/data/illustrative_xccy_basis.csv in "
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
