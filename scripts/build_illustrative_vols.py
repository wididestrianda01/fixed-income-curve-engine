"""Generate the illustrative ATM swaption normal-volatility grid.

The grid is CONSTRUCTED, not observed: it is not market data and it is not a
fit to any traded price. It exists so Hull-White calibration and the app stay
runnable without licensed settlement-volatility data. The construction is a
fully deterministic closed form (no random numbers, no external inputs), so
regeneration is byte-for-byte reproducible:

    sigma_bp(e, m) = (BASE_BP + HUMP_BP * e * exp(1 - e / PEAK_YEARS)) * exp(-TENOR_DECAY * m)

with e the option expiry in years and m the underlying swap tenor in years.
Expiry and maturity dates are ``asof + floor(years * 365.25)`` days, a
documented approximation; the as-of itself is read from the packaged snapshot
manifest, never hardcoded.

Bare invocation is a dry run: it prints the exact bytes the packaged resource
would contain (preamble + rows) and writes nothing. Writing is explicit:
``--write-packaged`` regenerates the packaged resource
``src/yieldcurve/data/illustrative_swaption_vols.csv`` in place. That file is
read-only at runtime (``Snapshot.save`` on the packaged snapshot raises), so
this flag is the deliberate regeneration path for the committed artifact —
after running it, update the dataset's ``sha256`` in
``src/yieldcurve/data/snapshot_manifest.toml`` (guarded by
``tests/market/test_snapshot_contents.py``). With ``--root``, the grid is
instead written through the Snapshot save contract to
``<root>/<snapshot-date>/illustrative_swaption_vols.csv``.

Run:

    python scripts/build_illustrative_vols.py              # print the grid (dry run)
    python scripts/build_illustrative_vols.py --write-packaged  # regenerate packaged CSV
    python scripts/build_illustrative_vols.py --root DIR   # write an external snapshot
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tomllib
from datetime import date, timedelta
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from yieldcurve.market.snapshot import Snapshot

BASE_BP = 60.0
HUMP_BP = 22.0
PEAK_YEARS = 1.5
TENOR_DECAY = 0.018

EXPIRIES = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
TENORS = (1.0, 2.0, 5.0, 10.0)

_MANIFEST = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")


def snapshot_asof() -> date:
    """The packaged snapshot's as-of date, from its manifest (never hardcoded)."""
    with _MANIFEST.open("rb") as handle:
        return date.fromisoformat(tomllib.load(handle)["snapshot_date"])


def sigma_bp(expiry: float, tenor: float) -> float:
    """Normal vol in basis points from the documented closed form."""
    hump = HUMP_BP * expiry * np.exp(1.0 - expiry / PEAK_YEARS)
    return float((BASE_BP + hump) * np.exp(-TENOR_DECAY * tenor))


def _offset_date(asof: date, years: float) -> date:
    return asof + timedelta(days=int(years * 365.25))


def generate_grid(asof: date) -> pd.DataFrame:
    """The full expiry x tenor grid as a DataFrame (expiry, maturity, vol)."""
    rows = []
    for expiry_years in EXPIRIES:
        expiry_date = _offset_date(asof, expiry_years)
        for tenor_years in TENORS:
            maturity_date = _offset_date(asof, expiry_years + tenor_years)
            rows.append(
                (
                    expiry_date.isoformat(),
                    maturity_date.isoformat(),
                    sigma_bp(expiry_years, tenor_years),
                )
            )
    return pd.DataFrame(rows, columns=["expiry", "maturity", "vol"])


def preamble(asof: date) -> list[str]:
    """The provenance preamble written above the CSV header row."""
    return [
        "# ILLUSTRATIVE, NOT MARKET DATA. These normal volatilities are constructed, not",
        "# observed; they are not a fit to any traded price.",
        f"# Snapshot as-of: {asof.isoformat()} (from src/yieldcurve/data/snapshot_manifest.toml).",
        "# Construction rule, exactly as implemented in scripts/build_illustrative_vols.py:",
        f"#   sigma_bp(e, m) = ({BASE_BP} + {HUMP_BP} * e * exp(1 - e / {PEAK_YEARS}))"
        f" * exp(-{TENOR_DECAY} * m)",
        "#   e = option expiry in years, m = underlying swap tenor in years.",
        "# Expiry/maturity dates are asof + floor(years * 365.25) days.",
        "# Shape is market-plausible: declining in expiry beyond a 1-2y hump, mild decay",
        "# across swap tenor. Fully deterministic: regeneration is byte-identical.",
    ]


def packaged_csv_bytes(asof: date) -> bytes:
    """The exact bytes the packaged resource should contain (preamble + rows)."""
    frame = generate_grid(asof)
    buffer = io.StringIO()
    for line in preamble(asof):
        buffer.write(line + "\n")
    writer = csv.writer(buffer)
    writer.writerow(["expiry", "maturity", "vol"])
    for _, row in frame.iterrows():
        writer.writerow([row["expiry"], row["maturity"], f"{row['vol']:.4f}"])
    return buffer.getvalue().encode("utf-8")


def packaged_path() -> Path:
    """The packaged resource location (src/yieldcurve/data/<name>.csv)."""
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "yieldcurve"
        / "data"
        / "illustrative_swaption_vols.csv"
    )


def write_packaged(asof: date) -> Path:
    """Regenerate the packaged resource in place and return its path."""
    target = packaged_path()
    target.write_bytes(packaged_csv_bytes(asof))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the illustrative ATM swaption normal-vol grid "
            "(constructed, not observed). The as-of date is read from the "
            "packaged snapshot manifest."
        )
    )
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "explicit external snapshot root; writes <root>/<date>/"
            "illustrative_swaption_vols.csv through the Snapshot API instead "
            "of the packaged resource"
        ),
    )
    write_mode.add_argument(
        "--write-packaged",
        action="store_true",
        help=(
            "regenerate src/yieldcurve/data/illustrative_swaption_vols.csv in "
            "place (deliberate maintainer action; update the dataset's sha256 "
            "in snapshot_manifest.toml afterwards). Bare invocation only "
            "prints the generated grid."
        ),
    )
    args = parser.parse_args(argv)

    asof = snapshot_asof()
    if args.root is not None:
        snapshot = Snapshot(date=asof, root=args.root)
        target = snapshot.save("illustrative_swaption_vols", generate_grid(asof))
        print(f"Wrote external snapshot {target}")
    elif args.write_packaged:
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
            "--write-packaged to regenerate the packaged CSV, or --root DIR for "
            "an external snapshot.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
