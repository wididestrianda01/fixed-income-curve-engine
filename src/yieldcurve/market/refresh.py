"""CLI: fetch every adapter and write a new dated snapshot.

The only module in the package that is expected to touch the network, and it is
only ever run by hand. Tests, notebooks and the app read a committed snapshot.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from yieldcurve.market import ecb, fred, riksbank, riksgalden
from yieldcurve.market.snapshot import DEFAULT_SNAPSHOT_ROOT, Snapshot

_HISTORY_YEARS = 5


def _builders(on: date) -> dict[str, Callable[[], pd.DataFrame]]:
    bonds = riksgalden.fetch_gov_bonds()
    maturities = riksgalden.maturities_by_tenor(bonds, on)
    start = on - timedelta(days=365 * _HISTORY_YEARS)
    return {
        "riksbank_bills": lambda: riksbank.fetch_bills(on, maturities),
        "riksbank_gov_benchmarks": lambda: riksbank.fetch_gov_benchmarks(on, maturities),
        "riksbank_swestr": lambda: riksbank.fetch_swestr(on),
        "riksgalden_gov_bonds": lambda: bonds,
        "fred_treasury_cmt": lambda: fred.fetch_treasury_cmt(on),
        "fred_treasury_cmt_history": lambda: fred.fetch_treasury_cmt_history(start, on),
        "usd_ois_swaps": lambda: fred.fetch_ois_swaps(on),
        "ecb_spot_curve": lambda: ecb.fetch_spot_curve(on),
        "ecb_svensson_parameters": lambda: ecb.fetch_svensson_parameters(on),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yieldcurve.market.refresh")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--root", type=str, default=str(DEFAULT_SNAPSHOT_ROOT))
    args = parser.parse_args(argv)

    snapshot = Snapshot(date=args.date, root=Path(args.root))
    builders = _builders(args.date)
    selected = args.only or list(builders)
    failures: list[str] = []
    for name in selected:
        try:
            snapshot.save(name, builders[name]())
        except Exception as exc:  # one bad source must not lose the rest
            failures.append(f"{name}: {exc}")
        else:
            print(f"wrote {name}")
    for failure in failures:
        print(f"FAILED {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
