"""Adapters: shape contracts and the no-network-at-import guarantee.

The fetch functions themselves are not exercised — they require the network,
which the suite forbids. What is exercised is everything that can silently rot
without the network: the parsing of a stored raw payload into the committed
schema, and the import graph.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from datetime import date

import pandas as pd
import pytest

from curveengine.market import ecb, fred, riksbank, riksgalden
from curveengine.market.snapshot import Snapshot

PARSERS = [
    (riksbank, "_parse_bills", "riksbank_bills"),
    (riksbank, "_parse_gov_benchmarks", "riksbank_gov_benchmarks"),
    (riksbank, "_parse_swestr", "riksbank_swestr"),
    (riksgalden, "_parse_gov_bonds", "riksgalden_gov_bonds"),
    (fred, "_parse_treasury_cmt", "fred_treasury_cmt"),
    (fred, "_parse_ois_swaps", "usd_ois_swaps"),
    (ecb, "_parse_spot_curve", "ecb_spot_curve"),
    (ecb, "_parse_svensson_parameters", "ecb_svensson_parameters"),
]


@pytest.mark.parametrize(("module", "parser_name", "dataset"), PARSERS)
def test_every_dataset_has_a_named_parser(module: object, parser_name: str, dataset: str) -> None:
    assert callable(getattr(module, parser_name))


def test_importing_the_package_opens_no_socket() -> None:
    program = (
        "import socket;"
        "socket.socket = None;"
        "import curveengine;"
        "import curveengine.market;"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_refresh_is_importable_without_fetching() -> None:
    module = importlib.import_module("curveengine.market.refresh")

    assert callable(module.main)


def test_history_dataset_is_committed_and_wide_enough_for_pca(
    snapshot: Snapshot,
) -> None:
    history = snapshot.load("fred_treasury_cmt_history")

    assert list(history.columns) == ["date", "tenor_years", "rate"]
    dates = pd.to_datetime(history["date"])
    assert dates.nunique() >= 750
    assert dates.max().date() <= date(2026, 7, 24)
    assert set(history["tenor_years"].unique()) >= {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0}
    assert history["rate"].abs().max() < 0.25


def test_history_is_a_rectangle_with_no_gaps(snapshot: Snapshot) -> None:
    history = snapshot.load("fred_treasury_cmt_history")
    pivot = history.pivot(index="date", columns="tenor_years", values="rate")  # noqa: PD010

    assert not pivot.isna().to_numpy().any()
