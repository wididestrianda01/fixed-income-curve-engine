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

from yieldcurve.market import ecb, fred, riksbank, riksgalden
from yieldcurve.market.snapshot import Snapshot

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

PARSER_TESTS = [
    (
        riksbank,
        "_parse_bills",
        {"1M": 3.45, "3M": 3.52},
        {"1M": date(2026, 8, 31), "3M": date(2026, 11, 2)},
        ["tenor", "maturity_date", "rate"],
    ),
    (
        riksbank,
        "_parse_gov_benchmarks",
        {"2Y": 2.10, "5Y": 2.50},
        {"2Y": date(2028, 7, 30), "5Y": date(2031, 7, 30)},
        ["tenor", "maturity_date", "yield"],
    ),
    (
        riksbank,
        "_parse_swestr",
        {"ON": 3.25, "1W": 3.26},
        {},
        ["tenor", "rate"],
    ),
    (
        riksgalden,
        "_parse_gov_bonds",
        {},
        {},
        ["isin", "coupon", "issue_date", "maturity_date", "outstanding_nominal"],
    ),
    (
        fred,
        "_parse_treasury_cmt",
        {1.0: 4.50, 2.0: 4.40},
        {},
        ["series_id", "tenor_years", "rate"],
    ),
    (
        fred,
        "_parse_ois_swaps",
        {1.0: 4.30, 2.0: 4.20},
        {},
        ["tenor_years", "par_rate"],
    ),
    (
        ecb,
        "_parse_spot_curve",
        {1.0: 2.50, 2.0: 2.45},
        {},
        ["tenor_years", "zero_rate"],
    ),
    (
        ecb,
        "_parse_svensson_parameters",
        {"BETA0": 3.0, "BETA1": -1.0, "BETA2": 0.5, "BETA3": -0.2, "TAU1": 1.5, "TAU2": 15.0},
        {},
        ["parameter", "value"],
    ),
]


@pytest.mark.parametrize(("module", "parser_name", "dataset"), PARSERS)
def test_every_dataset_has_a_named_parser(module: object, parser_name: str, dataset: str) -> None:
    assert callable(getattr(module, parser_name))


@pytest.mark.parametrize(
    ("module", "parser_name", "raw", "maturities", "expected_columns"), PARSER_TESTS
)
def test_parser_produces_committed_columns(
    module: object,
    parser_name: str,
    raw: dict[str, float],
    maturities: dict[str, date],
    expected_columns: list[str],
) -> None:
    parser = getattr(module, parser_name)
    if parser_name == "_parse_gov_bonds":
        result = parser()
    elif parser_name in ("_parse_bills", "_parse_gov_benchmarks"):
        result = parser(raw, maturities)
    else:
        result = parser(raw)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_columns
    assert len(result) > 0


def test_importing_the_package_opens_no_socket() -> None:
    program = (
        "import socket;socket.socket = None;import yieldcurve;import yieldcurve.market;print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_refresh_is_importable_without_fetching() -> None:
    module = importlib.import_module("yieldcurve.market.refresh")

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
