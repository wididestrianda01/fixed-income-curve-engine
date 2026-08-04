"""The frozen snapshot is the only market-data path: provenance, and the
no-network-at-import guarantee.

Task 11 removed the broken ECB, Riksbank, Riksgalden, and FRED HTTP adapters
and the ``yieldcurve.market.refresh`` CLI. The packaged frozen snapshot
(Task 10) is the only supported data path, so this file pins what must stay
true: every packaged dataset carries a provenance record in the manifest, the
adapters and refresh CLI no longer exist, and importing the market package
performs no network I/O. The fetch functions themselves are not exercised —
they are gone; the committed snapshot is the contract.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import date

import pandas as pd

from yieldcurve.market.snapshot import Snapshot

REMOVED_MODULES = ("ecb", "fred", "riksbank", "riksgalden", "refresh")
PROVENANCE_FIELDS = (
    "observation_date",
    "retrieval_date",
    "publisher",
    "primary_url",
    "classification",
    "licence",
    "transformation",
)


def test_adapter_modules_and_refresh_cli_no_longer_exist() -> None:
    """The broken HTTP adapters and the refresh CLI were removed (Task 11)."""
    for name in REMOVED_MODULES:
        spec = importlib.util.find_spec(f"yieldcurve.market.{name}")
        assert spec is None, f"yieldcurve.market.{name} must not exist"


def test_importing_the_market_package_opens_no_socket() -> None:
    """Importing the market package performs no network calls at socket level."""
    program = (
        "import socket;socket.socket = None;"
        "import yieldcurve.market;"
        "from yieldcurve.market import Snapshot;"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_every_dataset_records_provenance_in_the_manifest() -> None:
    """Section 6 contract: source, transformation, licence, classification, and
    observation/retrieval dates are recorded for every packaged dataset."""
    snapshot = Snapshot(date=date(2026, 7, 24))
    datasets = snapshot.manifest["datasets"]

    assert set(datasets) == set(snapshot.available())
    for name, block in datasets.items():
        missing = [field for field in PROVENANCE_FIELDS if not block.get(field)]
        assert not missing, f"dataset {name!r} lacks provenance fields: {missing}"


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
