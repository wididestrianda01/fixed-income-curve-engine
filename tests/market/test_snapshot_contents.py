"""The committed snapshot must satisfy the schema every later phase relies on."""

from __future__ import annotations

import hashlib
from importlib import resources

import pytest

from yieldcurve.market.snapshot import Snapshot

EXPECTED_COLUMNS = {
    "riksbank_bills": ["tenor", "maturity_date", "rate"],
    "riksbank_gov_benchmarks": ["tenor", "maturity_date", "yield"],
    "riksbank_swestr": ["tenor", "rate"],
    "riksgalden_gov_bonds": [
        "isin",
        "coupon",
        "issue_date",
        "maturity_date",
        "outstanding_nominal",
    ],
    "fred_treasury_cmt": ["series_id", "tenor_years", "rate"],
    "usd_ois_swaps": ["tenor_years", "par_rate"],
    "ecb_spot_curve": ["tenor_years", "zero_rate"],
    "ecb_svensson_parameters": ["parameter", "value"],
}

EXPECTED_CLASSIFICATION = {
    "riksbank_bills": "public",
    "riksbank_gov_benchmarks": "public",
    "riksbank_swestr": "public",
    "riksgalden_gov_bonds": "public",
    "fred_treasury_cmt": "public",
    "fred_treasury_cmt_history": "public",
    "usd_ois_swaps": "constructed",
    "usd_forecast_basis": "constructed",
    "ecb_spot_curve": "public",
    "ecb_svensson_parameters": "public",
    "illustrative_swaption_vols": "illustrative",
}


@pytest.mark.parametrize(("name", "columns"), EXPECTED_COLUMNS.items())
def test_committed_snapshot_has_the_expected_schema(
    snapshot: Snapshot, name: str, columns: list[str]
) -> None:
    frame = snapshot.load(name)

    assert list(frame.columns) == columns
    assert not frame.empty


def test_rates_are_decimals_not_percentages(snapshot: Snapshot) -> None:
    """A 2.31% bill quote must be stored as 0.0231. Getting this wrong by a
    factor of 100 is the single most common data-plumbing defect, and it stays
    invisible until a duration number looks merely implausible rather than wrong."""
    bills = snapshot.load("riksbank_bills")

    assert bills["rate"].abs().max() < 0.25


def test_svensson_parameters_are_complete(snapshot: Snapshot) -> None:
    params = snapshot.load("ecb_svensson_parameters").set_index("parameter")["value"]

    assert set(params.index) == {"BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"}
    assert params["TAU1"] > 0
    assert params["TAU2"] > 0


# --- Manifest and CSV contents agree (MKT-06, MKT-07, TQ-02) ----------------


def test_manifest_records_the_snapshot_date(snapshot: Snapshot) -> None:
    """The manifest's observation and retrieval dates equal the snapshot as-of."""
    manifest = snapshot.manifest

    assert manifest["snapshot_date"] == snapshot.date.isoformat()
    for block in manifest["datasets"].values():
        assert block["observation_date"] == manifest["snapshot_date"]
        assert block["retrieval_date"] == manifest["snapshot_date"]


def test_manifest_columns_and_units_agree_with_every_csv(snapshot: Snapshot) -> None:
    """Every packaged CSV has exactly the columns and per-column units the
    manifest records."""
    manifest = snapshot.manifest
    for name in snapshot.available():
        block = manifest["datasets"][name]
        frame = snapshot.load(name)

        assert list(frame.columns) == block["columns"]
        assert set(block["units"]) == set(block["columns"])
        assert all(block["units"][column] for column in block["columns"])


def test_manifest_checksums_match_the_packaged_csv_bytes(snapshot: Snapshot) -> None:
    """The manifest's sha256 for every dataset is the digest of the packaged file."""
    package = resources.files("yieldcurve.data")
    manifest = snapshot.manifest

    for name in snapshot.available():
        expected = hashlib.sha256(package.joinpath(f"{name}.csv").read_bytes()).hexdigest()
        assert manifest["datasets"][name]["sha256"] == expected


def test_manifest_covers_exactly_the_packaged_datasets(snapshot: Snapshot) -> None:
    """No packaged CSV is missing from the manifest and no manifest entry is
    orphaned: the manifest is the authoritative dataset list."""
    package = resources.files("yieldcurve.data")
    csv_files = {
        entry.name[: -len(".csv")] for entry in package.iterdir() if entry.name.endswith(".csv")
    }

    assert set(snapshot.manifest["datasets"]) == csv_files


def test_every_dataset_is_classified_public_constructed_or_illustrative(
    snapshot: Snapshot,
) -> None:
    """Each dataset carries the honest public/constructed/illustrative label."""
    manifest = snapshot.manifest
    for name, block in manifest["datasets"].items():
        assert block["classification"] in {"public", "constructed", "illustrative"}
        assert block["classification"] == EXPECTED_CLASSIFICATION[name]
