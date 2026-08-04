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
    "fred_treasury_cmt_history": ["date", "tenor_years", "rate"],
    "usd_ois_swaps": ["tenor_years", "par_rate"],
    "usd_forecast_basis": ["tenor_years", "basis_bp"],
    "ecb_spot_curve": ["tenor_years", "zero_rate"],
    "ecb_svensson_parameters": ["parameter", "value"],
    "illustrative_swaption_vols": ["expiry", "maturity", "vol"],
}

# Per-column units, transcribed by hand from the manifest so the pin is
# independent of both the CSV headers and the manifest text it guards.
EXPECTED_UNITS = {
    "riksbank_bills": {
        "tenor": "tenor label (1M, 3M, 6M)",
        "maturity_date": "ISO 8601 date",
        "rate": "decimal yield (0.01933 = 1.933%)",
    },
    "riksbank_gov_benchmarks": {
        "tenor": "tenor label (2Y, 5Y, 7Y, 10Y)",
        "maturity_date": "ISO 8601 date",
        "yield": "decimal yield (0.02478 = 2.478%)",
    },
    "riksbank_swestr": {
        "tenor": "tenor label (ON, 1W, 1M, 2M, 3M, 6M)",
        "rate": "decimal rate (0.0164 = 1.64%)",
    },
    "riksgalden_gov_bonds": {
        "isin": "ISIN identifier",
        "coupon": "decimal coupon rate (0.01 = 1%)",
        "issue_date": "ISO 8601 date",
        "maturity_date": "ISO 8601 date",
        "outstanding_nominal": "outstanding nominal in SEK",
    },
    "fred_treasury_cmt": {
        "series_id": "FRED series identifier (DGS1MO..DGS30)",
        "tenor_years": "tenor in years",
        "rate": "decimal yield (0.038 = 3.8%)",
    },
    "fred_treasury_cmt_history": {
        "date": "ISO 8601 observation date",
        "tenor_years": "tenor in years",
        "rate": "decimal yield",
    },
    "usd_ois_swaps": {
        "tenor_years": "tenor in years",
        "par_rate": "decimal par rate (0.0412 = 4.12%)",
    },
    "usd_forecast_basis": {
        "tenor_years": "tenor in years",
        "basis_bp": "basis spread in basis points",
    },
    "ecb_spot_curve": {
        "tenor_years": "tenor in years",
        "zero_rate": "decimal continuously compounded zero rate (0.0264139995 = 2.64139995%)",
    },
    "ecb_svensson_parameters": {
        "parameter": "Svensson parameter name (BETA0..BETA3, TAU1, TAU2)",
        "value": "BETA in percent, TAU in years",
    },
    "illustrative_swaption_vols": {
        "expiry": "ISO 8601 option expiry date",
        "maturity": "ISO 8601 swap maturity date",
        "vol": "normal volatility in basis points",
    },
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


@pytest.mark.parametrize(("name", "units"), EXPECTED_UNITS.items())
def test_manifest_units_match_the_independent_pin(
    snapshot: Snapshot, name: str, units: dict[str, str]
) -> None:
    """The manifest's per-column units equal the independently transcribed pin."""
    assert snapshot.manifest["datasets"][name]["units"] == units


def test_expected_schema_pins_cover_every_manifest_dataset(snapshot: Snapshot) -> None:
    """No packaged dataset escapes the independent columns/units pins."""
    datasets = set(snapshot.manifest["datasets"])

    assert set(EXPECTED_COLUMNS) == datasets
    assert set(EXPECTED_UNITS) == datasets


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


# --- Label assertions (MKT-04, MKT-12: constructed/CMT wording) ------------


def test_fred_datasets_describe_treasury_inputs_as_cmt_par_yields(
    snapshot: Snapshot,
) -> None:
    """Treasury inputs are labelled CMT par yields, and any curve built from
    them is a CMT-implied approximation rather than an official bootstrap."""
    for name in ("fred_treasury_cmt", "fred_treasury_cmt_history"):
        block = snapshot.manifest["datasets"][name]
        assert "CMT par yield" in block["transformation"], name


def test_usd_constructed_datasets_disclose_they_are_not_observed_quotes(
    snapshot: Snapshot,
) -> None:
    """The USD OIS grid and forecast basis are constructed inputs, never
    presented as observed live quotes."""
    for name in ("usd_ois_swaps", "usd_forecast_basis"):
        block = snapshot.manifest["datasets"][name]
        assert block["classification"] == "constructed"
        assert "not observed" in block["limitations"], name
        assert "constructed" in block["licence"].lower(), name


def test_illustrative_vol_grid_is_labelled_constructed_not_observed(
    snapshot: Snapshot,
) -> None:
    block = snapshot.manifest["datasets"]["illustrative_swaption_vols"]

    assert block["classification"] == "illustrative"
    assert "not observed" in block["limitations"]
    assert "not market data" in block["limitations"]


def test_manifest_has_no_stale_cme_redistribution_language(snapshot: Snapshot) -> None:
    """The stale CME redistribution narrative was removed from the manifest
    along with the deleted cme.py module (HYGIENE-06)."""
    from importlib import resources

    resource = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")
    with resource.open("rb") as handle:
        raw = handle.read().decode("utf-8")

    assert "CME" not in raw
