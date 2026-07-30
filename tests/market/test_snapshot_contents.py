"""The committed snapshot must satisfy the schema every later phase relies on."""

from __future__ import annotations

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
