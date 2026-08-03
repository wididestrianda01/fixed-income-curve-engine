"""Calibrating (a, sigma) to market swaption volatilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from yieldcurve.curves.build import usd_ois_curve
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.market.snapshot import MissingDatasetError, Snapshot
from yieldcurve.models.hullwhite import (
    HullWhite,
    atm_swaption_grid,
    calibrate,
    swaption_grid,
)

ASOF = date(2026, 7, 24)


def test_calibration_recovers_planted_parameters() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    truth = HullWhite(curve=curve, a=0.07, sigma=0.011)
    swaptions, _ = _synthetic_grid()
    vols = tuple(truth.swaption_normal_vol(s, ASOF) for s in swaptions)

    result = calibrate(curve, swaptions, vols, ASOF)

    assert result.a == pytest.approx(0.07, rel=1e-3)
    assert result.sigma == pytest.approx(0.011, rel=1e-3)
    assert result.rmse_vol_bp < 0.01


def test_calibration_is_deterministic() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    swaptions, vols = _synthetic_grid()

    first = calibrate(curve, swaptions, vols, ASOF)
    second = calibrate(curve, swaptions, vols, ASOF)

    assert (first.a, first.sigma) == (second.a, second.sigma)


def test_calibration_result_is_frozen() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    swaptions, vols = _synthetic_grid()

    result = calibrate(curve, swaptions, vols, ASOF)

    with pytest.raises(AttributeError):
        result.a = 0.99  # type: ignore[misc]


def test_mismatched_input_lengths_are_rejected() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    swaptions, vols = _synthetic_grid()

    with pytest.raises(ValueError, match="same length"):
        calibrate(curve, swaptions, vols[:-1], ASOF)


def test_rmse_is_reported_in_volatility_basis_points() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    swaptions, vols = _synthetic_grid()

    result = calibrate(curve, swaptions, vols, ASOF)

    assert 0.0 <= result.rmse_vol_bp < 100.0


@pytest.mark.skipif(
    not Snapshot(date=ASOF).directory.joinpath("cme_swaption_vols.csv").exists(),
    reason="cme_swaption_vols not in the committed snapshot",
)
def test_calibration_to_the_market_grid_fits_within_ten_vol_basis_points(
    snapshot: Snapshot,
) -> None:
    curve = usd_ois_curve(snapshot, ASOF)
    swaptions, vols = atm_swaption_grid(snapshot, ASOF, curve)

    result = calibrate(curve, swaptions, vols, ASOF)

    assert result.n_instruments >= 4
    assert result.rmse_vol_bp < 10.0
    assert 0.001 < result.a < 1.0
    assert 0.0005 < result.sigma < 0.05


def _synthetic_grid() -> tuple[tuple, tuple]:  # type: ignore[type-arg]
    from yieldcurve.calendars import USGovernmentBondCalendar
    from yieldcurve.conventions import BusinessDayConvention, DayCount
    from yieldcurve.instruments import Swaption, VanillaSwap

    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    truth = HullWhite(curve=curve, a=0.07, sigma=0.011)

    strike = 0.03

    expiry_maturity = (
        (date(2027, 7, 24), date(2030, 7, 24)),
        (date(2028, 7, 24), date(2036, 7, 24)),
        (date(2030, 7, 24), date(2033, 7, 24)),
        (date(2031, 7, 24), date(2036, 7, 24)),
        (date(2031, 7, 24), date(2051, 7, 24)),
        (date(2033, 7, 24), date(2036, 7, 24)),
        (date(2035, 7, 24), date(2056, 7, 24)),
    )

    swaptions = tuple(
        Swaption(
            expiry=expiry,
            swap=VanillaSwap(
                start=expiry,
                maturity=maturity,
                fixed_rate=strike,
                fixed_frequency=2,
                fixed_day_count=DayCount.THIRTY_360_BOND,
                float_tenor="3M",
                float_day_count=DayCount.ACT_360,
                calendar=USGovernmentBondCalendar(),
                bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
                notional=1.0,
            ),
            strike=strike,
            pay_fixed=True,
        )
        for expiry, maturity in expiry_maturity
    )
    vols = tuple(truth.swaption_normal_vol(s, ASOF) for s in swaptions)

    return swaptions, vols


# Tests for atm_swaption_grid()


@pytest.fixture
def flat_discount_curve() -> FlatCurve:
    """Fixture providing a flat discount curve for testing."""
    return FlatCurve(reference_date=ASOF, rate=0.03)


@pytest.fixture
def snapshot_with_swaption_data(
    tmp_path: Path,
) -> Snapshot:
    """Fixture providing a temporary snapshot with test swaption data."""
    import pandas as pd

    # Create snapshot directory
    snapshot_dir = tmp_path / ASOF.isoformat()
    snapshot_dir.mkdir(parents=True)

    # Create test data
    data = {
        "expiry": ["2027-07-24", "2028-07-24", "2030-07-24"],
        "maturity": ["2030-07-24", "2036-07-24", "2033-07-24"],
        "vol": ["35.0", "32.0", "28.0"],  # basis points
    }
    df = pd.DataFrame(data)
    df.to_csv(snapshot_dir / "cme_swaption_vols.csv", index=False)

    # Create and return snapshot with custom root
    return Snapshot(date=ASOF, root=tmp_path)


@pytest.fixture
def snapshot_with_single_swaption(tmp_path: Path) -> Snapshot:
    """Fixture providing a snapshot with single swaption."""
    import pandas as pd

    snapshot_dir = tmp_path / ASOF.isoformat()
    snapshot_dir.mkdir(parents=True)

    data = {
        "expiry": ["2027-07-24"],
        "maturity": ["2030-07-24"],
        "vol": ["35.0"],
    }
    df = pd.DataFrame(data)
    df.to_csv(snapshot_dir / "cme_swaption_vols.csv", index=False)

    return Snapshot(date=ASOF, root=tmp_path)


@pytest.fixture
def snapshot_with_empty_data(tmp_path: Path) -> Snapshot:
    """Fixture providing a snapshot with empty swaption data."""
    import pandas as pd

    snapshot_dir = tmp_path / ASOF.isoformat()
    snapshot_dir.mkdir(parents=True)

    data: dict[str, list[str]] = {
        "expiry": [],
        "maturity": [],
        "vol": [],
    }
    df = pd.DataFrame(data)
    df.to_csv(snapshot_dir / "cme_swaption_vols.csv", index=False)

    return Snapshot(date=ASOF, root=tmp_path)


def test_atm_swaption_grid_creates_correct_number_of_instruments(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Grid returns one swaption per row in snapshot data."""
    swaptions, vols = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify counts
    assert len(swaptions) == 3
    assert len(vols) == 3


def test_atm_swaption_grid_converts_volatilities_from_basis_points(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Volatilities in basis points are converted to decimal form."""
    _, vols = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify conversion (bp / 1e4)
    assert vols[0] == pytest.approx(0.0035, abs=1e-10)  # 35 bp
    assert vols[1] == pytest.approx(0.0032, abs=1e-10)  # 32 bp
    assert vols[2] == pytest.approx(0.0028, abs=1e-10)  # 28 bp


def test_atm_swaption_grid_sets_expiry_correctly(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Swaption expiry dates match snapshot data."""
    swaptions, _ = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify expiries round-trip
    assert swaptions[0].expiry == date(2027, 7, 24)
    assert swaptions[1].expiry == date(2028, 7, 24)
    assert swaptions[2].expiry == date(2030, 7, 24)


def test_atm_swaption_grid_sets_maturity_correctly(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Swaption swap maturity dates match snapshot data."""
    swaptions, _ = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify maturities round-trip
    assert swaptions[0].swap.maturity == date(2030, 7, 24)
    assert swaptions[1].swap.maturity == date(2036, 7, 24)
    assert swaptions[2].swap.maturity == date(2033, 7, 24)


def test_atm_swaption_grid_enforces_atm_property(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Each swaption's strike equals its swap's par rate (ATM condition)."""
    from yieldcurve.curves.pricing import par_rate

    swaptions, _ = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify ATM property
    curves = CurveSet.single(flat_discount_curve)
    for swaption in swaptions:
        strike = swaption.strike
        computed_par_rate = par_rate(swaption.swap, curves, ASOF)
        assert strike == pytest.approx(computed_par_rate, abs=1e-10)


def test_atm_swaption_grid_with_single_swaption(
    snapshot_with_single_swaption: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Grid handles minimal input (single swaption)."""
    swaptions, vols = atm_swaption_grid(snapshot_with_single_swaption, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify single row
    assert len(swaptions) == 1
    assert len(vols) == 1
    assert vols[0] == pytest.approx(0.0035, abs=1e-10)


def test_atm_swaption_grid_preserves_pay_fixed_convention(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """All swaptions are payer type (pay_fixed=True)."""
    swaptions, _ = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify payer convention
    for swaption in swaptions:
        assert swaption.pay_fixed is True


def test_atm_swaption_grid_uses_correct_swap_conventions(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Swaps in the grid use correct market conventions."""
    from yieldcurve.conventions import DayCount

    swaptions, _ = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify conventions
    for swaption in swaptions:
        swap = swaption.swap
        assert swap.fixed_frequency == 2
        assert swap.fixed_day_count == DayCount.THIRTY_360_BOND
        assert swap.float_tenor == "3M"
        assert swap.float_day_count == DayCount.ACT_360
        assert swap.notional == pytest.approx(1.0, abs=1e-10)


def test_atm_swaption_grid_returns_immutable_tuples(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Return values are immutable tuples, not lists."""
    swaptions, vols = atm_swaption_grid(snapshot_with_swaption_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify types
    assert isinstance(swaptions, tuple)
    assert isinstance(vols, tuple)


def test_atm_swaption_grid_with_empty_data(
    snapshot_with_empty_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """Grid handles empty snapshot data gracefully."""
    swaptions, vols = atm_swaption_grid(snapshot_with_empty_data, ASOF, flat_discount_curve)

    # Arrange-Act-Assert: verify empty result
    assert len(swaptions) == 0
    assert len(vols) == 0
    assert isinstance(swaptions, tuple)
    assert isinstance(vols, tuple)


def test_swaption_grid_builds_atm_swaptions_from_explicit_rows(
    flat_discount_curve: FlatCurve,
) -> None:
    """Grid construction is reusable without a licensed vendor CSV."""
    # Arrange
    rows = ((date(2027, 7, 24), date(2032, 7, 24), 85.0),)

    # Act
    swaptions, vols = swaption_grid(rows, ASOF, flat_discount_curve)

    # Assert
    assert len(swaptions) == 1
    assert vols == pytest.approx((85.0 / 1e4,))
    assert swaptions[0].strike == pytest.approx(swaptions[0].swap.fixed_rate)
    assert swaptions[0].expiry == date(2027, 7, 24)


def test_atm_swaption_grid_reads_an_alternative_dataset(snapshot: Snapshot) -> None:
    curve = usd_ois_curve(snapshot, ASOF)

    swaptions, vols = atm_swaption_grid(snapshot, ASOF, curve, dataset="illustrative_swaption_vols")

    assert len(swaptions) == len(vols)
    assert len(swaptions) > 0
    assert all(v > 0.0 for v in vols)


def test_atm_swaption_grid_still_defaults_to_the_cme_dataset(snapshot: Snapshot) -> None:
    curve = usd_ois_curve(snapshot, ASOF)

    with pytest.raises(MissingDatasetError, match="cme_swaption_vols"):
        atm_swaption_grid(snapshot, ASOF, curve)


def test_swaption_grid_and_the_snapshot_loader_agree(
    snapshot_with_swaption_data: Snapshot,
    flat_discount_curve: FlatCurve,
) -> None:
    """atm_swaption_grid is the CSV-reading wrapper around swaption_grid."""
    # Arrange
    data = snapshot_with_swaption_data.load("cme_swaption_vols")
    rows = tuple(
        (date.fromisoformat(r["expiry"]), date.fromisoformat(r["maturity"]), float(r["vol"]))
        for _, r in data.iterrows()
    )

    # Act
    csv_swaptions, csv_vols = atm_swaption_grid(
        snapshot_with_swaption_data, ASOF, flat_discount_curve
    )
    row_swaptions, row_vols = swaption_grid(rows, ASOF, flat_discount_curve)

    # Assert
    assert csv_vols == pytest.approx(row_vols)
    assert [s.expiry for s in csv_swaptions] == [s.expiry for s in row_swaptions]
    assert [s.strike for s in csv_swaptions] == pytest.approx([s.strike for s in row_swaptions])
