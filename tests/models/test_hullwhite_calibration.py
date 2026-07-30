"""Calibrating (a, sigma) to market swaption volatilities."""

from __future__ import annotations

from datetime import date

import pytest

from yieldcurve.curves.build import usd_ois_curve
from yieldcurve.curves.protocol import FlatCurve
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.models.hullwhite import (
    HullWhite,
    atm_swaption_grid,
    calibrate,
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
