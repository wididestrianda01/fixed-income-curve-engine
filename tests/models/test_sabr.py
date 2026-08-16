"""SABR implied volatility and calibration, cross-checked against QuantLib."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldcurve.models.sabr import (
    CalibrationError,
    SabrError,
    calibrate,
    sabr_lognormal_vol,
    sabr_normal_vol,
)

# A representative smile: forward 3%, normal vol level 60 bp, negative skew,
# vol-of-vol 50%.
_F = 0.03
_ALPHA = 0.0060
_BETA = 0.0
_RHO = -0.30
_NU = 0.50
_T = 2.0


def test_beta_zero_atm_normal_vol_matches_the_closed_form() -> None:
    atm = _ALPHA * (1.0 + (2.0 - 3.0 * _RHO * _RHO) / 24.0 * _NU * _NU * _T)

    assert sabr_normal_vol(_F, _F, _ALPHA, 0.0, _RHO, _NU, _T) == pytest.approx(atm, rel=1e-12)


def test_zero_vol_of_vol_gives_a_flat_normal_smile() -> None:
    # nu = 0 -> z = 0 -> z / x(z) = 1, so the smile is constant at alpha.
    for strike in (0.02, 0.03, 0.04):
        assert sabr_normal_vol(_F, strike, _ALPHA, 0.0, _RHO, 0.0, _T) == pytest.approx(
            _ALPHA, rel=1e-12
        )


def test_negative_skew_raises_low_strike_volatility() -> None:
    low = sabr_normal_vol(_F, 0.02, _ALPHA, 0.0, _RHO, _NU, _T)
    high = sabr_normal_vol(_F, 0.04, _ALPHA, 0.0, _RHO, _NU, _T)

    assert low > high


def test_positive_skew_raises_high_strike_volatility() -> None:
    low = sabr_normal_vol(_F, 0.02, _ALPHA, 0.0, 0.30, _NU, _T)
    high = sabr_normal_vol(_F, 0.04, _ALPHA, 0.0, 0.30, _NU, _T)

    assert high > low


def test_vol_of_vol_adds_convexity() -> None:
    # With nu = 0 the smile is flat; nu > 0 makes it U-shaped (convex).
    flat = [sabr_normal_vol(_F, k, _ALPHA, 0.0, 0.0, 0.0, _T) for k in (0.02, 0.03, 0.04)]
    curved = [sabr_normal_vol(_F, k, _ALPHA, 0.0, 0.0, _NU, _T) for k in (0.02, 0.03, 0.04)]

    assert flat == pytest.approx([_ALPHA, _ALPHA, _ALPHA], rel=1e-12)
    assert curved[1] < (curved[0] + curved[2]) / 2.0


def test_non_positive_forward_or_strike_is_rejected() -> None:
    with pytest.raises(SabrError, match="forward"):
        sabr_normal_vol(0.0, 0.03, _ALPHA, 0.0, _RHO, _NU, _T)
    with pytest.raises(SabrError, match="strike"):
        sabr_normal_vol(0.03, -0.01, _ALPHA, 0.0, _RHO, _NU, _T)


def test_degenerate_parameters_are_rejected() -> None:
    with pytest.raises(SabrError, match="alpha"):
        sabr_normal_vol(_F, _F, 0.0, 0.0, _RHO, _NU, _T)
    with pytest.raises(SabrError, match="beta"):
        sabr_normal_vol(_F, _F, _ALPHA, 1.5, _RHO, _NU, _T)
    with pytest.raises(SabrError, match="rho"):
        sabr_normal_vol(_F, _F, _ALPHA, 0.0, 1.0, _NU, _T)
    with pytest.raises(SabrError, match="nu"):
        sabr_normal_vol(_F, _F, _ALPHA, 0.0, _RHO, -0.1, _T)


@given(
    forward=st.floats(min_value=0.005, max_value=0.08, allow_nan=False),
    strike=st.floats(min_value=0.005, max_value=0.08, allow_nan=False),
    alpha=st.floats(min_value=0.001, max_value=0.02, allow_nan=False),
    beta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    rho=st.floats(min_value=-0.95, max_value=0.95, allow_nan=False),
    nu=st.floats(min_value=0.0, max_value=1.5, allow_nan=False),
    expiry=st.floats(min_value=0.05, max_value=15.0, allow_nan=False),
)
def test_implied_vols_are_positive_and_finite(
    forward: float,
    strike: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    expiry: float,
) -> None:
    for fn in (sabr_normal_vol, sabr_lognormal_vol):
        v = fn(forward, strike, alpha, beta, rho, nu, expiry)
        assert math.isfinite(v) and v > 0.0


@pytest.mark.parametrize(
    ("beta", "alpha", "rho", "nu"),
    [
        (0.0, 0.0060, -0.30, 0.50),
        (0.5, 0.20, 0.10, 0.40),
        (1.0, 0.25, -0.45, 0.60),
        (0.0, 0.0040, 0.0, 0.0),
    ],
)
def test_quantlib_parity(beta: float, alpha: float, rho: float, nu: float) -> None:
    ql = pytest.importorskip("QuantLib")
    strikes = (0.020, 0.025, 0.030, 0.035, 0.040)
    forward = 0.03
    expiry = 2.0

    for k in strikes:
        assert sabr_normal_vol(forward, k, alpha, beta, rho, nu, expiry) == pytest.approx(
            ql.sabrVolatility(k, forward, expiry, alpha, beta, nu, rho, ql.Normal),
            rel=1e-10,
        )
        assert sabr_lognormal_vol(forward, k, alpha, beta, rho, nu, expiry) == pytest.approx(
            ql.sabrVolatility(k, forward, expiry, alpha, beta, nu, rho, ql.ShiftedLognormal),
            rel=1e-10,
        )


def test_calibration_recovers_planted_parameters() -> None:
    strikes = (0.020, 0.025, 0.030, 0.035, 0.040)
    market = [sabr_normal_vol(_F, k, _ALPHA, 0.0, _RHO, _NU, _T) for k in strikes]

    result = calibrate(_F, strikes, market, _T, beta=0.0)

    assert result.alpha == pytest.approx(_ALPHA, rel=1e-3)
    assert result.rho == pytest.approx(_RHO, abs=1e-3)
    assert result.nu == pytest.approx(_NU, rel=1e-3)
    assert result.n_strikes == len(strikes)
    assert result.active_bounds == (False, False, False)
    assert result.jacobian_rank == 3
    assert result.rmse_vol_bp < 1e-3


def test_calibration_reports_the_fit_residual_in_bp() -> None:
    strikes = (0.020, 0.030, 0.040)
    market = [0.0060, 0.0055, 0.0050]

    result = calibrate(_F, strikes, market, _T, beta=0.0)

    assert result.residual_scale * 1e4 == pytest.approx(result.rmse_vol_bp, rel=1e-12)


def test_calibration_rejects_too_few_strikes() -> None:
    with pytest.raises(CalibrationError, match="three"):
        calibrate(_F, (0.025, 0.035), (0.006, 0.005), _T, beta=0.0)


def test_calibration_rejects_a_flat_smile_as_under_identified() -> None:
    # A perfectly flat smile is reproducible by any alpha with nu = 0: alpha and
    # nu are not jointly identified, so the fit is start-sensitive and rejected.
    strikes = (0.020, 0.025, 0.030, 0.035, 0.040)
    market = [0.0060] * len(strikes)

    with pytest.raises(CalibrationError, match="sensitive"):
        calibrate(_F, strikes, market, _T, beta=0.0)


def test_calibration_rejects_a_non_positive_forward() -> None:
    with pytest.raises(CalibrationError, match="forward"):
        calibrate(0.0, (0.02, 0.03, 0.04), (0.006, 0.006, 0.006), _T)
