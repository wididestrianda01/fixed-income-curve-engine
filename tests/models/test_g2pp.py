"""G2++ two-factor Gaussian model, cross-checked against QuantLib's ``ql.G2``."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldcurve.curves.protocol import FlatCurve
from yieldcurve.models.g2pp import (
    CalibrationError,
    G2pp,
    G2ppError,
    calibrate,
)

ASOF = date(2026, 7, 24)
RATE = 0.03
SEED = 20260816

# Representative parameters: two clearly separated mean-reversion speeds and
# volatilities, moderate positive correlation.
_A, _SIGMA, _B, _ETA, _RHO = 0.10, 0.010, 0.25, 0.020, 0.40


def _curve() -> FlatCurve:
    return FlatCurve(reference_date=ASOF, rate=RATE)


@pytest.fixture
def model() -> G2pp:
    return G2pp(curve=_curve(), a=_A, sigma=_SIGMA, b=_B, eta=_ETA, rho=_RHO)


def _ql_g2(ql: Any, a: float, sigma: float, b: float, eta: float, rho: float) -> Any:
    ql_date = ql.Date(ASOF.day, ASOF.month, ASOF.year)
    ql.Settings.instance().evaluationDate = ql_date
    handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_date, RATE, ql.Actual365Fixed(), ql.Continuous)
    )
    return ql.G2(handle, a, sigma, b, eta, rho)


def test_discount_bond_reproduces_the_curve_at_time_zero(model: G2pp) -> None:
    for t in (0.0, 1.0, 2.0, 5.0, 10.0):
        assert model.discount_bond(0.0, t, 0.0, 0.0) == pytest.approx(model.curve.df(t), rel=1e-14)


def test_discount_bond_falls_with_the_state(model: G2pp) -> None:
    # A higher short-rate state (larger x or y) lowers every bond price.
    base = model.discount_bond(1.0, 5.0, 0.0, 0.0)
    assert model.discount_bond(1.0, 5.0, 0.05, 0.0) < base
    assert model.discount_bond(1.0, 5.0, 0.0, 0.05) < base


def test_phi_zero_equals_the_instantaneous_forward(model: G2pp) -> None:
    assert model.phi(0.0) == pytest.approx(model.instantaneous_fwd(0.0), rel=1e-14)
    assert model.instantaneous_fwd(0.0) == pytest.approx(RATE, rel=1e-14)


def test_phi_matches_the_bond_factor_finite_difference(model: G2pp) -> None:
    # phi(t) = -d ln A(t,T)/dT at T = t; A(t, t) = 1, so the one-sided quotient
    # -ln A(t, t+h) / h converges to phi(t).
    t, h = 1.5, 1e-7
    fd = -math.log(model._bond_factor(t, t + h)) / h
    assert model.phi(t) == pytest.approx(fd, rel=1e-6)


def test_short_rate_is_x_plus_y_plus_phi(model: G2pp) -> None:
    t, x, y = 2.0, 0.01, -0.02
    assert model.short_rate(t, x, y) == pytest.approx(x + y + model.phi(t), rel=1e-14)


def test_state_covariance_is_positive_semidefinite(model: G2pp) -> None:
    cov = model.state_covariance(3.0)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0 and cov[1, 1] > 0.0
    # |rho| < 1 implies det > 0 and correlation strictly inside (-1, 1).
    assert cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2 > 0.0


def test_invalid_parameters_are_rejected() -> None:
    curve = _curve()
    for kwargs in (
        {"a": 0.0},
        {"sigma": -0.01},
        {"b": 0.0},
        {"eta": -0.01},
        {"rho": 1.0},
        {"rho": -1.0},
        {"a": float("nan")},
    ):
        with pytest.raises(G2ppError):
            G2pp(
                curve=curve,
                a=kwargs.get("a", _A),
                sigma=kwargs.get("sigma", _SIGMA),
                b=kwargs.get("b", _B),
                eta=kwargs.get("eta", _ETA),
                rho=kwargs.get("rho", _RHO),
            )


def test_discount_bond_rejects_bad_times_or_state(model: G2pp) -> None:
    with pytest.raises(ValueError, match="t"):
        model.discount_bond(-0.1, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="t"):
        model.discount_bond(2.0, 1.0, 0.0, 0.0)
    with pytest.raises(G2ppError, match="finite"):
        model.discount_bond(0.0, 1.0, float("nan"), 0.0)


def test_forward_rate_correlation_is_below_one_and_exact_on_the_diagonal(
    model: G2pp,
) -> None:
    assert model.forward_rate_correlation(1.0, 5.0, 5.0) == pytest.approx(1.0, rel=1e-14)
    for t1, t2 in ((1.0, 2.0), (1.0, 10.0), (5.0, 7.0)):
        corr = model.forward_rate_correlation(1.0, t1, t2)
        assert -1.0 < corr < 1.0


def test_forward_rate_correlation_degenerates_to_one_for_identical_factors() -> None:
    # With a == b and sigma == eta the two factors collapse into the single
    # process x + y, so every pair of forwards is perfectly correlated.
    curve = _curve()
    one_factor = G2pp(curve=curve, a=0.1, sigma=0.01, b=0.1, eta=0.01, rho=0.5)
    assert one_factor.forward_rate_correlation(1.0, 2.0, 10.0) == pytest.approx(1.0, rel=1e-12)


def test_forward_rate_correlation_matches_a_monte_carlo_estimate(model: G2pp) -> None:
    # f(t, T) is affine in (x, y) with loadings (e^{-a(T-t)}, e^{-b(T-t)}); the
    # deterministic part cancels in the correlation, so the correlation of the
    # random parts is the forward-rate correlation.
    t = 1.0
    paths = model.simulate([0.0, t], 200_000, seed=SEED)[:, -1, :]
    x, y = paths[:, 0], paths[:, 1]
    t1, t2 = 2.0, 7.0
    f1 = np.exp(-_A * (t1 - t)) * x + np.exp(-_B * (t1 - t)) * y
    f2 = np.exp(-_A * (t2 - t)) * x + np.exp(-_B * (t2 - t)) * y
    empirical = float(np.corrcoef(f1, f2)[0, 1])
    assert model.forward_rate_correlation(t, t1, t2) == pytest.approx(empirical, abs=1e-2)


def test_forward_rate_correlation_rejects_degenerate_inputs(model: G2pp) -> None:
    with pytest.raises(ValueError, match="undefined"):
        model.forward_rate_correlation(0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="t"):
        model.forward_rate_correlation(1.0, 0.5, 2.0)


def test_simulation_moments_match_the_analytic_covariance(model: G2pp) -> None:
    horizon = 3.0
    paths = model.simulate([0.0, horizon], 200_000, seed=SEED)[:, -1, :]
    empirical = np.cov(paths, rowvar=False)
    analytic = model.state_covariance(horizon)
    assert empirical[0, 0] == pytest.approx(analytic[0, 0], abs=2e-4)
    assert empirical[1, 1] == pytest.approx(analytic[1, 1], abs=2e-4)
    assert empirical[0, 1] == pytest.approx(analytic[0, 1], abs=2e-4)


def test_simulation_starts_at_the_origin(model: G2pp) -> None:
    paths = model.simulate([0.0, 1.0, 2.0], 5, seed=SEED)
    assert paths.shape == (5, 3, 2)
    assert np.all(paths[:, 0, :] == 0.0)


def test_simulate_short_rate_mean_equals_phi(model: G2pp) -> None:
    grid = [0.0, 1.0, 3.0, 5.0]
    paths = model.simulate_short_rate(grid, 200_000, seed=SEED)
    for i, t in enumerate(grid):
        assert paths[:, i].mean() == pytest.approx(model.phi(t), abs=1e-3)


def test_simulate_rejects_bad_grid_or_count(model: G2pp) -> None:
    with pytest.raises(ValueError, match="empty"):
        model.simulate([], 10, seed=SEED)
    with pytest.raises(ValueError, match="start at 0"):
        model.simulate([1.0, 2.0], 10, seed=SEED)
    with pytest.raises(ValueError, match="ascending"):
        model.simulate([0.0, 2.0, 1.0], 10, seed=SEED)
    with pytest.raises(ValueError, match="n_paths"):
        model.simulate([0.0, 1.0], 0, seed=SEED)


def test_bond_option_put_call_parity(model: G2pp) -> None:
    expiry, maturity, strike = 2.0, 5.0, 0.90
    call = model.bond_option(expiry, maturity, strike, call=True)
    put = model.bond_option(expiry, maturity, strike, call=False)
    assert call - put == pytest.approx(
        model.curve.df(maturity) - strike * model.curve.df(expiry), abs=1e-12
    )


def test_bond_option_at_maturity_is_intrinsic(model: G2pp) -> None:
    strike = 0.90
    assert model.bond_option(2.0, 2.0, strike, call=True) == pytest.approx(
        model.curve.df(2.0) * max(1.0 - strike, 0.0), rel=1e-14
    )
    assert model.bond_option(2.0, 2.0, strike, call=False) == pytest.approx(
        model.curve.df(2.0) * max(strike - 1.0, 0.0), rel=1e-14
    )


def test_bond_option_at_expiry_zero_is_intrinsic(model: G2pp) -> None:
    strike = 0.90
    assert model.bond_option(0.0, 5.0, strike, call=True) == pytest.approx(
        max(model.curve.df(5.0) - strike, 0.0), rel=1e-14
    )


def test_bond_option_rejects_bad_inputs(model: G2pp) -> None:
    with pytest.raises(G2ppError, match="strike"):
        model.bond_option(1.0, 2.0, 0.0, call=True)
    with pytest.raises(G2ppError, match="expiry"):
        model.bond_option(2.0, 1.0, 0.9, call=True)


def test_caplet_is_a_scaled_bond_put(model: G2pp) -> None:
    expiry, tenor, strike = 1.0, 0.5, 0.04
    tau = tenor
    bond_strike = 1.0 / (1.0 + tau * strike)
    assert model.caplet(expiry, tenor, strike) == pytest.approx(
        (1.0 + tau * strike) * model.bond_option(expiry, expiry + tenor, bond_strike, call=False),
        rel=1e-12,
    )


def test_caplet_normal_vol_is_positive(model: G2pp) -> None:
    forward = (model.curve.df(1.0) / model.curve.df(1.5) - 1.0) / 0.5
    vol = model.caplet_normal_vol(1.0, 0.5, forward)
    assert math.isfinite(vol) and vol > 0.0


@pytest.mark.parametrize(
    ("a", "sigma", "b", "eta", "rho"),
    [
        (0.10, 0.010, 0.25, 0.020, 0.40),
        (0.30, 0.020, 0.05, 0.008, -0.30),
        (0.05, 0.008, 0.50, 0.030, 0.10),
        (0.70, 0.025, 0.15, 0.012, 0.85),
    ],
)
def test_quantlib_discount_bond_parity(
    a: float, sigma: float, b: float, eta: float, rho: float
) -> None:
    ql = pytest.importorskip("QuantLib")
    theirs = _ql_g2(ql, a, sigma, b, eta, rho)
    model = G2pp(curve=_curve(), a=a, sigma=sigma, b=b, eta=eta, rho=rho)
    for t, t_maturity, x, y in (
        (0.0, 2.0, 0.0, 0.0),
        (1.0, 2.0, 0.01, -0.02),
        (0.5, 5.0, 0.03, 0.01),
        (0.0, 10.0, 0.0, 0.0),
        (2.0, 2.5, 0.0, 0.0),
    ):
        assert model.discount_bond(t, t_maturity, x, y) == pytest.approx(
            theirs.discountBond(t, t_maturity, ql.Array([x, y])),
            rel=1e-10,
        )


@pytest.mark.parametrize(
    ("a", "sigma", "b", "eta", "rho"),
    [
        (0.10, 0.010, 0.25, 0.020, 0.40),
        (0.30, 0.020, 0.05, 0.008, -0.30),
    ],
)
def test_quantlib_bond_option_parity(
    a: float, sigma: float, b: float, eta: float, rho: float
) -> None:
    ql = pytest.importorskip("QuantLib")
    theirs = _ql_g2(ql, a, sigma, b, eta, rho)
    model = G2pp(curve=_curve(), a=a, sigma=sigma, b=b, eta=eta, rho=rho)
    for expiry, maturity, strike, call in (
        (1.0, 2.0, 0.95, True),
        (1.0, 2.0, 0.95, False),
        (2.0, 5.0, 0.80, True),
        (0.5, 1.0, 0.97, False),
    ):
        typ = ql.Option.Call if call else ql.Option.Put
        assert model.bond_option(expiry, maturity, strike, call=call) == pytest.approx(
            theirs.discountBondOption(typ, strike, expiry, maturity),
            rel=1e-10,
        )


@given(
    a=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    sigma=st.floats(min_value=0.001, max_value=0.05, allow_nan=False),
    b=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    eta=st.floats(min_value=0.001, max_value=0.05, allow_nan=False),
    rho=st.floats(min_value=-0.95, max_value=0.95, allow_nan=False),
    t=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    maturity=st.floats(min_value=0.0, max_value=20.0, allow_nan=False),
    x=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
    y=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
)
def test_discount_bond_is_a_positive_discount(
    a: float,
    sigma: float,
    b: float,
    eta: float,
    rho: float,
    t: float,
    maturity: float,
    x: float,
    y: float,
) -> None:
    t_max = max(t, maturity)
    t_min = min(t, maturity)
    model = G2pp(curve=_curve(), a=a, sigma=sigma, b=b, eta=eta, rho=rho)
    price = model.discount_bond(t_min, t_max, x, y)
    assert math.isfinite(price) and price > 0.0


def test_phi_and_covariance_reject_negative_time(model: G2pp) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        model.instantaneous_fwd(-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        model.phi(-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        model.state_covariance(-0.1)


def test_bond_option_and_caplet_reject_non_finite(model: G2pp) -> None:
    with pytest.raises(G2ppError, match="finite"):
        model.bond_option(float("nan"), 2.0, 0.9, call=True)
    with pytest.raises(G2ppError, match="finite"):
        model.caplet(float("nan"), 0.5, 0.04)
    with pytest.raises(G2ppError, match="tenor"):
        model.caplet(1.0, 0.0, 0.04)


# --- calibration -----------------------------------------------------------

_CAPLET_GRID = (
    (0.5, 0.25),
    (0.5, 1.0),
    (1.0, 0.25),
    (1.0, 1.0),
    (2.0, 0.25),
    (2.0, 1.0),
    (3.0, 0.5),
    (5.0, 1.0),
    (7.0, 1.0),
    (10.0, 2.0),
)

_PLANTED = (0.08, 0.008, 0.30, 0.020, 0.40)


def _planted_quotes() -> list[tuple[float, float, float, float]]:
    curve = _curve()
    a, sigma, b, eta, rho = _PLANTED
    model = G2pp(curve=curve, a=a, sigma=sigma, b=b, eta=eta, rho=rho)
    quotes: list[tuple[float, float, float, float]] = []
    for expiry, tenor in _CAPLET_GRID:
        forward = (curve.df(expiry) / curve.df(expiry + tenor) - 1.0) / tenor
        quotes.append((expiry, tenor, forward, model.caplet_normal_vol(expiry, tenor, forward)))
    return quotes


def test_calibration_rejects_invalid_quote_geometry() -> None:
    quotes = _planted_quotes()
    quotes[0] = (0.0, 0.0, 0.04, 0.006)
    with pytest.raises(CalibrationError, match="invalid"):
        calibrate(_curve(), quotes, rho=_PLANTED[4])


def test_calibration_rejects_an_unordered_initial_guess() -> None:
    with pytest.raises(CalibrationError, match="convention"):
        calibrate(_curve(), _planted_quotes(), rho=_PLANTED[4], initial=(0.50, 0.01, 0.30, 0.02))


def test_calibration_recovers_planted_parameters() -> None:
    curve = _curve()
    a, sigma, b, eta, rho = _PLANTED
    result = calibrate(curve, _planted_quotes(), rho=rho)

    assert result.a == pytest.approx(a, rel=1e-3)
    assert result.sigma == pytest.approx(sigma, rel=1e-3)
    assert result.b == pytest.approx(b, rel=1e-3)
    assert result.eta == pytest.approx(eta, rel=1e-3)
    assert result.rho == rho
    assert result.n_instruments == len(_CAPLET_GRID)
    assert result.active_bounds == (False, False, False, False)
    assert result.jacobian_rank == 4
    assert result.rmse_vol_bp < 1e-3


def test_calibration_reports_the_fit_residual_in_bp() -> None:
    result = calibrate(_curve(), _planted_quotes(), rho=_PLANTED[4])
    assert result.residual_scale * 1e4 == pytest.approx(result.rmse_vol_bp, rel=1e-12)


def test_calibration_rejects_too_few_quotes() -> None:
    quotes = _planted_quotes()[:3]
    with pytest.raises(CalibrationError, match="four"):
        calibrate(_curve(), quotes, rho=_PLANTED[4])


def test_calibration_rejects_a_non_finite_or_negative_vol() -> None:
    quotes = _planted_quotes()
    bad_finite = [(e, t, k, float("nan")) for e, t, k, _ in quotes]
    with pytest.raises(CalibrationError, match="finite"):
        calibrate(_curve(), bad_finite, rho=_PLANTED[4])
    bad_negative = [(e, t, k, -0.001) for e, t, k, _ in quotes]
    with pytest.raises(CalibrationError, match="negative"):
        calibrate(_curve(), bad_negative, rho=_PLANTED[4])


def test_calibration_rejects_bad_rho_or_initial() -> None:
    quotes = _planted_quotes()
    with pytest.raises(CalibrationError, match="rho"):
        calibrate(_curve(), quotes, rho=1.0)
    with pytest.raises(CalibrationError, match="finite"):
        calibrate(_curve(), quotes, rho=_PLANTED[4], initial=(0.5, 0.01, float("nan"), 0.02))
    with pytest.raises(CalibrationError, match="bounds"):
        calibrate(_curve(), quotes, rho=_PLANTED[4], initial=(0.5, 0.01, 3.0, 0.02))


def test_calibration_rejects_rank_deficient_quotes() -> None:
    # Four identical quotes identify a single caplet price, not a 4-parameter
    # surface, so the Jacobian is rank-deficient.
    expiry, tenor, strike, vol = _planted_quotes()[0]
    quotes = [(expiry, tenor, strike, vol)] * 4
    with pytest.raises(CalibrationError, match="rank"):
        calibrate(_curve(), quotes, rho=_PLANTED[4])
