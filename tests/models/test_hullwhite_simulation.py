"""Exact Gaussian simulation of the Hull-White short rate."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from yieldcurve.curves.protocol import FlatCurve
from yieldcurve.models.hullwhite import HullWhite

ASOF = date(2026, 7, 24)
SEED = 20260727


@pytest.fixture
def model() -> HullWhite:
    return HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=0.05, sigma=0.01)


def test_simulation_starts_at_the_initial_short_rate(model: HullWhite) -> None:
    paths = model.simulate([0.0, 1.0], n_paths=100, seed=SEED)

    np.testing.assert_allclose(paths[:, 0], model.r0, atol=1e-12)


def test_simulation_shape_is_paths_by_times(model: HullWhite) -> None:
    paths = model.simulate([0.0, 0.5, 1.0, 2.0], n_paths=1000, seed=SEED)

    assert paths.shape == (1000, 4)


def test_terminal_distribution_matches_the_analytic_moments(model: HullWhite) -> None:
    paths = model.simulate([0.0, 5.0], n_paths=200_000, seed=SEED)
    terminal = paths[:, 1]

    expected_mean = model.conditional_mean(0.0, 5.0, model.r0)
    expected_sd = model.conditional_sd(0.0, 5.0)

    assert terminal.mean() == pytest.approx(expected_mean, abs=4 * expected_sd / math.sqrt(200_000))
    assert terminal.std(ddof=1) == pytest.approx(expected_sd, rel=0.01)


def test_the_scheme_is_step_size_independent(model: HullWhite) -> None:
    coarse = model.simulate([0.0, 5.0], n_paths=100_000, seed=SEED)[:, -1]
    fine = model.simulate(list(np.linspace(0.0, 5.0, 61)), n_paths=100_000, seed=SEED)[:, -1]

    assert coarse.mean() == pytest.approx(fine.mean(), abs=1e-4)
    assert coarse.std(ddof=1) == pytest.approx(fine.std(ddof=1), rel=0.02)


def test_conditional_sd_grows_and_saturates(model: HullWhite) -> None:
    sds = [model.conditional_sd(0.0, h) for h in (1.0, 5.0, 20.0, 100.0)]

    assert all(b > a for a, b in zip(sds, sds[1:], strict=False))  # noqa: RUF007
    assert sds[-1] == pytest.approx(model.sigma / math.sqrt(2 * model.a), rel=3e-5)


def test_zero_volatility_is_deterministic() -> None:
    curve = FlatCurve(reference_date=ASOF, rate=0.03)
    model = HullWhite(curve=curve, a=0.05, sigma=0.0)
    times = [0.0, 1.0, 5.0, 10.0]

    paths = model.simulate(times, n_paths=10, seed=SEED)

    for index, t in enumerate(times):
        np.testing.assert_allclose(paths[:, index], model.instantaneous_fwd(t), atol=1e-9)


def test_simulation_is_reproducible(model: HullWhite) -> None:
    first = model.simulate([0.0, 1.0, 5.0], n_paths=500, seed=SEED)
    second = model.simulate([0.0, 1.0, 5.0], n_paths=500, seed=SEED)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_paths(model: HullWhite) -> None:
    first = model.simulate([0.0, 5.0], n_paths=500, seed=SEED)
    second = model.simulate([0.0, 5.0], n_paths=500, seed=SEED + 1)

    assert not np.allclose(first, second)


@pytest.mark.parametrize("n_paths", [10_000, 40_000, 160_000])
def test_monte_carlo_bond_price_agrees_with_the_analytic_price_within_monte_carlo_error(
    model: HullWhite, n_paths: int
) -> None:
    # The monthly path-discount approximation carries an O(step^2) bias
    # (~2.9e-7 on this fixture), so the estimator converges to the trapezoid
    # expectation, not to the exact analytic bond price. The bias is ~1000x
    # below the 3-SE window here, so the estimate agrees with the analytic
    # price within Monte Carlo error; the bias itself is measured
    # deterministically in the time-step bias tests below.
    estimates = model.simulate_path_discount_factors(0.0, 5.0, n_paths=n_paths, seed=SEED)
    analytic = model.curve.df(5.0)

    standard_error = estimates.std(ddof=1) / math.sqrt(n_paths)

    assert abs(estimates.mean() - analytic) < 3 * standard_error


def test_monte_carlo_error_declines_at_one_over_root_n(model: HullWhite) -> None:
    errors = []
    for n_paths in (10_000, 40_000, 160_000):
        estimates = model.simulate_path_discount_factors(0.0, 5.0, n_paths=n_paths, seed=SEED)
        errors.append(estimates.std(ddof=1) / math.sqrt(n_paths))

    assert errors[1] == pytest.approx(errors[0] / 2.0, rel=0.15)
    assert errors[2] == pytest.approx(errors[1] / 2.0, rel=0.15)


# -- time-step bias of the monthly path-discount approximation -----------------------
#
# simulate_path_discount_factors returns exp(-trapezoid of the short-rate path),
# which is NOT the exact zero-coupon bond price: the trapezoid rule carries an
# O(step^2) bias. The closed forms below (OU covariance and conditional mean of
# the fitted model) give the exact expectation of exp(-trapezoid) for a Gaussian
# path, so the bias can be measured deterministically, apart from Monte Carlo
# error.


def _ou_covariance(model: HullWhite, s: float, u: float) -> float:
    """Cov(r(s), r(u)) for the OU short rate with the model's a and sigma."""
    a, sigma = model.a, model.sigma
    if a < 1e-8:
        return sigma**2 * min(s, u)
    return sigma**2 / (2.0 * a) * (math.exp(-a * abs(s - u)) - math.exp(-a * (s + u)))


def _expected_short_rate(model: HullWhite, t: float) -> float:
    """E[r(t)] = f(0, t) + sigma^2/(2 a^2) (1 - e^{-a t})^2 on the flat fixture curve."""
    a, sigma = model.a, model.sigma
    decay = -math.expm1(-a * t)
    return model.instantaneous_fwd(0.0) + sigma**2 / (2.0 * a**2) * decay**2


def _trapezoid_expectation(model: HullWhite, horizon: float, steps: int) -> float:
    """E[exp(-trapezoid integral of r)] on a ``steps`` grid, in closed form."""
    grid = [horizon * k / steps for k in range(steps + 1)]
    h = horizon / steps
    weights = np.full(steps + 1, h)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    mean = sum(w * _expected_short_rate(model, t) for w, t in zip(weights, grid, strict=True))
    variance = sum(
        weights[i] * weights[j] * _ou_covariance(model, grid[i], grid[j])
        for i in range(steps + 1)
        for j in range(steps + 1)
    )
    return math.exp(-mean + 0.5 * variance)


def test_path_discount_approximation_has_a_time_step_bias_that_shrinks(
    model: HullWhite,
) -> None:
    analytic = model.curve.df(5.0)

    monthly = _trapezoid_expectation(model, 5.0, 60)
    fine = _trapezoid_expectation(model, 5.0, 600)

    # The monthly quadrature is an approximation, not exact bond simulation...
    assert abs(monthly - analytic) > 0.0
    # ...and refining the grid shrinks the bias (600 steps is 100x finer).
    assert abs(fine - analytic) < abs(monthly - analytic)


def test_monte_carlo_estimates_land_on_the_bias_corrected_expectation(
    model: HullWhite,
) -> None:
    # Monte Carlo error and time-step bias are separate: with enough paths the
    # estimator converges to the closed-form trapezoid expectation, not to the
    # exact bond price.
    n_paths = 200_000
    estimates = model.simulate_path_discount_factors(0.0, 5.0, n_paths=n_paths, seed=SEED)
    expected = _trapezoid_expectation(model, 5.0, 60)

    standard_error = estimates.std(ddof=1) / math.sqrt(n_paths)

    assert abs(estimates.mean() - expected) < 4 * standard_error


def test_unsorted_times_are_rejected(model: HullWhite) -> None:
    with pytest.raises(ValueError, match="ascending"):
        model.simulate([0.0, 5.0, 1.0], n_paths=10, seed=SEED)
