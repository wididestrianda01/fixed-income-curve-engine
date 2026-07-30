"""Exact Gaussian simulation of the Hull-White short rate."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from curveengine.curves.protocol import FlatCurve
from curveengine.models.hullwhite import HullWhite

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
def test_monte_carlo_bond_price_converges_to_the_analytic_price(
    model: HullWhite, n_paths: int
) -> None:
    estimates = model.simulate_zcb(0.0, 5.0, n_paths=n_paths, seed=SEED)
    analytic = model.curve.df(5.0)

    standard_error = estimates.std(ddof=1) / math.sqrt(n_paths)

    assert abs(estimates.mean() - analytic) < 3 * standard_error


def test_monte_carlo_error_declines_at_one_over_root_n(model: HullWhite) -> None:
    errors = []
    for n_paths in (10_000, 40_000, 160_000):
        estimates = model.simulate_zcb(0.0, 5.0, n_paths=n_paths, seed=SEED)
        errors.append(estimates.std(ddof=1) / math.sqrt(n_paths))

    assert errors[1] == pytest.approx(errors[0] / 2.0, rel=0.15)
    assert errors[2] == pytest.approx(errors[1] / 2.0, rel=0.15)


def test_unsorted_times_are_rejected(model: HullWhite) -> None:
    with pytest.raises(ValueError, match="ascending"):
        model.simulate([0.0, 5.0, 1.0], n_paths=10, seed=SEED)
