"""Parametric curves, checked against the ECB's published Svensson parameters."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from curveengine.conventions import Compounding, to_continuous
from curveengine.curves.parametric import FitError, NelsonSiegel, Svensson
from curveengine.curves.protocol import DiscountCurve
from curveengine.market.snapshot import Snapshot

REFERENCE = date(2026, 7, 24)

# ECB spot rates are continuously compounded (ECB technical notes §3).
# Beta parameters are published as percentages; divide by 100 for decimal.
_ECB_COMPOUNDING = Compounding.CONTINUOUS

RECONSTRUCTION_TOLERANCE_BP = 0.5
FIT_TOLERANCE_BP = 1.0
FIT_RMSE_BP = 0.5


@pytest.fixture(scope="module")
def ecb_parameters() -> dict[str, float]:
    frame = Snapshot(date=REFERENCE).load("ecb_svensson_parameters")
    return dict(zip(frame["parameter"], frame["value"], strict=True))


@pytest.fixture(scope="module")
def ecb_spot() -> tuple[tuple[float, ...], tuple[float, ...]]:
    frame = Snapshot(date=REFERENCE).load("ecb_spot_curve").sort_values("tenor_years")
    times = tuple(float(t) for t in frame["tenor_years"])
    continuous = tuple(
        to_continuous(float(r), float(t), _ECB_COMPOUNDING)
        for t, r in zip(frame["tenor_years"], frame["zero_rate"], strict=True)
    )
    return times, continuous


def test_svensson_satisfies_the_discount_curve_protocol() -> None:
    curve = Svensson(beta=(0.03, -0.01, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)

    assert isinstance(curve, DiscountCurve)
    assert curve.df(5.0) == pytest.approx(math.exp(-curve.zero(5.0) * 5.0), rel=1e-12)


def test_svensson_short_end_limit_is_beta0_plus_beta1() -> None:
    curve = Svensson(beta=(0.032, -0.011, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)

    assert curve.zero(1e-8) == pytest.approx(0.032 - 0.011, abs=1e-6)


def test_svensson_long_end_limit_is_beta0() -> None:
    curve = Svensson(beta=(0.032, -0.011, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)

    assert curve.zero(300.0) == pytest.approx(0.032, abs=1e-4)


def test_nelson_siegel_is_svensson_with_the_fourth_term_switched_off() -> None:
    ns = NelsonSiegel(beta=(0.03, -0.01, 0.02), tau=2.0, reference_date=REFERENCE)
    nss = Svensson(beta=(0.03, -0.01, 0.02, 0.0), tau=(2.0, 5.0), reference_date=REFERENCE)

    for t in (0.5, 2.0, 7.0, 25.0):
        assert ns.zero(t) == pytest.approx(nss.zero(t), rel=1e-12)


def test_reconstruction_from_ecb_parameters_matches_the_ecb_spot_curve(
    ecb_parameters: dict[str, float],
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    curve = Svensson(
        beta=(
            ecb_parameters["BETA0"] / 100,
            ecb_parameters["BETA1"] / 100,
            ecb_parameters["BETA2"] / 100,
            ecb_parameters["BETA3"] / 100,
        ),
        tau=(ecb_parameters["TAU1"], ecb_parameters["TAU2"]),
        reference_date=REFERENCE,
    )
    times, published = ecb_spot

    errors_bp = [abs(curve.zero(t) - z) * 1e4 for t, z in zip(times, published, strict=True)]

    assert (
        max(errors_bp) < RECONSTRUCTION_TOLERANCE_BP
    ), f"Worst tenor off by {max(errors_bp):.2f}bp at t={times[errors_bp.index(max(errors_bp))]}"


def test_our_fit_reproduces_the_ecb_spot_curve(
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    times, published = ecb_spot

    fitted = Svensson.fit(times, published, reference_date=REFERENCE)

    errors_bp = [abs(fitted.zero(t) - z) * 1e4 for t, z in zip(times, published, strict=True)]
    assert max(errors_bp) < FIT_TOLERANCE_BP
    assert fitted.rmse(times, published) * 1e4 < FIT_RMSE_BP


def test_the_fit_is_deterministic(
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    times, published = ecb_spot

    first = Svensson.fit(times, published, reference_date=REFERENCE)
    second = Svensson.fit(times, published, reference_date=REFERENCE)

    assert first.beta == second.beta
    assert first.tau == second.tau


def test_the_fit_respects_the_parameter_constraints() -> None:
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)

    fitted = Svensson.fit(times, zeros, reference_date=REFERENCE)

    assert fitted.beta[0] > 0.0
    assert fitted.beta[0] + fitted.beta[1] > 0.0
    assert fitted.tau[0] > 0.0
    assert fitted.tau[1] > 0.0


def test_a_svensson_fit_needs_at_least_six_points() -> None:
    with pytest.raises(FitError, match="6 parameters"):
        Svensson.fit((1.0, 2.0, 5.0), (0.02, 0.025, 0.03), reference_date=REFERENCE)


def test_nelson_siegel_fits_a_curve_it_can_represent_almost_exactly() -> None:
    """Self-consistency: generate from known parameters, fit, recover the curve.
    NS is not identifiable — different parameter vectors give nearly identical
    curves — so the assertion is on the curve values, not the parameters."""
    truth = NelsonSiegel(beta=(0.033, -0.012, 0.018), tau=2.4, reference_date=REFERENCE)
    times: tuple[float, ...] = tuple(np.linspace(0.25, 30.0, 24))
    zeros = tuple(truth.zero(float(t)) for t in times)

    fitted = NelsonSiegel.fit(times, zeros, reference_date=REFERENCE)

    for t in times:
        assert fitted.zero(float(t)) == pytest.approx(truth.zero(float(t)), abs=1e-6)


def test_a_non_positive_tau_is_rejected_at_construction() -> None:
    with pytest.raises(FitError, match="positive"):
        Svensson(beta=(0.03, -0.01, 0.0, 0.0), tau=(0.0, 5.0), reference_date=REFERENCE)
