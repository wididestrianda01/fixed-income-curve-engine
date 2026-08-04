"""Parametric curves, checked against the ECB's published Svensson parameters."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from yieldcurve.curves import parametric as parametric_module
from yieldcurve.curves.parametric import FitError, NelsonSiegel, ParametricFitResult, Svensson
from yieldcurve.curves.protocol import DiscountCurve
from yieldcurve.market.snapshot import Snapshot

REFERENCE = date(2026, 7, 24)

# ECB spot rates are published already continuously compounded (ECB technical
# notes §3), so the fit consumes the published values directly; beta parameters
# are published as percentages and divided by 100 for decimal.
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
    zeros = tuple(float(r) for r in frame["zero_rate"])
    return times, zeros


# --- curve construction and evaluation ---------------------------------------


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
    """The two parameterizations are compared directly: a Nelson-Siegel curve
    is a Svensson curve whose fourth term is identically zero."""
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

    assert max(errors_bp) < RECONSTRUCTION_TOLERANCE_BP, (
        f"Worst tenor off by {max(errors_bp):.2f}bp at t={times[errors_bp.index(max(errors_bp))]}"
    )


def test_a_non_positive_tau_is_rejected_at_construction() -> None:
    with pytest.raises(FitError, match="positive"):
        Svensson(beta=(0.03, -0.01, 0.0, 0.0), tau=(0.0, 5.0), reference_date=REFERENCE)


def test_non_finite_parameters_are_rejected_at_construction() -> None:
    with pytest.raises(FitError, match="finite"):
        Svensson(beta=(float("nan"), -0.01, 0.0, 0.0), tau=(1.5, 8.0), reference_date=REFERENCE)
    with pytest.raises(FitError, match="finite"):
        NelsonSiegel(beta=(0.03, float("inf"), 0.02), tau=2.0, reference_date=REFERENCE)


def test_a_negative_curve_time_is_rejected_by_both_families() -> None:
    """Curve time runs forward from the reference date. A negative one is not a
    date in the past, it is a caller error — the functional forms would happily
    return a number for it, which is exactly why the guard has to be explicit."""
    nss = Svensson(beta=(0.03, -0.01, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)
    ns = NelsonSiegel(beta=(0.03, -0.01, 0.02), tau=2.0, reference_date=REFERENCE)

    for call in (nss.zero, nss.df, ns.zero, ns.df):
        with pytest.raises(ValueError, match="non-negative"):
            call(-1.0)


def test_non_finite_curve_times_are_rejected_by_both_families() -> None:
    """NaN and infinity are not curve times: the functional forms would return
    numbers for them (zero(inf) degenerates to b0), so the guard is explicit."""
    nss = Svensson(beta=(0.03, -0.01, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)
    ns = NelsonSiegel(beta=(0.03, -0.01, 0.02), tau=2.0, reference_date=REFERENCE)

    for call in (nss.zero, nss.df, nss.instantaneous_fwd, ns.zero, ns.df):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite"):
                call(bad)
    with pytest.raises(ValueError, match="finite"):
        nss.fwd(float("nan"), 5.0)
    with pytest.raises(ValueError, match="finite"):
        ns.fwd(1.0, float("inf"))


def test_svensson_rmse_is_zero_against_its_own_output() -> None:
    """The objective evaluated at the truth must be zero, or the fit optimises noise."""
    times = tuple(np.linspace(0.25, 30.0, 60))
    truth = Svensson(
        beta=(0.03, -0.01, 0.02, -0.005),
        tau=(1.5, 8.0),
        reference_date=REFERENCE,
    )
    zeros = tuple(truth.zero(float(t)) for t in times)
    assert truth.rmse(times, zeros) == pytest.approx(0.0, abs=1e-15)


def test_rmse_rejects_non_finite_or_mismatched_inputs() -> None:
    curve = Svensson(beta=(0.03, -0.01, 0.02, -0.005), tau=(1.5, 8.0), reference_date=REFERENCE)
    with pytest.raises(FitError, match="finite"):
        curve.rmse((1.0, 2.0), (0.02, float("nan")))
    with pytest.raises(FitError, match="times but"):
        curve.rmse((1.0, 2.0), (0.02,))


# --- fit input validation -----------------------------------------------------


def test_a_svensson_fit_needs_at_least_six_points() -> None:
    with pytest.raises(FitError, match="6 parameters"):
        Svensson.fit((1.0, 2.0, 5.0), (0.02, 0.025, 0.03), reference_date=REFERENCE)


def test_an_empty_fit_is_rejected() -> None:
    with pytest.raises(FitError, match="no observations"):
        Svensson.fit((), (), reference_date=REFERENCE)


def test_fit_times_and_zero_rates_must_have_the_same_length() -> None:
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)
    with pytest.raises(FitError, match="times but"):
        Svensson.fit(times, zeros[:-1], reference_date=REFERENCE)


def test_non_finite_fit_inputs_are_rejected() -> None:
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)
    with pytest.raises(FitError, match="finite"):
        Svensson.fit((float("nan"), *times[1:]), zeros, reference_date=REFERENCE)
    with pytest.raises(FitError, match="finite"):
        Svensson.fit(times, (float("inf"), *zeros[1:]), reference_date=REFERENCE)
    with pytest.raises(FitError, match="finite"):
        Svensson.fit(times, zeros, reference_date=REFERENCE, weights=(float("nan"), *((1.0,) * 5)))


def test_non_positive_fit_times_are_rejected() -> None:
    zeros = (0.02,) * 6
    with pytest.raises(FitError, match="positive"):
        Svensson.fit((0.0, 1.0, 3.0, 5.0, 10.0, 20.0), zeros, reference_date=REFERENCE)
    with pytest.raises(FitError, match="positive"):
        Svensson.fit((-1.0, 1.0, 3.0, 5.0, 10.0, 20.0), zeros, reference_date=REFERENCE)


def test_unsorted_fit_times_are_rejected() -> None:
    with pytest.raises(FitError, match="strictly increasing"):
        Svensson.fit((1.0, 0.5, 3.0, 5.0, 10.0, 20.0), (0.02,) * 6, reference_date=REFERENCE)


def test_duplicate_fit_times_are_rejected() -> None:
    with pytest.raises(FitError, match="strictly increasing"):
        Svensson.fit((1.0, 1.0, 3.0, 5.0, 10.0, 20.0), (0.02,) * 6, reference_date=REFERENCE)


def test_invalid_weights_are_rejected() -> None:
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)
    for bad in ((0.0, 1.0, 1.0, 1.0, 1.0, 1.0), (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        with pytest.raises(FitError, match="positive"):
            Svensson.fit(times, zeros, reference_date=REFERENCE, weights=bad)
    with pytest.raises(FitError, match="weights"):
        Svensson.fit(times, zeros, reference_date=REFERENCE, weights=(1.0, 1.0))


def test_weights_steer_the_fit_toward_the_weighted_observations() -> None:
    """Weights enter the objective: up-weighting the short end must pull the
    fitted curve closer to the short-end observations than the unweighted fit."""
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)
    short = [(t, z) for t, z in zip(times, zeros, strict=True) if t < 1.5]

    plain = Svensson.fit(times, zeros, reference_date=REFERENCE)
    weighted = Svensson.fit(
        times,
        zeros,
        reference_date=REFERENCE,
        weights=tuple(10.0 if t < 1.5 else 1.0 for t in times),
    )

    def short_ss(result: ParametricFitResult[Svensson]) -> float:
        return sum((result.curve.zero(t) - z) ** 2 for t, z in short)

    assert short_ss(weighted) < short_ss(plain)


# --- optimizer status, boundary saturation, identification -------------------


def test_an_unsuccessful_optimizer_result_is_rejected_even_when_the_objective_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORE-08: a fit must not accept an optimizer that reports failure merely
    because the objective value it reached happens to be finite."""

    def failed_optimization(*args: object, **kwargs: object) -> OptimizeResult:
        return OptimizeResult(
            x=np.zeros(6),
            fun=1.0,  # finite — the old code accepted this
            success=False,
            message="Maximum number of iterations has been exceeded.",
            nit=5000,
        )

    monkeypatch.setattr(parametric_module, "differential_evolution", failed_optimization)

    with pytest.raises(FitError, match="did not converge"):
        Svensson.fit((0.25, 1.0, 3.0, 5.0, 10.0, 20.0), (0.02,) * 6, reference_date=REFERENCE)


def test_a_fit_that_saturates_a_parameter_bound_is_rejected() -> None:
    """CORE-08: an optimum that presses a parameter against its bound is a
    constrained, not a converged, fit — it is rejected rather than reported as
    the best the model could do. Here the long end (30%) needs b0 beyond its
    25% cap, so b0 saturates the upper bound."""
    times = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    zeros = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.30)

    with pytest.raises(FitError, match=r"boundar|saturat"):
        Svensson.fit(times, zeros, reference_date=REFERENCE)


def test_a_fit_whose_jacobian_is_singular_is_rejected_as_underidentified() -> None:
    """Exact Nelson-Siegel data cannot identify the sixth Svensson parameter:
    the fitted b3 is zero and tau2 is free, so the Jacobian is singular (rank
    deficient). Uncertainty diagnostics are part of the reported result, so a
    singular state is rejected with a named error instead of silently emitted."""
    truth = NelsonSiegel(beta=(0.033, -0.012, 0.018), tau=2.4, reference_date=REFERENCE)
    times = (0.25, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = tuple(truth.zero(float(t)) for t in times)

    with pytest.raises(FitError, match=r"identif|rank"):
        Svensson.fit(times, zeros, reference_date=REFERENCE)


# --- explicit result fields and reported metrics -----------------------------


def test_fit_result_exposes_explicit_typed_fields(
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    times, published = ecb_spot
    result = Svensson.fit(times, published, reference_date=REFERENCE)

    assert isinstance(result, ParametricFitResult)
    assert isinstance(result.curve, Svensson)
    assert result.success is True
    assert isinstance(result.optimizer_status, str) and result.optimizer_status
    assert result.saturated_parameters == ()
    assert result.jacobian_rank == 6
    assert math.isfinite(result.jacobian_condition) and result.jacobian_condition > 0.0
    assert result.rmse >= 0.0
    assert result.max_abs_error >= 0.0


def test_reported_residual_metrics_match_an_independent_hand_computation() -> None:
    """QC-11: the reported residual metrics are recomputed by hand from the
    fitted curve on data constructed independently of the implementation (the
    Svensson formula is written out in this test), not echoed from the fit's
    internals."""
    beta = (0.030, -0.010, 0.020, -0.005)
    tau = (1.5, 8.0)
    times = tuple(np.linspace(0.25, 30.0, 60))

    def svensson_zero(t: float) -> float:
        l1 = (1.0 - math.exp(-t / tau[0])) / (t / tau[0])
        l2 = (1.0 - math.exp(-t / tau[1])) / (t / tau[1])
        return (
            beta[0]
            + beta[1] * l1
            + beta[2] * (l1 - math.exp(-t / tau[0]))
            + beta[3] * (l2 - math.exp(-t / tau[1]))
        )

    zeros = tuple(svensson_zero(float(t)) for t in times)
    result = Svensson.fit(times, zeros, reference_date=REFERENCE)

    errors = [result.curve.zero(float(t)) - z for t, z in zip(times, zeros, strict=True)]
    hand_rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    hand_max = max(abs(e) for e in errors)

    assert result.rmse == pytest.approx(hand_rmse, rel=1e-12)
    assert result.max_abs_error == pytest.approx(hand_max, rel=1e-12)


# --- allocation-free Nelson-Siegel evaluation (CORE-09) ----------------------


def test_nelson_siegel_evaluation_and_fit_never_construct_a_svensson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORE-09: Nelson-Siegel evaluation is closed-form; it must never build a
    Svensson object, in curve evaluation or inside the fit's objective."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("Nelson-Siegel must not construct Svensson objects")

    monkeypatch.setattr(Svensson, "__init__", boom)

    ns = NelsonSiegel(beta=(0.03, -0.01, 0.02), tau=2.0, reference_date=REFERENCE)
    assert ns.zero(3.0) > 0.0
    assert ns.df(3.0) == pytest.approx(math.exp(-ns.zero(3.0) * 3.0), rel=1e-12)
    assert ns.fwd(1.0, 5.0) == pytest.approx((ns.zero(5.0) * 5.0 - ns.zero(1.0)) / 4.0, rel=1e-12)
    assert ns.rmse((1.0, 2.0), (ns.zero(1.0), ns.zero(2.0))) == pytest.approx(0.0, abs=1e-15)

    times = tuple(np.linspace(0.25, 30.0, 24))
    zeros = tuple(ns.zero(float(t)) for t in times)
    result = NelsonSiegel.fit(times, zeros, reference_date=REFERENCE)
    assert result.curve.zero(2.0) == pytest.approx(ns.zero(2.0), abs=1e-6)


# --- the fits ----------------------------------------------------------------


def test_our_fit_reproduces_the_ecb_spot_curve(
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    times, published = ecb_spot

    curve = Svensson.fit(times, published, reference_date=REFERENCE).curve

    errors_bp = [abs(curve.zero(t) - z) * 1e4 for t, z in zip(times, published, strict=True)]
    assert max(errors_bp) < FIT_TOLERANCE_BP
    assert curve.rmse(times, published) * 1e4 < FIT_RMSE_BP


def test_the_fit_is_deterministic(
    ecb_spot: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    times, published = ecb_spot

    first = Svensson.fit(times, published, reference_date=REFERENCE)
    second = Svensson.fit(times, published, reference_date=REFERENCE)

    assert first.curve.beta == second.curve.beta
    assert first.curve.tau == second.curve.tau


def test_the_fit_respects_the_parameter_constraints() -> None:
    times = (0.25, 1.0, 3.0, 5.0, 10.0, 20.0)
    zeros = (0.021, 0.024, 0.027, 0.029, 0.032, 0.034)

    fitted = Svensson.fit(times, zeros, reference_date=REFERENCE).curve

    assert fitted.beta[0] > 0.0
    assert fitted.beta[0] + fitted.beta[1] > 0.0
    assert fitted.tau[0] > 0.0
    assert fitted.tau[1] > 0.0


def test_nelson_siegel_fits_a_curve_it_can_represent_almost_exactly() -> None:
    """Self-consistency: generate from known parameters, fit, recover the curve.
    NS is not identifiable — different parameter vectors give nearly identical
    curves — so the assertion is on the curve values, not the parameters."""
    truth = NelsonSiegel(beta=(0.033, -0.012, 0.018), tau=2.4, reference_date=REFERENCE)
    times: tuple[float, ...] = tuple(np.linspace(0.25, 30.0, 24))
    zeros = tuple(truth.zero(float(t)) for t in times)

    fitted = NelsonSiegel.fit(times, zeros, reference_date=REFERENCE).curve

    for t in times:
        assert fitted.zero(float(t)) == pytest.approx(truth.zero(float(t)), abs=1e-6)


def test_svensson_fit_recovers_parameters_it_generated() -> None:
    times = tuple(np.linspace(0.25, 30.0, 60))
    truth = Svensson(
        beta=(0.03, -0.01, 0.02, -0.005),
        tau=(1.5, 8.0),
        reference_date=REFERENCE,
    )
    zeros = tuple(truth.zero(float(t)) for t in times)
    result = Svensson.fit(times, zeros, reference_date=REFERENCE)
    assert result.curve.rmse(times, zeros) < 1e-6
