"""Interpolation over discount factors."""

from __future__ import annotations

import itertools
import math
from datetime import date

import numpy as np
import pytest

from yieldcurve.curves.interpolation import (
    CurveConstructionError,
    InterpMethod,
    InterpolatedDiscountCurve,
)

REFERENCE = date(2026, 7, 24)
TIMES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
ZEROS = (0.0215, 0.0228, 0.0250, 0.0272, 0.0301, 0.0324)
DFS = tuple(math.exp(-z * t) for t, z in zip(TIMES, ZEROS, strict=True))

ALL_METHODS = list(InterpMethod)
MONOTONE_METHODS = [InterpMethod.LOG_LINEAR_DF, InterpMethod.MONOTONE_CONVEX]


def build(method: InterpMethod) -> InterpolatedDiscountCurve:
    return InterpolatedDiscountCurve(reference_date=REFERENCE, times=TIMES, dfs=DFS, method=method)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_curve_passes_through_its_knots(method: InterpMethod) -> None:
    """Non-negotiable for every scheme: an interpolator that misses its own
    inputs is not an interpolator."""
    curve = build(method)

    for t, df in zip(TIMES, DFS, strict=True):
        assert curve.df(t) == pytest.approx(df, rel=1e-12)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_df_at_zero_is_one(method: InterpMethod) -> None:
    assert build(method).df(0.0) == 1.0


@pytest.mark.parametrize("method", MONOTONE_METHODS)
def test_discount_factors_decrease_with_time_for_monotone_knots(method: InterpMethod) -> None:
    """For monotone knots, log-linear DF and monotone convex preserve
    monotonicity. Cubic is deliberately excluded: a natural cubic spline can
    overshoot and is only guaranteed to pass through its knots, which the
    overshoot test below pins down."""
    curve = build(method)
    grid = np.linspace(0.01, 10.0, 400)
    values = [curve.df(float(t)) for t in grid]  # type: ignore[attr-defined]

    assert all(later < earlier for earlier, later in itertools.pairwise(values))


def test_cubic_log_df_can_overshoot_between_monotone_knots() -> None:
    """DOC-11/QC-10: a cubic spline on log discount factors is not guaranteed
    monotone even when the knot discount factors are monotone decreasing. The
    natural spline below dips through the data and rises again between knots."""
    times = (0.25, 0.5, 1.0, 2.0)
    dfs = (1.0, 0.97, 0.90, 0.89)  # strictly decreasing knots
    assert all(later < earlier for earlier, later in itertools.pairwise(dfs))

    curve = InterpolatedDiscountCurve(
        reference_date=REFERENCE, times=times, dfs=dfs, method=InterpMethod.CUBIC_LOG_DF
    )
    grid = np.linspace(0.02, 2.0, 300)
    values = [curve.df(float(t)) for t in grid]  # type: ignore[attr-defined]

    assert any(later > earlier for earlier, later in itertools.pairwise(values))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_zero_rate_recovers_the_discount_factor(method: InterpMethod) -> None:
    curve = build(method)

    for t in (0.3, 1.7, 4.2, 9.9):
        assert curve.df(t) == pytest.approx(math.exp(-curve.zero(t) * t), rel=1e-12)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_forward_composes_into_the_discount_factor(method: InterpMethod) -> None:
    """df(t2) = df(t1) * exp(-f(t1,t2) * (t2-t1)) is the definition of fwd, and
    a scheme that violates it is arbitrageable on its own output."""
    curve = build(method)

    t1, t2 = 1.5, 3.5
    implied = curve.df(t1) * math.exp(-curve.fwd(t1, t2) * (t2 - t1))

    assert implied == pytest.approx(curve.df(t2), rel=1e-12)


def test_log_linear_is_exactly_linear_in_log_df() -> None:
    curve = build(InterpMethod.LOG_LINEAR_DF)

    midpoint_log_df = 0.5 * (math.log(DFS[2]) + math.log(DFS[3]))

    assert math.log(curve.df(1.5)) == pytest.approx(midpoint_log_df, rel=1e-12)


def test_log_linear_forwards_are_piecewise_constant() -> None:
    """The known cost of log-linear: forwards are flat within a bucket and jump
    at every knot. This test pins the artefact rather than hiding it, because
    Notebook 02 plots exactly this."""
    curve = build(InterpMethod.LOG_LINEAR_DF)

    inside = [curve.instantaneous_fwd(t) for t in (1.1, 1.5, 1.9)]

    assert inside[0] == pytest.approx(inside[1], rel=1e-9)
    assert inside[1] == pytest.approx(inside[2], rel=1e-9)
    assert curve.instantaneous_fwd(2.1) != pytest.approx(inside[0], rel=1e-6)


def test_monotone_convex_forwards_are_continuous_across_a_knot() -> None:
    """The reason to prefer monotone convex: the forward curve has no jumps."""
    curve = build(InterpMethod.MONOTONE_CONVEX)

    left = curve.instantaneous_fwd(2.0 - 1e-6)
    right = curve.instantaneous_fwd(2.0 + 1e-6)

    assert left == pytest.approx(right, abs=1e-6)


def test_monotone_convex_handles_negative_forwards() -> None:
    """A curve inverted at the front end has negative forwards there. Hagan-West's
    positivity amendment would clamp them to zero, which is wrong for SEK and EUR,
    so it is deliberately not implemented — see the module docstring."""
    inverted_zeros = (0.030, 0.010, 0.019, 0.021, 0.026, 0.030)
    dfs = tuple(math.exp(-z * t) for t, z in zip(TIMES, inverted_zeros, strict=True))
    curve = InterpolatedDiscountCurve(
        reference_date=REFERENCE, times=TIMES, dfs=dfs, method=InterpMethod.MONOTONE_CONVEX
    )

    assert curve.instantaneous_fwd(0.4) < 0.0
    for t, df in zip(TIMES, dfs, strict=True):
        assert curve.df(t) == pytest.approx(df, rel=1e-12)


def test_log_linear_supports_negative_rates() -> None:
    """Negative zero rates give discount factors above 1, and log-linear
    interpolation on them must stay well defined and positive."""
    negative_front = (-0.005, 0.005, 0.015, 0.020, 0.025, 0.030)
    dfs = tuple(math.exp(-z * t) for t, z in zip(TIMES, negative_front, strict=True))
    curve = InterpolatedDiscountCurve(
        reference_date=REFERENCE, times=TIMES, dfs=dfs, method=InterpMethod.LOG_LINEAR_DF
    )

    assert curve.df(0.25) > 1.0  # a negative zero rate
    assert curve.zero(0.1) < 0.0
    assert all(curve.df(t) > 0.0 for t in (0.05, 0.5, 2.0, 5.0, 10.0))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_extrapolation_is_flat_in_the_zero_rate(method: InterpMethod) -> None:
    curve = build(method)

    assert curve.zero(30.0) == pytest.approx(curve.zero(10.0), rel=1e-12)
    assert curve.zero(0.05) == pytest.approx(curve.zero(0.25), rel=1e-12)
    assert 0.0 < curve.df(30.0) < curve.df(10.0)


def test_covered_horizon_defaults_to_the_last_knot_and_can_be_stated() -> None:
    curve = build(InterpMethod.LOG_LINEAR_DF)
    assert curve.covered_horizon == TIMES[-1]

    stated = InterpolatedDiscountCurve(
        reference_date=REFERENCE,
        times=TIMES,
        dfs=DFS,
        method=InterpMethod.LOG_LINEAR_DF,
        covered_horizon=5.0,
    )
    assert stated.covered_horizon == 5.0


def test_covered_horizon_may_not_exceed_the_last_knot() -> None:
    with pytest.raises(CurveConstructionError, match="horizon"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=TIMES,
            dfs=DFS,
            method=InterpMethod.LOG_LINEAR_DF,
            covered_horizon=20.0,
        )


def test_non_finite_knot_times_are_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="finite"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=(float("nan"), 1.0),
            dfs=(0.99, 0.98),
            method=InterpMethod.LOG_LINEAR_DF,
        )


def test_non_finite_discount_factors_are_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="finite"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=(1.0, 2.0),
            dfs=(float("inf"), 0.98),
            method=InterpMethod.LOG_LINEAR_DF,
        )


def test_unsorted_times_are_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="strictly increasing"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=(1.0, 0.5),
            dfs=(0.97, 0.98),
            method=InterpMethod.LOG_LINEAR_DF,
        )


def test_non_positive_discount_factor_is_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="positive"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=(1.0, 2.0),
            dfs=(0.97, 0.0),
            method=InterpMethod.LOG_LINEAR_DF,
        )


def test_a_time_of_zero_among_the_knots_is_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="implicit"):
        InterpolatedDiscountCurve(
            reference_date=REFERENCE,
            times=(0.0, 1.0),
            dfs=(1.0, 0.97),
            method=InterpMethod.LOG_LINEAR_DF,
        )
