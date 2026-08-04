"""PCA of daily curve changes: deterministic orientation, scale, and labels."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from yieldcurve.calendars import USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.risk.pca import (
    PCAError,
    PCAResult,
    daily_changes,
    fit_pca,
    pca_durations,
    pca_exposure,
)

ASOF = date(2026, 7, 24)
SEED = 20260727


@pytest.fixture
def history_result(snapshot: Snapshot) -> PCAResult:
    changes, tenors = daily_changes(snapshot.load("fred_treasury_cmt_history"))
    # Exclude DGS1MO (0.083y): money-market dynamics dominate the short end,
    # pulling explained variance down from ~97.9% to ~93.1%.
    mask = np.array([t >= 0.25 for t in tenors])
    return fit_pca(changes[:, mask], tuple(t for t, m in zip(tenors, mask, strict=True) if m))


def test_three_components_explain_most_of_the_variance(
    history_result: PCAResult,
) -> None:
    assert sum(history_result.explained_variance_ratio) > 0.95


def test_the_first_component_is_a_level_shift(history_result: PCAResult) -> None:
    first = history_result.loadings[0]

    assert np.all(first > 0) or np.all(first < 0)


def test_the_second_component_changes_sign_once(history_result: PCAResult) -> None:
    second = history_result.loadings[1]
    crossings = int(np.sum(np.diff(np.sign(second)) != 0))

    assert crossings == 1


def test_the_third_component_changes_sign_twice(history_result: PCAResult) -> None:
    third = history_result.loadings[2]
    crossings = int(np.sum(np.diff(np.sign(third)) != 0))

    assert crossings == 2


def test_components_are_orthonormal(history_result: PCAResult) -> None:
    gram = history_result.loadings @ history_result.loadings.T

    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


def test_explained_variance_ratios_are_decreasing(history_result: PCAResult) -> None:
    ratios = history_result.explained_variance_ratio

    assert list(ratios) == sorted(ratios, reverse=True)


def test_fit_recovers_a_planted_structure() -> None:
    rng = np.random.default_rng(SEED)
    tenors = (1.0, 2.0, 5.0, 10.0, 30.0)
    level = np.ones(5) / np.sqrt(5)
    slope = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    slope /= np.linalg.norm(slope)
    scores = rng.normal(size=(2000, 2)) * np.array([0.0010, 0.0004])
    changes = scores @ np.vstack([level, slope]) + rng.normal(scale=1e-6, size=(2000, 5))

    result = fit_pca(changes, tenors, n_components=2)

    assert abs(float(result.loadings[0] @ level)) > 0.999
    assert abs(float(result.loadings[1] @ slope)) > 0.999
    assert result.component_sd[0] == pytest.approx(0.0010, rel=0.1)


def test_daily_changes_are_differences_not_levels(snapshot: Snapshot) -> None:
    changes, _ = daily_changes(snapshot.load("fred_treasury_cmt_history"))

    assert np.abs(changes).mean() < 0.001
    assert changes.shape[0] >= 749


def test_pca_durations_are_named_and_finite(history_result: PCAResult) -> None:
    bond = FixedCouponBond(
        issue=ASOF,
        maturity=date(2036, 7, 24),
        coupon=0.04,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))

    durations = pca_durations(bond, curves, ASOF, history_result)

    assert set(durations) == {"level", "slope", "curvature"}
    assert all(np.isfinite(v) for v in durations.values())


def test_level_duration_dominates_for_a_bullet_bond(
    history_result: PCAResult,
) -> None:
    bond = FixedCouponBond(
        issue=ASOF,
        maturity=date(2036, 7, 24),
        coupon=0.04,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))

    durations = pca_durations(bond, curves, ASOF, history_result)

    assert abs(durations["level"]) > abs(durations["slope"])


def test_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="observations"):
        fit_pca(np.zeros((5, 10)), tuple(range(10)))


def test_repeated_fits_are_identical_and_sign_oriented() -> None:
    """QUANTRISK-11: deterministic sign orientation, stable under repeated fits.

    Two fits on the same history are bit-identical; every component's
    largest-magnitude loading entry is positive (the pinned convention, ties
    to the earliest tenor); and flipping the overall sign of the history —
    which leaves the covariance unchanged — must not flip the loadings. numpy's
    raw SVD does flip them, so this pins the orientation rule itself."""
    rng = np.random.default_rng(SEED)
    tenors = (0.25, 1.0, 2.0, 5.0, 10.0)
    changes = rng.normal(scale=0.001, size=(400, len(tenors)))

    first = fit_pca(changes, tenors)
    second = fit_pca(changes, tenors)
    flipped = fit_pca(-changes, tenors)

    np.testing.assert_array_equal(first.loadings, second.loadings)
    np.testing.assert_array_equal(first.loadings, flipped.loadings)
    assert first.component_names == second.component_names
    for k in range(first.loadings.shape[0]):
        pivot = int(np.argmax(np.abs(first.loadings[k])))
        assert first.loadings[k, pivot] > 0.0


def test_pca_durations_are_direction_only_and_unit_pinned() -> None:
    """QUANTRISK-02/04: the direction duration is the per-unit-rate modified
    duration along the component's unit-norm loading direction, in years.

    Independent oracle: a zero-coupon bond on a flat curve prices at
    100*exp(-r*T). A uniform (level) loading of norm 1 over n tenors shifts
    every zero rate by 1/sqrt(n) per unit of direction, so the direction
    duration is exactly T/sqrt(n) — no library pricing or risk code takes part
    in the expected value."""
    rng = np.random.default_rng(SEED)
    tenors = (1.0, 2.0, 5.0, 10.0, 30.0)
    n = len(tenors)
    level = np.ones(n) / np.sqrt(n)
    scores = rng.normal(size=(2000, 1)) * 0.001
    changes = scores @ level.reshape(1, -1) + rng.normal(scale=1e-6, size=(2000, n))

    result = fit_pca(changes, tenors, n_components=1)
    assert result.component_names == ("level",)
    assert np.all(result.loadings[0] > 0.0)

    maturity = date(2036, 7, 21)  # exactly 3650 days after ASOF: T = 10.0
    bond = FixedCouponBond(
        issue=ASOF,
        maturity=maturity,
        coupon=0.0,
        frequency=1,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))

    duration = pca_durations(bond, curves, ASOF, result)["level"]
    t_years = (maturity - ASOF).days / 365.0
    assert duration == pytest.approx(t_years / np.sqrt(n), rel=1e-2)


def test_pca_exposure_retains_the_empirical_component_scale() -> None:
    """QUANTRISK-02/04: the one-standard-deviation exposure equals the
    direction duration times the empirical component standard deviation. The
    empirical scale enters the exposure, never the direction duration."""
    rng = np.random.default_rng(SEED)
    tenors = (1.0, 2.0, 5.0, 10.0, 30.0)
    n = len(tenors)
    level = np.ones(n) / np.sqrt(n)
    slope = np.array([-2.0, -1.5, -0.5, 0.5, 1.5])
    slope /= np.linalg.norm(slope)
    scores = rng.normal(size=(2000, 2)) * np.array([0.0010, 0.0004])
    changes = scores @ np.vstack([level, slope]) + rng.normal(scale=1e-6, size=(2000, n))

    result = fit_pca(changes, tenors, n_components=2)
    assert result.component_names == ("level", "slope")

    bond = FixedCouponBond(
        issue=ASOF,
        maturity=date(2036, 7, 21),
        coupon=0.0,
        frequency=1,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))

    durations = pca_durations(bond, curves, ASOF, result)
    exposures = pca_exposure(bond, curves, ASOF, result)
    assert set(exposures) == {"level", "slope"}
    for name, sd in zip(result.component_names, result.component_sd, strict=True):
        assert exposures[name] == pytest.approx(durations[name] * sd, rel=1e-3)


@pytest.mark.parametrize(
    "measure",
    [pca_durations, pca_exposure],
    ids=["pca_durations", "pca_exposure"],
)
def test_pca_measures_reject_a_materially_zero_base_pv(
    history_result: PCAResult, measure: Callable[..., dict[str, float]]
) -> None:
    """Error policy: a normalized PCA measure must not divide by a materially
    zero present value. A 40-year zero-coupon bond on a 50% flat curve prices
    at 100*exp(-0.5*40) ≈ 2e-7 — far below MIN_UNIT_PRICE x face — so both
    APIs raise PCAError (via the _require_base_price wrapper) instead of
    returning inf/NaN."""
    bond = FixedCouponBond(
        issue=ASOF,
        maturity=date(2066, 7, 24),
        coupon=0.0,
        frequency=1,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.5))

    with pytest.raises(PCAError, match="materially zero"):
        measure(bond, curves, ASOF, history_result)


def test_economic_labels_stay_pc_when_loading_shape_fails() -> None:
    """Spec section 4: components stay PC1/PC2/PC3 unless the loading-shape
    criterion for that position passes. A history dominated by a V-shaped
    mode (two sign changes) must not earn the 'level' label, and the shape
    diagnostic records why."""
    rng = np.random.default_rng(SEED)
    tenors = (1.0, 2.0, 5.0, 10.0, 30.0)
    v = np.array([-1.0, 0.5, 1.0, 0.5, -1.0])
    v /= np.linalg.norm(v)
    changes = rng.normal(size=(300, 1)) * 0.001 @ v.reshape(1, -1)

    result = fit_pca(changes, tenors, n_components=1)

    assert result.component_names == ("PC1",)
    assert result.loading_shape == ("two sign changes",)


def test_a_constant_history_is_rejected() -> None:
    with pytest.raises(PCAError, match="degenerate"):
        fit_pca(np.zeros((100, 5)), (1.0, 2.0, 5.0, 10.0, 30.0))


def test_a_rank_deficient_history_is_rejected() -> None:
    rng = np.random.default_rng(SEED)
    tenors = (1.0, 2.0, 5.0, 10.0, 30.0)
    changes = rng.normal(size=(300, 1)) @ np.ones((1, len(tenors)))
    with pytest.raises(PCAError, match="rank"):
        fit_pca(changes, tenors, n_components=3)


def test_a_nonfinite_history_is_rejected() -> None:
    changes = np.full((100, 5), 0.001)
    changes[0, 0] = np.nan
    with pytest.raises(PCAError, match="finite"):
        fit_pca(changes, (1.0, 2.0, 5.0, 10.0, 30.0))


def test_duplicate_tenors_are_rejected() -> None:
    """A tenor grid must be strictly ascending: a duplicate would silently
    overwrite a loading entry downstream (error policy: reject before
    arithmetic)."""
    changes = np.zeros((100, 5))
    with pytest.raises(PCAError, match="increasing"):
        fit_pca(changes, (1.0, 2.0, 5.0, 5.0, 10.0))


def test_daily_changes_rejects_an_incomplete_history() -> None:
    """A tenor missing on one date is a misaligned history: the pivot cell is
    NaN and the daily differences would silently skip a date."""
    history = pd.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 2)],
            "tenor_years": [1.0, 5.0, 1.0],
            "rate": [0.0400, 0.0450, 0.0401],
        }
    )
    with pytest.raises(PCAError, match="tenor"):
        daily_changes(history)


def test_daily_changes_rejects_an_empty_history() -> None:
    with pytest.raises(PCAError, match="no observations"):
        daily_changes(pd.DataFrame(columns=["date", "tenor_years", "rate"]))
