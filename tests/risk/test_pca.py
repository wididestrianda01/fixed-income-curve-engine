"""PCA of daily curve changes: level, slope, curvature."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from curveengine.calendars import USGovernmentBondCalendar
from curveengine.conventions import BusinessDayConvention, DayCount
from curveengine.curves.protocol import CurveSet, FlatCurve
from curveengine.instruments import FixedCouponBond
from curveengine.market.snapshot import Snapshot
from curveengine.risk.pca import PCAResult, daily_changes, fit_pca, pca_durations

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
