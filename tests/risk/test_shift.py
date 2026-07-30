"""The one curve transformation every risk number is built on."""

from __future__ import annotations

import math
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.risk.scenarios import Scenario, parallel, shift_curve, shift_curveset

ASOF = date(2026, 7, 24)


def test_parallel_shift_moves_every_zero_by_the_same_amount() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    shifted = shift_curve(base, parallel(0.01))

    assert shifted.reference_date == ASOF
    for t in (0.25, 1.0, 5.0, 30.0):
        assert shifted.zero(t) == pytest.approx(0.04, abs=1e-12)


def test_shifted_discount_factors_are_exact_not_approximated() -> None:
    """df_new = df_old * exp(-dz * t). A shift implemented by re-deriving zeros
    from bumped discount factors and re-exponentiating loses digits; this
    pins the closed form."""
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    shifted = shift_curve(base, parallel(0.005))

    for t in (0.5, 3.0, 10.0):
        assert shifted.df(t) == pytest.approx(base.df(t) * math.exp(-0.005 * t), rel=1e-15)


def test_zero_shift_is_the_identity() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    shifted = shift_curve(base, parallel(0.0))

    for t in (0.1, 2.0, 25.0):
        assert shifted.df(t) == pytest.approx(base.df(t), rel=1e-15)


def test_shifting_twice_composes() -> None:
    """Additivity in the zero rate. Relied on by the key-rate identity: the sum
    of the hat shifts must equal one parallel shift."""
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    once = shift_curve(shift_curve(base, parallel(0.01)), parallel(0.005))
    twice = shift_curve(base, parallel(0.015))

    for t in (1.0, 7.0):
        assert once.df(t) == pytest.approx(twice.df(t), rel=1e-14)


@given(
    rate=st.floats(min_value=-0.01, max_value=0.10),
    size=st.floats(min_value=-0.03, max_value=0.03),
    t=st.floats(min_value=0.01, max_value=30.0),
)
def test_shifted_curve_still_satisfies_the_protocol_identities(
    rate: float, size: float, t: float
) -> None:
    """A shifted curve is a curve. zero(t) and df(t) must stay consistent, and
    a negative zero must remain representable — the whole point of continuous
    compounding is that df stays positive for any real rate."""
    shifted = shift_curve(FlatCurve(reference_date=ASOF, rate=rate), parallel(size))

    assert shifted.df(t) > 0.0
    assert shifted.zero(t) == pytest.approx(-math.log(shifted.df(t)) / t, abs=1e-12)


def test_forward_rates_are_unchanged_by_a_parallel_shift_plus_the_shift() -> None:
    """A parallel zero shift of s moves every instantaneous forward by exactly
    s too. If the forward moves by something else, ``fwd`` is being computed
    from a stale curve rather than the shifted one."""
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    shifted = shift_curve(base, parallel(0.01))

    assert shifted.fwd(2.0, 3.0) == pytest.approx(base.fwd(2.0, 3.0) + 0.01, abs=1e-12)


def test_shift_curveset_moves_both_curves() -> None:
    """A scenario that shocks the discount curve and leaves the forecast curve
    alone silently reports basis risk as rate risk. Both move."""
    discount = FlatCurve(reference_date=ASOF, rate=0.03)
    forecast = FlatCurve(reference_date=ASOF, rate=0.035)
    curves = CurveSet(discount=discount, forecast={"3M": forecast})

    shifted = shift_curveset(curves, parallel(0.01))

    assert shifted.discount.zero(5.0) == pytest.approx(0.04, abs=1e-12)
    assert shifted.forecast_for("3M").zero(5.0) == pytest.approx(0.045, abs=1e-12)


def test_a_non_parallel_scenario_is_applied_pointwise() -> None:
    steep = Scenario(name="steep", shift=lambda t: 0.0001 * t)
    base = FlatCurve(reference_date=ASOF, rate=0.03)

    shifted = shift_curve(base, steep)

    assert shifted.zero(1.0) == pytest.approx(0.0301, abs=1e-12)
    assert shifted.zero(10.0) == pytest.approx(0.0310, abs=1e-12)


def test_scenario_is_frozen() -> None:
    scenario = parallel(0.02)

    with pytest.raises(AttributeError):
        scenario.name = "mutated"  # type: ignore[misc]


def test_shifted_fwd_rejects_non_increasing_tenors() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.03)
    shifted = shift_curve(base, parallel(0.01))

    with pytest.raises(ValueError, match=r"fwd requires t2 > t1"):
        shifted.fwd(2.0, 1.0)


def test_shifting_a_single_curveset_shifts_the_one_curve() -> None:
    curves = CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.03))

    shifted = shift_curveset(curves, parallel(0.01))

    assert shifted.discount.zero(5.0) == pytest.approx(0.04, abs=1e-12)
    assert shifted.forecast_for("3M").zero(5.0) == pytest.approx(0.04, abs=1e-12)
