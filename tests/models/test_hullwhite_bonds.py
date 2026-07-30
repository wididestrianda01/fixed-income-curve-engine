"""Hull-White analytic bond prices and the initial-curve fit."""

from __future__ import annotations

import math
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from curveengine.curves.parametric import Svensson
from curveengine.curves.protocol import FlatCurve
from curveengine.models.hullwhite import HullWhite, ModelError

ASOF = date(2026, 7, 24)
SEED = 20260727


@pytest.fixture
def flat_model() -> HullWhite:
    return HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=0.05, sigma=0.01)


def test_the_model_reprices_the_initial_curve_exactly(flat_model: HullWhite) -> None:
    for T in (0.25, 1.0, 2.0, 5.0, 10.0, 30.0):  # noqa: N806
        assert flat_model.zcb(0.0, T, flat_model.r0) == pytest.approx(
            flat_model.curve.df(T), abs=1e-10
        )


def test_the_model_reprices_a_sloped_curve_exactly() -> None:
    curve = Svensson(
        reference_date=ASOF,
        beta=(0.035, -0.015, 0.020, -0.010),
        tau=(1.5, 8.0),
    )
    model = HullWhite(curve=curve, a=0.08, sigma=0.012)

    for T in (0.5, 3.0, 7.0, 20.0):  # noqa: N806
        assert model.zcb(0.0, T, model.r0) == pytest.approx(curve.df(T), abs=1e-10)


def test_B_matches_its_closed_form(flat_model: HullWhite) -> None:  # noqa: N802
    assert flat_model.B(1.0, 6.0) == pytest.approx((1 - math.exp(-0.05 * 5.0)) / 0.05, rel=1e-14)


def test_B_degenerates_to_the_time_gap_as_mean_reversion_vanishes() -> None:  # noqa: N802
    tiny = HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=1e-12, sigma=0.01)

    assert tiny.B(0.0, 10.0) == pytest.approx(10.0, rel=1e-9)


def test_B_is_zero_at_equal_times(flat_model: HullWhite) -> None:  # noqa: N802
    assert flat_model.B(3.0, 3.0) == pytest.approx(0.0, abs=1e-15)


def test_a_bond_maturing_now_is_worth_par(flat_model: HullWhite) -> None:
    assert flat_model.zcb(5.0, 5.0, 0.04) == pytest.approx(1.0, rel=1e-14)


def test_bond_price_falls_as_the_short_rate_rises(flat_model: HullWhite) -> None:
    prices = [flat_model.zcb(1.0, 10.0, r) for r in (0.01, 0.02, 0.03, 0.04)]

    assert all(b < a for a, b in zip(prices, prices[1:], strict=False))  # noqa: RUF007


def test_longer_bonds_are_more_sensitive_to_the_short_rate(flat_model: HullWhite) -> None:
    short = flat_model.zcb(1.0, 2.0, 0.05) / flat_model.zcb(1.0, 2.0, 0.03)
    long = flat_model.zcb(1.0, 20.0, 0.05) / flat_model.zcb(1.0, 20.0, 0.03)

    assert long < short


def test_instantaneous_forward_matches_a_flat_curve(flat_model: HullWhite) -> None:
    for t in (0.0, 1.0, 10.0):
        assert flat_model.instantaneous_fwd(t) == pytest.approx(0.03, abs=1e-8)


def test_instantaneous_forward_matches_svensson_analytically() -> None:
    curve = Svensson(reference_date=ASOF, beta=(0.035, -0.015, 0.020, -0.010), tau=(1.5, 8.0))
    model = HullWhite(curve=curve, a=0.05, sigma=0.01)

    for t in (0.5, 2.0, 10.0, 25.0):
        assert model.instantaneous_fwd(t) == pytest.approx(curve.instantaneous_fwd(t), abs=1e-7)


def test_the_model_binds_to_the_protocol_not_a_concrete_curve() -> None:
    from pathlib import Path

    import curveengine.models.hullwhite as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    assert "InterpolatedDiscountCurve" not in body
    assert "Svensson" not in body


def test_negative_volatility_is_rejected() -> None:
    with pytest.raises(ModelError, match="sigma"):
        HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=0.05, sigma=-0.01)


def test_negative_mean_reversion_is_rejected() -> None:
    with pytest.raises(ModelError, match="mean reversion"):
        HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=-0.05, sigma=0.01)


@given(
    a=st.floats(min_value=0.001, max_value=0.5),
    sigma=st.floats(min_value=0.0001, max_value=0.05),
    T=st.floats(min_value=0.1, max_value=30.0),
)
@settings(max_examples=200, deadline=None)
def test_initial_curve_fit_holds_for_every_parameter_pair(a: float, sigma: float, T: float) -> None:  # noqa: N803
    model = HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=a, sigma=sigma)

    assert model.zcb(0.0, T, model.r0) == pytest.approx(model.curve.df(T), abs=1e-10)
