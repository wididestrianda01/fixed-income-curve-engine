"""The curve contract, exercised through the flat implementation."""

from __future__ import annotations

import math
from datetime import date

import pytest

from curveengine.curves.protocol import CurveSet, DiscountCurve, FlatCurve, curve_time

REFERENCE = date(2026, 7, 24)


def test_flat_curve_satisfies_the_discount_curve_protocol() -> None:
    assert isinstance(FlatCurve(reference_date=REFERENCE, rate=0.03), DiscountCurve)


def test_curve_time_is_act_365f_years() -> None:
    assert curve_time(REFERENCE, date(2027, 7, 24)) == pytest.approx(365 / 365)
    assert curve_time(REFERENCE, REFERENCE) == 0.0


def test_flat_curve_discounts_continuously() -> None:
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)

    assert curve.df(0.0) == 1.0
    assert curve.df(5.0) == pytest.approx(math.exp(-0.15))
    assert curve.zero(5.0) == pytest.approx(0.03)


def test_flat_curve_forward_equals_the_flat_rate() -> None:
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)

    assert curve.fwd(2.0, 5.0) == pytest.approx(0.03)


def test_negative_time_is_rejected() -> None:
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)

    with pytest.raises(ValueError, match="non-negative"):
        curve.df(-1.0)


def test_curveset_single_maps_every_forecast_lookup_to_the_discount_curve() -> None:
    """The naive single-curve world, kept available so Phase 3 can price the
    same instrument both ways and show the difference."""
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)
    curves = CurveSet.single(curve)

    assert curves.forecast_for("3M") is curve
    assert curves.discount is curve


def test_curveset_forecast_lookup_names_the_available_tenors_when_missing() -> None:
    curves = CurveSet(
        discount=FlatCurve(reference_date=REFERENCE, rate=0.03),
        forecast={"3M": FlatCurve(reference_date=REFERENCE, rate=0.032)},
    )

    with pytest.raises(KeyError, match="3M"):
        curves.forecast_for("6M")
