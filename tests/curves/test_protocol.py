"""The curve contract, exercised through the flat implementation."""

from __future__ import annotations

import math
from datetime import date

import pytest

from yieldcurve.curves.protocol import (
    CurveSet,
    DiscountCurve,
    Fixings,
    FlatCurve,
    MissingFixingError,
    curve_time,
)

REFERENCE = date(2026, 7, 24)
LATER = date(2026, 8, 24)


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
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)
    curves = CurveSet.single(curve)

    assert curves.forecast_for("3M") is curve
    assert curves.forecast_for("9M") is curve  # never raises
    assert curves.discount is curve


def test_curveset_forecast_lookup_names_the_available_tenors_when_missing() -> None:
    curves = CurveSet(
        discount=FlatCurve(reference_date=REFERENCE, rate=0.03),
        forecast={"3M": FlatCurve(reference_date=REFERENCE, rate=0.032)},
    )

    with pytest.raises(KeyError, match="3M"):
        curves.forecast_for("6M")


def test_curveset_rejects_mixed_reference_dates() -> None:
    with pytest.raises(ValueError, match="reference date differs"):
        CurveSet(
            discount=FlatCurve(reference_date=REFERENCE, rate=0.03),
            forecast={
                "3M": FlatCurve(reference_date=LATER, rate=0.032),
            },
        )


def test_curveset_single_does_not_allocate_on_read() -> None:
    curve = FlatCurve(reference_date=REFERENCE, rate=0.03)
    curves = CurveSet.single(curve)
    # Repeated forecast lookups return the same object reference
    a = curves.forecast_for("3M")
    b = curves.forecast_for("6M")
    assert a is b is curve


def test_fixings_term_rate_raises_on_missing() -> None:
    fix = Fixings()
    with pytest.raises(MissingFixingError, match="3M @ 2026-07-24"):
        fix.term_rate("3M", REFERENCE)


def test_fixings_overnight_rate_raises_on_missing() -> None:
    fix = Fixings()
    with pytest.raises(MissingFixingError, match="2026-07-24"):
        fix.overnight_rate(REFERENCE)
