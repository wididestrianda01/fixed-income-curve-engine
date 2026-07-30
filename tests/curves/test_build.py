"""Curves built from the committed snapshot."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from yieldcurve.curves.build import (
    CurveDataError,
    sek_government_curve,
    usd_curveset,
    usd_forecast_curve,
    usd_ois_curve,
)
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.market.snapshot import Snapshot

pytestmark = pytest.mark.usefixtures("snapshot")


def test_ois_curve_reprices_its_own_quotes(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import usd_ois_quotes
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.curves.pricing import par_rate
    from yieldcurve.curves.protocol import CurveSet

    asof = date(2026, 7, 24)
    curve = usd_ois_curve(snapshot, asof, method=InterpMethod.LOG_LINEAR_DF)
    curves = CurveSet.single(curve)

    for quote in usd_ois_quotes(snapshot, asof):
        assert par_rate(quote.instrument, curves, asof) == pytest.approx(  # type: ignore[arg-type]
            quote.rate, abs=1e-10
        )


def test_discount_and_forecast_are_genuinely_different_curves(
    snapshot: Snapshot,
) -> None:
    asof = date(2026, 7, 24)
    curves = usd_curveset(snapshot, asof)

    discount_5y = curves.discount.zero(5.0)
    forecast_5y = curves.forecast_for("3M").zero(5.0)

    assert abs(forecast_5y - discount_5y) > 1e-4


def test_forecast_curve_lies_above_the_ois_curve(snapshot: Snapshot) -> None:
    asof = date(2026, 7, 24)
    ois = usd_ois_curve(snapshot, asof)
    forecast = usd_forecast_curve(snapshot, asof)

    spreads = [forecast.zero(t) - ois.zero(t) for t in (1.0, 2.0, 5.0, 10.0)]

    assert all(s > 0.0 for s in spreads), spreads


def test_forecast_for_an_unknown_tenor_names_what_is_available(
    snapshot: Snapshot,
) -> None:
    curves = usd_curveset(snapshot, date(2026, 7, 24))

    with pytest.raises(KeyError, match="6M"):
        curves.forecast_for("6M")


@pytest.mark.parametrize("method", list(InterpMethod))
def test_every_interpolation_method_builds(snapshot: Snapshot, method: InterpMethod) -> None:
    curve = usd_ois_curve(snapshot, date(2026, 7, 24), method=method)

    assert curve.df(10.0) < curve.df(1.0) < 1.0


def test_sek_curve_covers_the_key_rate_grid(snapshot: Snapshot) -> None:
    curve = sek_government_curve(snapshot, date(2026, 7, 24))

    assert max(curve.times) == pytest.approx(10.0, abs=0.6)
    assert all(curve.df(t) > 0.0 for t in (0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0))


def test_basis_is_defined_at_every_requested_tenor(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import government_swap_basis

    tenors = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
    basis = government_swap_basis(snapshot, date(2026, 7, 24), tenors)

    assert set(basis) == set(tenors)
    assert all(isinstance(v, float) for v in basis.values())


def test_basis_is_of_a_plausible_magnitude(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import government_swap_basis

    basis = government_swap_basis(snapshot, date(2026, 7, 24), (2.0, 5.0, 10.0, 30.0))

    assert all(abs(v) < 0.015 for v in basis.values()), basis


def test_basis_compares_zeros_not_a_zero_against_a_par_yield(
    snapshot: Snapshot,
) -> None:
    from yieldcurve.curves.build import government_swap_basis, usd_government_curve

    asof = date(2026, 7, 24)
    gov = usd_government_curve(snapshot, asof)
    ois = usd_ois_curve(snapshot, asof)
    basis = government_swap_basis(snapshot, asof, (10.0,))

    assert basis[10.0] == pytest.approx(ois.zero(10.0) - gov.zero(10.0), abs=1e-12)


def test_missing_dataset_raises_a_named_error(snapshot: Snapshot, tmp_path: Path) -> None:
    empty = Snapshot(date=date(2026, 7, 24), root=Path(str(tmp_path)))

    with pytest.raises((CurveDataError, FileNotFoundError)):
        usd_ois_curve(empty, date(2026, 7, 24))
