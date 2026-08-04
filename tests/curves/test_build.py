"""Curves built from the committed snapshot."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from yieldcurve.curves.bootstrap import repricing_report
from yieldcurve.curves.build import (
    CurveDataError,
    sek_government_curve,
    usd_curveset,
    usd_forecast_curve,
    usd_ois_curve,
)
from yieldcurve.curves.interpolation import InterpMethod, overlay_curve
from yieldcurve.curves.protocol import curve_time
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.market.snapshot import Snapshot

pytestmark = pytest.mark.usefixtures("snapshot")

ASOF = date(2026, 7, 24)


def test_builders_default_to_the_canonical_log_linear_method(snapshot: Snapshot) -> None:
    assert usd_ois_curve(snapshot, ASOF).method is InterpMethod.LOG_LINEAR_DF
    assert sek_government_curve(snapshot, ASOF).method is InterpMethod.LOG_LINEAR_DF


def test_ois_curve_repricing_report_is_exact_within_tolerance(snapshot: Snapshot) -> None:
    """The canonical contract on snapshot data: the default log-linear build
    reprices every quote within the tolerance the typed report verifies. The
    independent oracle for the deposit/swap math lives in test_bootstrap
    (hand-derived discount factors); this asserts the report verdicts."""
    from yieldcurve.curves.build import usd_ois_quotes

    curve = usd_ois_curve(snapshot, ASOF)
    quotes = usd_ois_quotes(snapshot, ASOF)
    report = repricing_report(curve, quotes, asof=ASOF)

    assert len(report) == len(quotes)
    assert all(row.ok for row in report), [row for row in report if not row.ok]


def test_year_tenors_land_on_calendar_anniversaries(snapshot: Snapshot) -> None:
    """CORE-01: year tenors use calendar arithmetic, so 10Y and 30Y land on
    anniversaries rather than on rounded 365-day dates."""
    from yieldcurve.curves.build import usd_ois_quotes

    maturities = [quote.instrument.maturity for quote in usd_ois_quotes(snapshot, ASOF)]
    assert date(2027, 7, 24) in maturities  # 1Y anniversary
    assert date(2036, 7, 24) in maturities  # 10Y anniversary
    assert date(2056, 7, 24) in maturities  # 30Y anniversary
    assert date(2056, 7, 21) not in maturities  # the old rounded 365-day date


def test_cmt_one_month_tenor_lands_on_the_calendar_month_date(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import usd_government_curve

    curve = usd_government_curve(snapshot, ASOF)
    assert curve.times[0] == pytest.approx(curve_time(ASOF, date(2026, 8, 24)))


def test_forecast_curve_states_its_covered_horizon_and_extrapolation_rule(
    snapshot: Snapshot,
) -> None:
    """CORE-05: the forecast curve's data-backed horizon is explicit, and what
    happens beyond it is a stated, tested rule (flat in the zero rate), not a
    silent accident."""
    forecast = usd_forecast_curve(snapshot, ASOF)
    ois = usd_ois_curve(snapshot, ASOF)

    # The horizon is the largest data-backed curve time: the last knot, which
    # for calendar-year tenors is the anniversary's ACT/365F time.
    assert ois.covered_horizon == ois.times[-1]
    assert ois.covered_horizon == pytest.approx(30.0, abs=0.1)
    assert forecast.covered_horizon == forecast.times[-1]
    assert forecast.covered_horizon == pytest.approx(10.0, abs=0.05)  # last basis tenor
    # Beyond the covered horizon the stated rule is flat-in-zero extrapolation.
    assert forecast.zero(20.0) == pytest.approx(forecast.zero(forecast.times[-1]), rel=1e-12)
    assert forecast.zero(30.0) == pytest.approx(forecast.zero(20.0), rel=1e-12)
    assert 0.0 < forecast.df(20.0) < 1.0


def test_forecast_basis_with_a_hole_in_its_covered_span_is_rejected(tmp_path: Path) -> None:
    """CORE-05: a basis that reaches 10Y but omits the 5Y OIS tenor is a hole,
    not a shorter horizon, and the builder must say so instead of building a
    truncated forecast grid."""
    snap = Snapshot(date=ASOF, root=tmp_path / "snapshot")
    snap.save(
        "usd_ois_swaps",
        pd.DataFrame(
            {
                "tenor_years": [1, 2, 3, 5, 7, 10],
                "par_rate": [0.04, 0.042, 0.043, 0.044, 0.045, 0.046],
            }
        ),
    )
    snap.save(
        "usd_forecast_basis",
        pd.DataFrame({"tenor_years": [1, 2, 3, 7, 10], "basis_bp": [1.5, 2.0, 2.5, 3.5, 4.0]}),
    )

    with pytest.raises(CurveDataError, match="5\\.0"):
        usd_forecast_curve(snap, ASOF)


def test_sek_benchmark_yields_are_par_yield_inputs(snapshot: Snapshot) -> None:
    """MKT-05 (build.py portion): Riksbank benchmark yields enter as par-yield
    inputs — the quoted yield is used as the coupon of a par bond — and that
    documented mapping reprices exactly."""
    from yieldcurve.curves.build import sek_government_quotes

    quotes = sek_government_quotes(snapshot, ASOF)
    benchmarks = [q for q in quotes if isinstance(q.instrument, FixedCouponBond)]
    assert benchmarks
    for quote in benchmarks:
        bond = quote.instrument
        assert isinstance(bond, FixedCouponBond)
        assert bond.coupon == quote.rate  # the quoted yield IS the par coupon

    report = repricing_report(sek_government_curve(snapshot, ASOF), quotes, asof=ASOF)
    assert all(row.ok for row in report), [row for row in report if not row.ok]


def test_discount_and_forecast_are_genuinely_different_curves(snapshot: Snapshot) -> None:
    curves = usd_curveset(snapshot, ASOF)

    discount_5y = curves.discount.zero(5.0)
    forecast_5y = curves.forecast_for("3M").zero(5.0)

    assert abs(forecast_5y - discount_5y) > 1e-4


def test_forecast_curve_lies_above_the_ois_curve(snapshot: Snapshot) -> None:
    ois = usd_ois_curve(snapshot, ASOF)
    forecast = usd_forecast_curve(snapshot, ASOF)

    spreads = [forecast.zero(t) - ois.zero(t) for t in (1.0, 2.0, 5.0, 10.0)]

    assert all(s > 0.0 for s in spreads), spreads


def test_forecast_for_an_unknown_tenor_names_what_is_available(snapshot: Snapshot) -> None:
    curves = usd_curveset(snapshot, ASOF)

    with pytest.raises(KeyError, match="6M"):
        curves.forecast_for("6M")


@pytest.mark.parametrize("method", list(InterpMethod))
def test_non_canonical_methods_build_as_overlays_on_canonical_nodes(
    snapshot: Snapshot, method: InterpMethod
) -> None:
    canonical = usd_ois_curve(snapshot, ASOF)
    overlay = overlay_curve(canonical, method)

    assert overlay.times == canonical.times
    assert overlay.dfs == canonical.dfs
    assert overlay.df(10.0) < overlay.df(1.0) < 1.0


def test_sek_curve_covers_the_key_rate_grid(snapshot: Snapshot) -> None:
    curve = sek_government_curve(snapshot, ASOF)

    assert max(curve.times) == pytest.approx(10.0, abs=0.6)
    assert all(curve.df(t) > 0.0 for t in (0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0))


def test_basis_is_defined_at_every_requested_tenor(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import government_swap_basis

    tenors = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
    basis = government_swap_basis(snapshot, ASOF, tenors)

    assert set(basis) == set(tenors)
    assert all(isinstance(v, float) for v in basis.values())


def test_basis_is_of_a_plausible_magnitude(snapshot: Snapshot) -> None:
    from yieldcurve.curves.build import government_swap_basis

    basis = government_swap_basis(snapshot, ASOF, (2.0, 5.0, 10.0, 30.0))

    assert all(abs(v) < 0.015 for v in basis.values()), basis


def test_basis_compares_zeros_not_a_zero_against_a_par_yield(
    snapshot: Snapshot,
) -> None:
    from yieldcurve.curves.build import government_swap_basis, usd_government_curve

    gov = usd_government_curve(snapshot, ASOF)
    ois = usd_ois_curve(snapshot, ASOF)
    basis = government_swap_basis(snapshot, ASOF, (10.0,))

    assert basis[10.0] == pytest.approx(ois.zero(10.0) - gov.zero(10.0), abs=1e-12)


def test_missing_dataset_raises_a_named_error(snapshot: Snapshot, tmp_path: Path) -> None:
    empty = Snapshot(date=ASOF, root=Path(str(tmp_path)))

    with pytest.raises((CurveDataError, FileNotFoundError)):
        usd_ois_curve(empty, ASOF)
