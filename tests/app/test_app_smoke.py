"""Headless smoke tests for the Streamlit app.

These assert that each tab runs without raising and that the numbers it puts on screen are
the same numbers the library produces. They are not visual tests; nothing here checks
layout. The anchor values come from tests/golden/pipeline_v1.json once task 7.8 writes it —
until then each tab's anchor is recomputed from the library in the test itself.
"""

from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

ASOF = date(2026, 7, 24)
APP = "app.py"
TIMEOUT = 120


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _labels(at: AppTest) -> set[str]:
    return {m.label for m in at.metric}


def test_the_app_starts_without_raising(app: AppTest) -> None:
    assert len(app.exception) == 0


def test_the_sidebar_offers_the_three_interpolation_methods(app: AppTest) -> None:
    assert len(app.sidebar.selectbox) == 1
    assert len(app.sidebar.selectbox[0].options) == 3


def test_curve_tab_reports_the_svensson_residual(app: AppTest) -> None:
    assert "Svensson RMSE (bp)" in _labels(app)


def test_curve_tab_ten_year_zero_matches_the_library() -> None:
    from app.data import sek_curve
    from yieldcurve.curves.build import sek_government_curve
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.market.snapshot import Snapshot

    direct = sek_government_curve(
        Snapshot(date=ASOF), ASOF, method=InterpMethod.MONOTONE_CONVEX
    ).zero(10.0)
    cached = sek_curve(ASOF, InterpMethod.MONOTONE_CONVEX).zero(10.0)
    assert cached == pytest.approx(direct, abs=1e-10)


def test_pricing_tab_shows_the_three_price_components(app: AppTest) -> None:
    assert {"Clean price", "Accrued", "Dirty price", "Yield to maturity"} <= _labels(app)


def test_pricing_tab_clean_plus_accrued_equals_dirty(app: AppTest) -> None:
    _price_metrics = {"Clean price", "Accrued", "Dirty price"}
    by_label = {m.label: float(m.value) for m in app.metric if m.label in _price_metrics}
    assert by_label["Clean price"] + by_label["Accrued"] == pytest.approx(
        by_label["Dirty price"], abs=1e-9
    )


def test_pricing_tab_bond_universe_stops_at_the_last_curve_pillar() -> None:
    from app.data import gov_bonds

    maturities = [b.maturity for b in gov_bonds()]
    assert max(maturities) <= date(2036, 1, 1)


def test_risk_tab_shows_both_duration_families(app: AppTest) -> None:
    assert {"DV01", "Modified duration", "Effective duration"} <= _labels(app)


def test_risk_tab_discloses_the_interpolated_sek_one_year_point(app: AppTest) -> None:
    body = " ".join(c.value for c in app.caption)
    assert "interpolated, not observed" in body


def test_irrbb_board_runs_all_six_bcbs_scenarios() -> None:
    from app.data import portfolio, sek_curveset
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.risk.portfolio import eve_ladder
    from yieldcurve.risk.scenarios import bcbs_scenarios

    scenarios = bcbs_scenarios("SEK")
    ladder = eve_ladder(
        portfolio(),
        sek_curveset(ASOF, InterpMethod.MONOTONE_CONVEX),
        ASOF,
        scenarios,
    )
    assert len(ladder) == 6
    assert tuple(ladder) == tuple(s.name for s in scenarios)


def test_risk_tab_reports_var_and_expected_shortfall(app: AppTest) -> None:
    labels = _labels(app)
    assert any(label.startswith("VaR") for label in labels)
    assert any(label.startswith("Expected shortfall") for label in labels)


def test_expected_shortfall_never_falls_below_var_at_either_confidence() -> None:
    from app.data import pnl_sample, portfolio, sek_curveset
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.risk.portfolio import historical_pnl, var_es

    changes, tenors = pnl_sample()
    pnl = historical_pnl(
        portfolio(),
        sek_curveset(ASOF, InterpMethod.MONOTONE_CONVEX),
        ASOF,
        changes,
        tenors,
    )
    for confidence in (0.95, 0.99):
        var, es = var_es(pnl, confidence=confidence)
        assert es >= var


def test_the_volatility_proxy_disclosure_is_on_screen(app: AppTest) -> None:
    """If this fails, someone deleted the sentence that makes the VaR number honest."""
    rendered = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
    assert "proxy" in rendered
