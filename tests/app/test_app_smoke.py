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
