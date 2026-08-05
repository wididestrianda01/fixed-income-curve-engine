"""Headless smoke tests for the Streamlit app.

These assert that each tab runs without raising and that the numbers it puts on screen are
the same numbers the library produces. They are not visual tests; nothing here checks
layout. The anchor values come from tests/golden/pipeline_v1.json once task 7.8 writes it —
until then each tab's anchor is recomputed from the library in the test itself.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from typing import cast

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


def _app_sanitize(text: str) -> str:
    """Run app.py's path sanitizer. ``import app`` resolves to the app/ package,
    so the entry-point script is loaded under a distinct module name instead."""
    path = Path(__file__).resolve().parents[2] / APP
    spec = importlib.util.spec_from_file_location("app_entry_point", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(str, module._sanitize(text))


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
    from yieldcurve.risk.scenarios import eu_scenarios

    scenarios = eu_scenarios("SEK")
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


def test_beyond_tab_reports_a_finite_government_swap_basis() -> None:
    from yieldcurve.curves.build import government_swap_basis
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.market.snapshot import Snapshot

    basis = government_swap_basis(
        Snapshot(date=ASOF), ASOF, (2.0, 5.0, 10.0), method=InterpMethod.MONOTONE_CONVEX
    )
    assert len(basis) == 3
    assert all(v == v and abs(v) < 1.0 for v in basis.values())


def test_beyond_tab_pca_variance_ratios_descend_and_sum_below_one() -> None:
    from app.data import cmt_history
    from yieldcurve.risk.pca import daily_changes, fit_pca

    changes, tenors = daily_changes(cmt_history())
    result = fit_pca(changes, tenors, n_components=3)
    ratios = list(result.explained_variance_ratio)
    assert len(ratios) == 3
    assert ratios == sorted(ratios, reverse=True)
    assert sum(ratios) <= 1.0 + 1e-12


def test_hull_white_residual_is_strictly_positive_on_the_illustrative_grid() -> None:
    """A zero residual would mean the vols came from the model being fitted."""
    from app.data import load_snapshot, usd_curves
    from yieldcurve.curves.interpolation import InterpMethod
    from yieldcurve.models.hullwhite import atm_swaption_grid, calibrate

    curves = usd_curves(ASOF, InterpMethod.MONOTONE_CONVEX)
    swaptions, vols = atm_swaption_grid(
        load_snapshot(),
        ASOF,
        curves.discount,
        dataset="illustrative_swaption_vols",
    )
    result = calibrate(curves.discount, swaptions, vols, ASOF)
    assert result.rmse_vol_bp > 0.0
    assert 1e-4 < result.a < 2.0
    assert 1e-5 < result.sigma < 0.20


def test_beyond_tab_says_the_volatilities_are_illustrative(app: AppTest) -> None:
    rendered = (
        " ".join(m.value for m in app.markdown)
        + " ".join(w.value for w in app.warning)
        + " ".join(c.value for c in app.caption)
    )
    assert "illustrative" in rendered.lower()
    assert "Information License Agreement" in rendered
    # MKT-04: the disclosure must not claim a market/cme.py module still exists.
    assert "cme.py" not in rendered


def test_the_app_guards_missing_snapshot_before_tabs_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or corrupt packaged snapshot stops the app with a sanitized
    recovery message before any tab renders: no paths, no traceback, no tabs."""
    import app.data as app_data
    import app.tabs.beyond as beyond_mod  # bind tab imports against the real data layer
    from yieldcurve.market.snapshot import MissingDatasetError

    def _missing() -> None:
        raise MissingDatasetError("packaged snapshot manifest: is missing; reinstall the package")

    # beyond.py binds app.data.load_snapshot at import time; importing it before the
    # patch keeps that binding honest for every later test in this process. The patch
    # still reaches app.py: each at.run() re-executes the app script, and its
    # `from app.data import load_snapshot` re-reads the (already-imported) module
    # attribute at execution time, so build_sidebar calls the raising stub.
    assert beyond_mod.load_snapshot is app_data.load_snapshot  # type: ignore[attr-defined]
    monkeypatch.setattr(app_data, "load_snapshot", _missing)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert len(at.exception) == 0
    rendered = " ".join(e.value for e in at.error)
    assert "packaged snapshot" in rendered.lower()
    assert ASOF.isoformat() in rendered  # names the expected packaged snapshot
    assert "reinstall" in rendered.lower()  # a concrete recovery action
    assert "Traceback" not in rendered
    assert "/home" not in rendered  # no local filesystem path in the browser
    assert not at.tabs  # nothing is rendered behind the guard


def test_known_domain_errors_are_sanitized_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Known domain failures show type + sanitized message on screen; the full
    technical detail (including any path) goes to the server-side log only."""
    import logging

    import app.tabs.beyond as beyond_mod
    from yieldcurve.risk.portfolio import PortfolioError

    def _boom(*args: object, **kwargs: object) -> object:
        raise PortfolioError(
            "/home/alice/.cache/yieldcurve/data/demo_portfolio.toml: invalid TOML: "
            "expected '=' after a key (at line 3)"
        )

    # Patch the tab's own binding (an uncached call site) so the failure genuinely
    # surfaces inside a tab render rather than being masked by the data-layer cache.
    monkeypatch.setattr(beyond_mod, "fit_pca", _boom)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    with caplog.at_level(logging.ERROR, logger="app"):
        at.run()
    assert len(at.exception) == 0
    rendered = " ".join(e.value for e in at.error)
    assert "PortfolioError" in rendered
    assert "Traceback" not in rendered
    assert "/home/alice" not in rendered
    assert "demo_portfolio.toml" not in rendered
    assert "/home/alice/.cache/yieldcurve/data/demo_portfolio.toml" in caplog.text


def test_sanitizer_preserves_percent_fractions() -> None:
    """Percentages survive: ``%`` is a token boundary, so the POSIX-prefix
    branch must not eat ``/20`` out of ``10%/20%`` — while real paths still mask."""
    for text in ("10%/20%", "50%/50%", "shift of 10%/20%"):
        assert _app_sanitize(text) == text, text
    assert _app_sanitize("read /tmp/cache.toml") == "read [path]"
    assert _app_sanitize("open data/demo_portfolio.toml") == "open [path]"
    assert _app_sanitize("C:\\Users\\me\\x.toml") == "[path]"


def test_sanitizer_leaves_currency_shaped_relative_paths() -> None:
    """Documented trade-off: ``ABC/def.toml`` is not stripped. The leading
    three-uppercase-letter component is excluded as a currency pair (USD/SEK),
    and that exclusion shields this genuine-looking relative path too. Accepted:
    the sanitizer must not mangle currency text, and the rare ABC/<file> shape
    pays that cost."""
    assert _app_sanitize("ABC/def.toml") == "ABC/def.toml"


def test_hull_white_calibration_second_request_is_a_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expensive calibration is a cached pure output: asking twice must not
    recompute, even outside an app rerun."""
    from typing import Any

    import app.data as app_data
    from yieldcurve.curves.interpolation import InterpMethod

    real: Any = app_data.calibrate  # type: ignore[attr-defined]  # internal global, patched below
    calls = {"n": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(app_data, "calibrate", counting)
    app_data.hullwhite_calibration(ASOF, InterpMethod.MONOTONE_CONVEX)
    # The Streamlit cache is process-global, so the first request is a hit (0 calls)
    # when an earlier test warmed this key and a cold start (1 call) otherwise.
    first = calls["n"]
    app_data.hullwhite_calibration(ASOF, InterpMethod.MONOTONE_CONVEX)
    # The second request must be a cache hit: it adds no calls on top of the first.
    assert calls["n"] == first


def test_expensive_calibration_is_cached_and_the_method_control_is_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hidden-tab work: a rerun without changes reuses the cached calibration,
    and the sidebar interpolation control reaches the Risk tab (global scope)."""
    from typing import Any

    import app.data as app_data
    from yieldcurve.curves.interpolation import InterpMethod

    real: Any = app_data.calibrate  # type: ignore[attr-defined]  # internal global, patched below
    calls = {"n": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(app_data, "calibrate", counting)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    # The Streamlit cache is process-global, so the MONOTONE key may already be warm
    # from earlier tests (0 calls) or cold (1 call) — never more than one.
    n_monotone = calls["n"]
    assert n_monotone <= 1
    captions = " ".join(c.value for c in at.caption)
    assert "Under monotone convex the ladder is additive only to about 1.4%" in captions

    at.sidebar.selectbox[0].select(InterpMethod.CUBIC_LOG_DF)
    at.run()
    # CUBIC is a fresh cache key, so switching methods recalibrates exactly once on
    # top of whatever the MONOTONE run did — regardless of process-wide cache warmth.
    n_cubic = calls["n"]
    assert n_cubic == n_monotone + 1
    captions = " ".join(c.value for c in at.caption)
    assert "Under this smooth scheme the ladder is additive to about 1e-4" in captions
    assert len(at.exception) == 0

    at.run()  # unchanged choice: the rerun must reuse the cached calibration
    assert calls["n"] == n_cubic
    assert len(at.exception) == 0
