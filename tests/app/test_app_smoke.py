"""Headless behavioral tests for the Streamlit app.

These assert the rendered surface: each tab runs without raising, the numbers on screen
are the same numbers the library produces, controls change the rendered values, tables and
unit labels render, and the known failure modes show their recovery text. Layout and DOM
accessibility (true focus order, aria attributes, 390 px viewport rendering) are browser
checks owned by Task 26; this suite pins the structural preconditions (labeled controls,
fluid chart/table containers) instead.

Runtime warning: this module takes roughly 13 minutes to run. Every AppTest run
re-executes the whole app (all four tabs), each under the TIMEOUT=120 s per-run bound,
and the interaction tests each run the app two or three times — the cost is dominated by
full app executions, not by individual assertions. Budget accordingly in CI.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
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


def _widget(at: AppTest, kind: str, label: str) -> object:
    """The first widget of ``kind`` labelled ``label``, anywhere in the tree.

    AppTest exposes widget parameters (label, disabled, help, options) from the
    protobuf; the main tree also contains the sidebar widgets, so lookups by
    label are unambiguous regardless of where the control lives.
    """
    for element in getattr(at, kind):
        if getattr(element, "label", None) == label:
            return element
    raise AssertionError(f"no {kind} widget labelled {label!r} rendered")


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


def test_every_top_level_tab_renders_without_exception(app: AppTest) -> None:
    """Streamlit executes every tab body on every run, so a clean run proves
    each top-level section renders. Pin the four tabs and their section
    headings explicitly so a dropped or renamed tab fails this test."""
    assert [t.label for t in app.tabs] == [
        "The curve",
        "Pricing",
        "Risk",
        "Beyond the curve",
    ]
    assert all(len(t.subheader) >= 1 for t in app.tabs)
    assert len(app.exception) == 0


def test_the_sidebar_offers_the_three_interpolation_methods(app: AppTest) -> None:
    assert len(app.sidebar.selectbox) == 1
    assert len(app.sidebar.selectbox[0].options) == 3


def test_the_date_control_is_pinned_to_the_committed_snapshot(app: AppTest) -> None:
    """The only date selector is the 'As of' control, and it is deliberately
    non-interactive: one committed, read-only snapshot and no fetching. The
    control displays the snapshot date, is disabled, and carries the
    never-fetches help text a reader sees as the control's tooltip."""
    date_input = app.sidebar.date_input[0]
    assert date_input.label == "As of"
    assert date_input.value == ASOF
    assert date_input.disabled is True
    assert "never fetches data" in date_input.help


def test_bond_selector_changes_the_rendered_price_metrics() -> None:
    """Selecting a different bond on the Pricing tab reprices the rendered
    metrics, and the Risk tab follows the selection (session state)."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    bond = _widget(at, "selectbox", "Bond")
    before = {m.label: m.value for m in at.metric}
    bond.select_index(3)  # type: ignore[attr-defined]
    at.run()
    after = {m.label: m.value for m in at.metric}
    assert after["Dirty price (per 100 face)"] != before["Dirty price (per 100 face)"]
    assert after["Clean price (per 100 face)"] != before["Clean price (per 100 face)"]
    captions = " ".join(c.value for c in at.caption)
    assert "SGB 0.125% May-2031" in captions  # Risk tab names the selected bond
    assert len(at.exception) == 0


def test_confidence_radio_changes_the_rendered_var_es_values() -> None:
    """The Risk tab's confidence radio repaints the linearized-delta VaR/ES
    metrics at the chosen confidence with a different rendered value."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    var99 = {m.label: m.value for m in at.metric}["Linearized delta VaR (99%)"]
    radio = _widget(at, "radio", "Confidence")
    radio.set_value(0.95)  # type: ignore[attr-defined]
    at.run()
    by_label = {m.label: m.value for m in at.metric}
    assert "Linearized delta VaR (95%)" in by_label
    assert "Linearized delta ES (95%)" in by_label
    assert by_label["Linearized delta VaR (95%)"] != var99
    assert len(at.exception) == 0


def test_sidebar_interpolation_control_changes_rendered_values_across_tabs() -> None:
    """The global interpolation control reaches every tab: switching the
    sidebar method reprices the Pricing metrics and recalibrates the Beyond
    Hull-White parameters — both rendered values change."""
    from yieldcurve.curves.interpolation import InterpMethod

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    by_label = {m.label: m.value for m in at.metric}
    dirty = by_label["Dirty price (per 100 face)"]
    calibrated_a = by_label["Calibrated a (1/yr)"]
    at.sidebar.selectbox[0].select(InterpMethod.CUBIC_LOG_DF)
    at.run()
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["Dirty price (per 100 face)"] != dirty
    assert by_label["Calibrated a (1/yr)"] != calibrated_a
    assert len(at.exception) == 0


def test_svensson_checkbox_toggles_the_fit_section() -> None:
    """The Curve tab's 'Show Svensson fit' checkbox adds and removes the
    rendered fit metric — the control drives the rendered surface."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    checkbox = _widget(at, "checkbox", "Show Svensson fit")
    checkbox.uncheck()  # type: ignore[attr-defined]
    at.run()
    assert "Svensson RMSE (bp)" not in _labels(at)
    # Re-fetch: element references are bound to the tree that produced them,
    # so a value set on a stale reference would not reach the next run.
    _widget(at, "checkbox", "Show Svensson fit").check()  # type: ignore[attr-defined]
    at.run()
    assert "Svensson RMSE (bp)" in _labels(at)
    assert len(at.exception) == 0


def test_curve_tab_method_multiselect_drives_the_rendered_tables() -> None:
    """Deselecting overlay methods narrows the rendered residual table to the
    remaining method's columns — the calibration-method control changes the
    rendered tables, not just the charts."""
    from yieldcurve.curves.interpolation import InterpMethod

    def residual_table() -> pd.DataFrame:
        for frame in at.dataframe:
            cols = list(frame.value.columns)
            if any("Target rate" in c for c in cols):
                # frame.value is Any without streamlit stubs; the element is a DataFrame.
                return frame.value  # type: ignore[no-any-return]
        raise AssertionError("residual table not rendered")

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    full = residual_table()
    assert len([c for c in full.columns if c.startswith("Residual (bp)")]) == 3
    multiselect = _widget(at, "multiselect", "Interpolation methods to overlay")
    multiselect.set_value([InterpMethod.LOG_LINEAR_DF])  # type: ignore[attr-defined]
    at.run()
    single = residual_table()
    residual_cols = [c for c in single.columns if c.startswith("Residual (bp)")]
    assert len(residual_cols) == 1
    assert "Log-linear DF (canonical calibration)" in residual_cols[0]
    assert not any("Monotone convex" in c for c in single.columns)
    assert len(at.exception) == 0


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
    assert {
        "Clean price (per 100 face)",
        "Accrued (per 100 face)",
        "Dirty price (per 100 face)",
        "Yield to maturity (% p.a.)",
    } <= _labels(app)


def test_pricing_tab_clean_plus_accrued_equals_dirty(app: AppTest) -> None:
    _price_metrics = {
        "Clean price (per 100 face)",
        "Accrued (per 100 face)",
        "Dirty price (per 100 face)",
    }
    by_label = {m.label: float(m.value) for m in app.metric if m.label in _price_metrics}
    assert by_label["Clean price (per 100 face)"] + by_label[
        "Accrued (per 100 face)"
    ] == pytest.approx(by_label["Dirty price (per 100 face)"], abs=1e-9)


def test_pricing_tab_bond_universe_stops_at_the_last_curve_pillar() -> None:
    from app.data import gov_bonds

    maturities = [b.maturity for b in gov_bonds()]
    assert max(maturities) <= date(2036, 1, 1)


def test_risk_tab_shows_both_duration_families(app: AppTest) -> None:
    labels = _labels(app)
    assert any(label.startswith("DV01") and "1 bp" in label for label in labels)
    assert {"Modified duration (years)", "Effective duration (years)"} <= labels


def test_risk_tab_discloses_the_interpolated_sek_one_year_point(app: AppTest) -> None:
    body = " ".join(c.value for c in app.caption)
    assert "interpolated, not observed" in body


def test_risk_tab_renders_all_six_eu_2024_856_scenarios(app: AppTest) -> None:
    """The six EU 2024/856 shocks run and are presented on screen, in the EBA
    template order: the ΔEVE data table lists every scenario name with its
    illustrative ΔEVE in SEK, in the order the regulation's Annex presents
    them (parallel up/down, short up/down, steepener, flattener — the order
    ``eu_scenarios`` returns and the app renders via ``list(ladder)``).
    (Task 8 renamed the old ``test_irrbb_board_runs_all_six_bcbs_scenarios``;
    the source-level ladder math stays in tests/risk.)"""
    from yieldcurve.risk.scenarios import eu_scenarios

    names = [s.name for s in eu_scenarios("SEK")]
    assert len(names) == 6
    table = next(
        d.value
        for d in app.dataframe
        if "Scenario" in list(d.value.columns)
        and "Illustrative ΔEVE (SEK)" in list(d.value.columns)
    )
    assert len(table) == 6
    assert list(table["Scenario"]) == names


def test_risk_tab_reports_var_and_expected_shortfall(app: AppTest) -> None:
    labels = _labels(app)
    assert any(label.startswith("Linearized delta VaR") for label in labels)
    assert any(label.startswith("Linearized delta ES") for label in labels)


def test_risk_tab_renders_the_loss_tail_direction_convention(app: AppTest) -> None:
    """TQ-06 app portion: the on-screen copy states the tail convention —
    losses are positive magnitudes and expected shortfall never falls below
    VaR — so a reader cannot misread the sign of the reported numbers. The
    source-level math (ES >= VaR on the actual sample) stays in tests/risk."""
    captions = " ".join(c.value for c in app.caption)
    assert "positive loss" in captions  # DV01 caption: loss magnitudes, not signs
    assert "Expected shortfall is the mean of the tail beyond VaR" in captions
    assert "never be the smaller of the two" in captions  # ES >= VaR, rendered


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
    assert "licensed" in rendered.lower()
    # MKT-04/MKT-15: no stale CME redistribution claim remains on screen.
    assert "CME" not in rendered
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


def test_krd_failure_degrades_to_an_info_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-PV instrument (par swap) makes normalized KRD undefined: the Risk
    tab must explain via st.info and keep rendering — never raise."""
    import app.tabs.risk as risk_mod

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("normalized KRD is undefined for a zero-PV instrument")

    monkeypatch.setattr(risk_mod, "krd", _boom)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert len(at.exception) == 0
    rendered = " ".join(i.value for i in at.info)
    assert "Key-rate duration is undefined" in rendered
    assert "par swap" in rendered
    # the tab continues past the guard: the par-rate ladder section still renders
    captions = " ".join(c.value for c in at.caption)
    assert "par-rate ladder" in captions


def test_svensson_fit_failure_degrades_to_a_warning_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected Svensson fit must degrade to an st.warning and skip the fit
    section — the bootstrapped curve stands alone, the tab does not crash."""
    import app.tabs.curve as curve_mod
    from yieldcurve.curves.parametric import FitError

    class _RejectedFit:
        @classmethod
        def fit(cls, *args: object, **kwargs: object) -> object:
            raise FitError("optimizer did not converge")

    monkeypatch.setattr(curve_mod, "Svensson", _RejectedFit)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert len(at.exception) == 0
    rendered = " ".join(w.value for w in at.warning)
    assert "The Svensson fit was rejected" in rendered
    assert "skipped rather than reported" in rendered


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
    """Hidden-tab work: the app opens on the canonical log-linear method, a
    rerun without changes reuses the cached calibration, and the sidebar
    interpolation control reaches the Risk tab (global scope)."""
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
    # The app opens canonical: the sidebar default is LOG_LINEAR_DF (not the
    # first overlay), and the Risk tab renders the smooth additivity caption.
    assert at.sidebar.selectbox[0].value == InterpMethod.LOG_LINEAR_DF
    # The Streamlit cache is process-global, so the LOG_LINEAR key may already
    # be warm from earlier tests (0 calls) or cold (1 call) — never more than one.
    n_first = calls["n"]
    assert n_first <= 1
    captions = " ".join(c.value for c in at.caption)
    assert "Under this smooth scheme the ladder is additive to about 1e-4" in captions

    at.run()  # unchanged choice: the rerun must reuse the cached calibration
    assert calls["n"] == n_first

    at.sidebar.selectbox[0].select(InterpMethod.MONOTONE_CONVEX)
    at.run()
    # The sidebar control reaches the Risk tab: the additivity caption flips to
    # the monotone wording. (MONOTONE may already be cached from the direct
    # calibration tests above, so the switch adds at most one fresh call.)
    n_monotone = calls["n"]
    assert n_monotone <= n_first + 1
    captions = " ".join(c.value for c in at.caption)
    assert "Under monotone convex the ladder is additive only to about 1.4%" in captions
    assert len(at.exception) == 0

    at.run()  # unchanged choice: the rerun must reuse the cached calibration
    assert calls["n"] == n_monotone
    assert len(at.exception) == 0


def _rendered(app: AppTest) -> str:
    """Every text element the browser shows, joined — the rendered-value surface."""
    return (
        " ".join(m.value for m in app.markdown)
        + " ".join(c.value for c in app.caption)
        + " ".join(w.value for w in app.warning)
        + " ".join(i.value for i in app.info)
    )


def test_sidebar_labels_the_interpolation_control_scope(app: AppTest) -> None:
    """The global interpolation control states its actual scope in its label."""
    assert "all tabs" in app.sidebar.selectbox[0].label


def test_curve_tab_distinguishes_canonical_calibration_from_overlays(app: AppTest) -> None:
    rendered = _rendered(app)
    assert "canonical" in rendered
    assert "overlay" in rendered


def test_curve_tab_shows_quote_repricing_residuals(app: AppTest) -> None:
    """Task 4's repricing_report renders as a visible table with residual columns."""
    columns = [list(d.value.columns) for d in app.dataframe]
    assert any(any("Residual (bp" in c for c in col) for col in columns), columns
    assert any(any("Target rate" in c for c in col) for col in columns), columns


def test_curve_tab_grid_expander_table_carries_unit_columns(app: AppTest) -> None:
    """The zero/forward grid expander is the chart's text alternative and its
    columns carry units (percent for zeros, 3M forward rate percent)."""
    grid = next(
        (
            list(d.value.columns)
            for d in app.dataframe
            if list(d.value.columns) and next(iter(d.value.columns)) == "Maturity (y)"
        ),
        None,
    )
    assert grid is not None, "the grid expander table did not render"
    assert any(c.startswith("Zero (%)") for c in grid)
    assert any(c.startswith("3M fwd (%)") for c in grid)


def test_curve_tab_states_the_svensson_fit_target(app: AppTest) -> None:
    rendered = " ".join(c.value for c in app.caption)
    assert "bootstrapped zero rates" in rendered
    assert "grid" in rendered


def test_curve_tab_corrects_the_monotone_convex_positivity_claim(app: AppTest) -> None:
    """The library's monotone-convex overlay omits the Hagan-West positivity
    amendment and can represent negative forwards — the app must mirror that."""
    rendered = _rendered(app)
    assert "negative forwards" in rendered
    assert "keeps the forwards positive" not in rendered


def test_curve_tab_empty_selection_shows_an_empty_state() -> None:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.multiselect[0].set_value([])
    at.run()
    rendered = " ".join(i.value for i in at.info)
    assert "Select at least one interpolation method" in rendered
    assert len(at.exception) == 0


def test_risk_tab_is_illustrative_delta_eve_not_the_irrbb_board(app: AppTest) -> None:
    rendered = _rendered(app)
    # Δ (U+0394) lowercases to δ, so do not lowercase the whole string for this check.
    assert "illustrative ΔEVE" in rendered
    assert "IRRBB board" not in rendered
    assert "debt office" not in rendered
    assert "cannot be too small" not in rendered


def test_risk_tab_discloses_the_invented_capital_denominator(app: AppTest) -> None:
    rendered = _rendered(app)
    assert "not regulatory capital" in rendered
    assert "outlier" not in rendered.lower()
    assert not any("Tier 1" in label for label in _labels(app))


def test_risk_tab_uses_current_eu_citations_only(app: AppTest) -> None:
    rendered = _rendered(app)
    assert "2024/856" in rendered
    assert "BCBS" not in rendered
    assert "d368" not in rendered
    assert "GL/2018" not in rendered


def test_risk_tab_var_es_are_named_linearized_delta_proxy(app: AppTest) -> None:
    rendered = _rendered(app)
    assert "linearized delta" in rendered.lower()
    assert "proxy" in rendered


def test_beyond_tab_discloses_constructed_curves(app: AppTest) -> None:
    rendered = _rendered(app)
    assert "constructed" in rendered.lower()
    assert "not observed" in rendered.lower()


def test_beyond_tab_metrics_carry_units(app: AppTest) -> None:
    labels = _labels(app)
    assert any("1/yr" in label for label in labels)
    assert any("1/√yr" in label for label in labels)
    assert any(label.startswith("Residual (bp") for label in labels)


def test_risk_tab_ladder_expander_tables_carry_units(app: AppTest) -> None:
    """The ladder expander is the chart text alternative and its columns carry
    the Task 7 units: KRD in price-bp per yield-bp, par ladder per 100 face
    per 1 bp."""
    columns = [list(d.value.columns) for d in app.dataframe]
    assert any("KRD (price-bp per yield-bp)" in cols for cols in columns), columns
    assert any("Par-rate delta (per 100 face per 1 bp)" in cols for cols in columns), columns


def test_beyond_tab_hull_white_table_carries_bp_units(app: AppTest) -> None:
    """The calibrated vol table (expiry x swap maturity grid) carries bp units
    on every vol column and names the model difference explicitly."""
    columns = [list(d.value.columns) for d in app.dataframe]
    target = next((cols for cols in columns if "Illustrative vol (bp)" in cols), None)
    assert target is not None, columns
    assert {"Expiry", "Swap maturity", "Model vol (bp)", "Difference (bp)"} <= set(target)


def test_risk_tab_renders_illustrative_delta_eve_units(app: AppTest) -> None:
    """APP-UX-010/REGULATION-23 smoke portions: the ΔEVE comparison is named
    illustrative and every presented value carries SEK units — metric and
    data-table column (pinned rendered surface). The chart axis title is a
    figure contract pinned in tests/app/test_charts.py, not here."""
    rendered = _rendered(app)
    assert "illustrative ΔEVE" in rendered
    labels = _labels(app)
    assert any(label.startswith("Worst-case illustrative ΔEVE (SEK)") for label in labels)
    columns = [list(d.value.columns) for d in app.dataframe]
    assert any("Illustrative ΔEVE (SEK)" in cols for cols in columns), columns


def test_scenario_config_error_degrades_to_a_sanitized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt scenario configuration is representative invalid data: the
    Risk tab shows the named error, sanitized (no path, no traceback), and the
    other tabs keep rendering behind the guarded tab."""
    import app.tabs.risk as risk_mod
    from yieldcurve.risk.scenarios import ScenarioConfigError

    def _bad_config(*args: object, **kwargs: object) -> object:
        raise ScenarioConfigError(
            "/opt/yieldcurve/scenarios.toml: missing 'severe_parallel_up' row"
        )

    monkeypatch.setattr(risk_mod, "eu_scenarios", _bad_config)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert len(at.exception) == 0
    rendered = " ".join(e.value for e in at.error)
    assert "ScenarioConfigError" in rendered
    assert "[path]" in rendered  # the embedded path is stripped
    assert "/opt" not in rendered
    assert "Traceback" not in rendered
    assert "Svensson RMSE (bp)" in _labels(at)  # the Curve tab still renders


def test_all_interactive_controls_are_labeled_and_reachable() -> None:
    """Keyboard-reachability structural proxy: every interactive control
    rendered anywhere in the tree — sidebar, tabs, expanders, or the main
    body outside any tab — has a non-empty label and is enabled; the only
    disabled control is the pinned 'As of' date. The root-level ``at.get``
    enumerates the whole rendered tree, so a control placed outside the
    sidebar and the tabs (a per-container walk would miss it) still fails
    this test. True DOM focus order, aria attributes, and the 390 px
    viewport journey are Task 26 browser checks."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    widgets: list[tuple[str, object]] = []
    for kind in ("selectbox", "radio", "slider", "checkbox", "multiselect", "date_input"):
        widgets.extend((kind, element) for element in at.get(kind))
    assert widgets, "no controls rendered — the app surface changed"
    for kind, widget in widgets:
        label = getattr(widget, "label", "")
        assert label, f"{kind} control rendered without a label"
        if label == "As of":
            assert widget.disabled is True  # type: ignore[attr-defined]
        else:
            assert getattr(widget, "disabled", False) is False, label
    assert len(at.exception) == 0


def test_chart_and_table_containers_are_fluid_for_narrow_screens() -> None:
    """390 px first-screen overflow proxy (structural): every chart and data
    table renders in a fluid container (use_container_width=True or
    width='stretch', including module-level constants bound to those values),
    and no st.columns/st.container call pins an explicit width — neither a
    numeric ``width=`` keyword nor a numeric widths list passed positionally
    to st.columns (``st.columns(3)`` is a column count, not a pinned width).
    The real 390x844 viewport journey — first-screen fit, keyboard access,
    horizontal overflow — is owned by Task 26; this test guards the app-level
    preconditions that journey depends on."""
    app_root = Path(__file__).resolve().parents[2]
    sources = [app_root / "app.py", *sorted((app_root / "app" / "tabs").glob("*.py"))]
    offenders: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "st"):
                continue
            name = node.func.attr
            if name in ("plotly_chart", "dataframe"):
                if not _uses_fluid_width(node, constants):
                    offenders.append(f"{path.name}:{node.lineno} {name} not fluid")
            elif name in ("columns", "container") and _pinned_width(node, constants):
                offenders.append(f"{path.name}:{node.lineno} {name} pinned width")
    assert not offenders, offenders


def _module_constants(tree: ast.Module) -> dict[str, object]:
    """Module-level name -> literal value, resolved by a static pass over one file.

    Lets the proxy see through ``WIDTHS = [300, 200]; st.columns(WIDTHS)`` and
    ``FLUID = True; st.plotly_chart(..., use_container_width=FLUID)`` without
    executing anything. Assignments whose value is not a literal are skipped.
    """
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                with contextlib.suppress(ValueError, SyntaxError):
                    values[target.id] = ast.literal_eval(node.value)
    return values


def _literal_value(node: ast.AST, constants: dict[str, object]) -> object:
    """The literal ``node`` denotes: the node itself, a module-level name, or
    a compound literal (list/tuple) built from constants."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _pinned_width(node: ast.Call, constants: dict[str, object]) -> bool:
    """True when an st.columns/st.container call pins an explicit width.

    Catches a numeric ``width=`` keyword (literal or module-level constant)
    and, for st.columns, a positional widths spec such as
    ``st.columns([300, 200])`` (literal or module-level). A bare int spec
    (``st.columns(3)``) is a column count and is not a pinned width.
    """
    if not isinstance(node.func, ast.Attribute):
        return False
    for kw in node.keywords:
        if kw.arg == "width" and isinstance(_literal_value(kw.value, constants), (int, float)):
            return True
    if node.func.attr == "columns" and node.args:
        spec = _literal_value(node.args[0], constants)
        if isinstance(spec, (list, tuple)) and all(isinstance(item, (int, float)) for item in spec):
            return True
    return False


def _uses_fluid_width(node: ast.Call, constants: dict[str, object]) -> bool:
    """True when the call passes use_container_width=True or width='stretch',
    including via module-level constants bound to those values."""
    for kw in node.keywords:
        if kw.arg is None:
            continue
        if kw.arg == "use_container_width":
            if _literal_value(kw.value, constants) is True:
                return True
        elif kw.arg == "width" and _literal_value(kw.value, constants) == "stretch":
            return True
    return False
