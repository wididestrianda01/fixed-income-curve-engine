"""A Swedish government curve, and what it can and cannot tell you.

Entry point. Builds the sidebar, then routes to four tabs. Each tab is rendered inside a
handler that turns a known library exception into an st.error rather than a traceback: a
reader who chooses an interpolation method that cannot fit the data should be told that,
not shown a stack.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from app.data import SNAPSHOT_DATE, load_snapshot
from app.state import AppState
from app.tabs import beyond, curve, pricing, risk
from yieldcurve.curves.build import CurveDataError
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.parametric import FitError
from yieldcurve.market.snapshot import MissingDatasetError
from yieldcurve.models.hullwhite import CalibrationError
from yieldcurve.risk.portfolio import PortfolioError
from yieldcurve.risk.scenarios import ScenarioConfigError

RENDER_ERRORS = (
    CurveDataError,
    MissingDatasetError,
    ScenarioConfigError,
    CalibrationError,
    FitError,
    PortfolioError,
)

_METHOD_LABELS = {
    InterpMethod.MONOTONE_CONVEX: "Monotone convex (default)",
    InterpMethod.CUBIC_LOG_DF: "Cubic on log discount factors",
    InterpMethod.LOG_LINEAR_DF: "Log-linear on discount factors",
}


def build_sidebar() -> AppState:
    """Collect the two choices a reader is allowed to make."""
    st.sidebar.header("Settings")
    st.sidebar.date_input(
        "As of",
        value=SNAPSHOT_DATE,
        disabled=True,
        help="One snapshot is committed to the repository. The app never fetches data.",
    )
    method = st.sidebar.selectbox(
        "Interpolation",
        options=list(_METHOD_LABELS),
        format_func=lambda m: _METHOD_LABELS[m],
    )
    st.sidebar.caption(
        "Monotone convex keeps forwards positive but is not linear in the inputs, so "
        "risk ladders built under it do not add up exactly. The two smooth methods do."
    )
    return AppState(snapshot=load_snapshot(), asof=SNAPSHOT_DATE, method=method)


def render_guarded(render: Callable[[AppState], None], state: AppState) -> None:
    """Render one tab, converting a known data or model failure into a message."""
    try:
        render(state)
    except RENDER_ERRORS as exc:
        st.error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Fixed income yield curve engine",
        page_icon="📈",
        layout="wide",
    )
    st.title("A Swedish government curve, and what it can and cannot tell you")
    state = build_sidebar()
    tab_curve, tab_pricing, tab_risk, tab_beyond = st.tabs(
        ["The curve", "Pricing", "Risk", "Beyond the curve"]
    )
    with tab_curve:
        render_guarded(curve.render, state)
    with tab_pricing:
        render_guarded(pricing.render, state)
    with tab_risk:
        render_guarded(risk.render, state)
    with tab_beyond:
        render_guarded(beyond.render, state)


main()
