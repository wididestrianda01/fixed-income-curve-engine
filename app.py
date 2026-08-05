"""A Swedish government curve, and what it can and cannot tell you.

Entry point. Builds the sidebar, then routes to four tabs. Initialization is guarded: if
the packaged snapshot is missing or unreadable, the app stops with a sanitized recovery
message before any tab renders. Each tab is rendered inside a handler that turns a known
library exception into an st.error rather than a traceback — technical detail goes to the
server log, and the browser only ever sees a message with no paths and no stack traces.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import streamlit as st

from app.data import SNAPSHOT_DATE, load_snapshot
from app.state import AppState
from app.tabs import beyond, curve, pricing, risk
from yieldcurve.curves.build import CurveDataError
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.parametric import FitError
from yieldcurve.curves.protocol import MissingFixingError
from yieldcurve.market.snapshot import MissingDatasetError
from yieldcurve.models.hullwhite import CalibrationError
from yieldcurve.risk.portfolio import PortfolioError
from yieldcurve.risk.scenarios import ScenarioConfigError

RENDER_ERRORS = (
    CurveDataError,
    MissingFixingError,
    MissingDatasetError,
    ScenarioConfigError,
    CalibrationError,
    FitError,
    PortfolioError,
)

# Fatal at startup: the packaged snapshot is unusable. The loader converts the
# packaged-resource failures it detects into MissingDatasetError, but a corrupt-but-
# present resource can still escape that conversion as a raw OSError (a read failure
# inside the TOML parser). A partial install can also surface a ModuleNotFoundError
# from a lazy import reached at build_sidebar time — a wholly absent yieldcurve would
# fail earlier, at app.py's module-level imports, outside this try. The trailing
# Exception is the documented fallback:
# initialization must never leak a raw traceback to the browser, so any failure here
# shows the same sanitized recovery message and logs full detail server-side.
_SNAPSHOT_INIT_ERRORS: tuple[type[BaseException], ...] = (
    MissingDatasetError,
    ModuleNotFoundError,
    OSError,
    Exception,
)

_LOGGER = logging.getLogger("app")

# Local filesystem locations must never reach the browser (SEC-06). Some domain error
# messages embed paths (PortfolioError carries the TOML path; an external-root
# MissingDatasetError carries the target path), so anything displayed is run through this
# sanitizer: absolute POSIX paths (/tmp, /usr/local/bin), Windows drive paths (C:\x,
# C:\Users\me), ~-home prefixes (~/bin), and relative paths with two or more components
# (data/demo_portfolio.toml) become "[path]". Currency pairs (USD/SEK) and number
# fractions, dates, and percentages (1/2, 12.5/1.0, 2026/07/24, 10%/20%) are not paths
# and survive. The full message still reaches the server log.
_PATH_TOKEN = re.compile(
    r"(?:"
    # absolute POSIX, Windows drive, and ~-home paths: a prefix plus 1+ components
    r"(?<![A-Za-z0-9._~%-])(?:[A-Za-z]:[\\/]|~?/)[A-Za-z0-9._~-]+(?:[\\/][A-Za-z0-9._~-]+)*(?<!\.)"
    r"|"
    # relative paths with 2+ components; a leading all-caps-three-letter (currency) or
    # numeric (fraction/date) component excludes the token from this branch
    r"(?<![A-Za-z0-9._~-])(?!(?:[A-Z]{3}|\d+(?:\.\d+)?)/[A-Za-z0-9._~-]+(?![A-Za-z0-9]))"
    r"[A-Za-z0-9._~-]+[\\/][A-Za-z0-9._~-]+(?:[\\/][A-Za-z0-9._~-]+)*(?![A-Za-z0-9])(?<!\.)"
    r")"
)

_METHOD_LABELS = {
    InterpMethod.MONOTONE_CONVEX: "Monotone convex (default)",
    InterpMethod.CUBIC_LOG_DF: "Cubic on log discount factors",
    InterpMethod.LOG_LINEAR_DF: "Log-linear on discount factors",
}


def _sanitize(text: str) -> str:
    """Remove path-like tokens so no local filesystem location reaches the browser."""
    return _PATH_TOKEN.sub("[path]", text)


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
    load_snapshot()  # fail fast: validate the packaged snapshot before any tab renders
    return AppState(asof=SNAPSHOT_DATE, method=method)


def render_guarded(render: Callable[[AppState], None], state: AppState) -> None:
    """Render one tab; turn a known domain failure into a sanitized message.

    The full exception (type, message, traceback) is logged server-side with technical
    detail; the browser sees only the type name and a path-free message.
    """
    try:
        render(state)
    except RENDER_ERRORS as exc:
        _LOGGER.exception("Known app error while rendering %s: %s", type(exc).__name__, exc)
        st.error(f"{type(exc).__name__}: {_sanitize(str(exc))}")


def main() -> None:
    st.set_page_config(
        page_title="Fixed income yield curve engine",
        page_icon="📈",
        layout="wide",
    )
    st.title("A Swedish government curve, and what it can and cannot tell you")
    try:
        state = build_sidebar()
    except _SNAPSHOT_INIT_ERRORS:
        _LOGGER.exception("App initialization failed: the packaged snapshot is unusable")
        st.error(
            f"The packaged market-data snapshot ({SNAPSHOT_DATE.isoformat()}) is missing "
            "or unreadable, so the app cannot start. This app runs entirely on one "
            "committed, read-only snapshot that ships with the package. Reinstall the "
            "package — for example `uv sync --frozen --extra app` — or restore the "
            "packaged snapshot resources, then restart the app. Technical details were "
            "logged server-side."
        )
        return
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


if __name__ == "__main__":
    main()
