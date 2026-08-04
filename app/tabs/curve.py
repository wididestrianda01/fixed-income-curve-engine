"""Tab 1 — building the curve, and why the interpolation scheme is a modelling choice."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import streamlit as st

from app.charts import overlay_figure
from app.data import sek_curve, sek_quotes
from app.state import AppState
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.parametric import Svensson
from yieldcurve.curves.protocol import curve_time

_METHOD_NAMES = {
    InterpMethod.MONOTONE_CONVEX: "Monotone convex",
    InterpMethod.CUBIC_LOG_DF: "Cubic log-DF",
    InterpMethod.LOG_LINEAR_DF: "Log-linear DF",
}
_GRID = np.linspace(0.05, 10.0, 400)
_FORWARD_TENOR = 0.25
_BP = 10_000.0


def render(state: AppState) -> None:
    st.subheader("The curve")
    st.markdown(
        "A bootstrap is not a fit. It is the unique set of discount factors that reprices "
        "every input quote exactly: each Riksbank bill and benchmark yield here returns to "
        "par by construction, to machine precision. What a bootstrap does **not** determine "
        "is what happens between the pillars, and that is a modelling choice — which is what "
        "the two charts below are about."
    )

    selected = st.multiselect(
        "Interpolation methods to overlay",
        options=list(_METHOD_NAMES),
        default=list(_METHOD_NAMES),
        format_func=lambda m: _METHOD_NAMES[m],
    )
    show_svensson = st.checkbox("Show Svensson fit", value=True)
    if not selected:
        st.info("Select at least one interpolation method.")
        return

    curves = {m: sek_curve(state.asof, m) for m in selected}

    zeros = {
        _METHOD_NAMES[m]: (_GRID.tolist(), [c.zero(float(t)) * 100.0 for t in _GRID])
        for m, c in curves.items()
    }
    st.plotly_chart(overlay_figure(zeros, y_title="Zero rate (%)"), use_container_width=True)
    pillars = [curve_time(state.asof, q.instrument.maturity) for q in sek_quotes(state.asof)]  # type: ignore[attr-defined]
    st.caption(
        f"Bootstrap pillars at {', '.join(f'{t:.2f}y' for t in sorted(pillars))}. "
        "Between them, every line is an assumption."
    )

    forwards = {
        _METHOD_NAMES[m]: (
            _GRID.tolist(),
            [c.fwd(float(t), float(t) + _FORWARD_TENOR) * 100.0 for t in _GRID],
        )
        for m, c in curves.items()
    }
    st.plotly_chart(
        overlay_figure(forwards, y_title="3-month forward rate (%)"),
        use_container_width=True,
    )
    st.markdown(
        "The forward curve is where interpolation schemes stop being interchangeable. "
        "Log-linear interpolation on discount factors is continuous in the zeros and "
        "**discontinuous** in the forwards — it sawtooths at every pillar. Monotone convex "
        "keeps the forwards positive and continuous, which is why the library defaults to "
        "it. The price of that is additivity: the scheme's amendment tests are branches, so "
        "risk ladders built under it do not sum exactly (see the Risk tab)."
    )

    if show_svensson:
        times: Sequence[float] = _GRID.tolist()
        base = curves.get(InterpMethod.MONOTONE_CONVEX, next(iter(curves.values())))
        observed: Sequence[float] = [base.zero(float(t)) for t in times]
        fit = Svensson.fit(times, observed, reference_date=state.asof)
        st.metric("Svensson RMSE (bp)", f"{fit.rmse * _BP:.2f}")
        st.plotly_chart(
            overlay_figure(
                {
                    "Bootstrapped": (list(times), [z * 100.0 for z in observed]),
                    "Svensson": (
                        list(times),
                        [fit.curve.zero(float(t)) * 100.0 for t in times],
                    ),
                },
                y_title="Zero rate (%)",
            ),
            use_container_width=True,
        )
        st.caption(
            "Six parameters against the whole curve. The residual is the price of that "
            "parsimony, and it is reported rather than described."
        )
