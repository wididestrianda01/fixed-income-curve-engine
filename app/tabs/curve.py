"""Tab 1 — building the curve, and why the interpolation scheme is a modelling choice."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import overlay_figure
from app.data import sek_curve, sek_quotes
from app.state import AppState
from yieldcurve.curves.bootstrap import repricing_report
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.parametric import FitError, Svensson
from yieldcurve.curves.protocol import curve_time

_METHOD_NAMES = {
    InterpMethod.MONOTONE_CONVEX: "Monotone convex (comparative overlay)",
    InterpMethod.CUBIC_LOG_DF: "Cubic log-DF (comparative overlay)",
    InterpMethod.LOG_LINEAR_DF: "Log-linear DF (canonical calibration)",
}
_GRID = np.linspace(0.05, 10.0, 400)
_FORWARD_TENOR = 0.25
_BP = 10_000.0


def render(state: AppState) -> None:
    st.subheader("The curve")
    st.markdown(
        "A bootstrap is not a fit: it is the set of discount factors that reprices every "
        "input quote. The **canonical calibration here is log-linear on discount "
        "factors** — under it each Riksbank bill and benchmark yield returns to its "
        "quoted rate to the documented tolerance, and that is verified mechanically. "
        "Cubic log-DF and monotone convex are **comparative overlays** that "
        "re-interpolate the same knots: they do not reprice the quotes exactly, and "
        "their residuals are measured and shown below. What a bootstrap does **not** "
        "determine is what happens between the pillars, and that is a modelling choice "
        "— which is what the two charts below are about."
    )

    selected = st.multiselect(
        "Interpolation methods to overlay",
        options=list(_METHOD_NAMES),
        default=list(_METHOD_NAMES),
        format_func=lambda m: _METHOD_NAMES[m],
        help=(
            "Log-linear DF is the package's canonical calibration. Cubic log-DF and "
            "monotone convex are comparative overlays built on the same knots; their "
            "final quote residuals are reported in the table below."
        ),
    )
    show_svensson = st.checkbox("Show Svensson fit", value=True)
    if not selected:
        st.info("Select at least one interpolation method.")
        return

    curves = {m: sek_curve(state.asof, m) for m in selected}
    quotes = sek_quotes(state.asof)
    pillars = sorted(curve_time(state.asof, q.instrument.maturity) for q in quotes)  # type: ignore[attr-defined]

    zeros = {
        _METHOD_NAMES[m]: (_GRID.tolist(), [c.zero(float(t)) * 100.0 for t in _GRID])
        for m, c in curves.items()
    }
    st.plotly_chart(
        overlay_figure(zeros, y_title="Zero rate (%)", pillar_times=pillars),
        use_container_width=True,
    )
    st.caption(
        f"Calibration pillars at {', '.join(f'{t:.2f}y' for t in pillars)} (dotted "
        "lines). Between them, every line is an interpolation assumption."
    )

    forwards = {
        _METHOD_NAMES[m]: (
            _GRID.tolist(),
            [c.fwd(float(t), float(t) + _FORWARD_TENOR) * 100.0 for t in _GRID],
        )
        for m, c in curves.items()
    }
    st.plotly_chart(
        overlay_figure(forwards, y_title="3-month forward rate (%)", pillar_times=pillars),
        use_container_width=True,
    )
    st.markdown(
        "The forward curve is where interpolation schemes stop being interchangeable. "
        "Log-linear interpolation on discount factors is continuous in the zeros and "
        "**discontinuous** in the forwards — it sawtooths at every pillar. Monotone "
        "convex keeps the forwards **continuous**; it is the Hagan-West construction "
        "without the positivity amendment, so it **can represent negative forwards**, "
        "and it is a comparative overlay whose quote residuals are measured, not "
        "asserted to vanish. The price of its region-switching amendments is "
        "additivity: risk ladders built under it do not sum exactly (see the Risk tab)."
    )

    grid_data: dict[str, list[float]] = {"Maturity (y)": _GRID.tolist()}
    for m in selected:
        label = _METHOD_NAMES[m]
        curve = curves[m]
        grid_data[f"Zero (%) — {label}"] = [curve.zero(float(t)) * 100.0 for t in _GRID]
        grid_data[f"3M fwd (%) — {label}"] = [
            curve.fwd(float(t), float(t) + _FORWARD_TENOR) * 100.0 for t in _GRID
        ]
    with st.expander("Zero and 3-month forward grid (data)"):
        st.dataframe(pd.DataFrame(grid_data), use_container_width=True, hide_index=True)

    residual_rows: dict[str, list[object]] = {
        "Instrument": [
            f"{type(q.instrument).__name__} {q.instrument.maturity:%Y-%m-%d}"
            for q in quotes  # type: ignore[attr-defined]
        ],
        "Target rate": [q.rate for q in quotes],
    }
    for m in selected:
        report = repricing_report(curves[m], quotes, state.asof)
        residual_rows[f"Model rate — {_METHOD_NAMES[m]}"] = [r.model_rate for r in report]
        residual_rows[f"Residual (bp) — {_METHOD_NAMES[m]}"] = [r.residual * _BP for r in report]
    st.markdown(
        "**Quote-repricing residuals** — each selected method's final model quote "
        "against its target."
    )
    st.dataframe(pd.DataFrame(residual_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Residual is model minus target rate, in basis points. The canonical log-linear "
        "build stays within the documented 1e-6 bp tolerance for every quote; the "
        "comparative overlays leave measured residuals wherever a payment falls between "
        "knots."
    )

    if show_svensson:
        times: Sequence[float] = _GRID.tolist()
        base = curves.get(InterpMethod.MONOTONE_CONVEX, next(iter(curves.values())))
        observed: Sequence[float] = [base.zero(float(t)) for t in times]
        try:
            fit = Svensson.fit(times, observed, reference_date=state.asof)
        except FitError as exc:
            st.warning(
                f"The Svensson fit was rejected ({exc}). The bootstrapped curve stands "
                "alone; the fit is skipped rather than reported as a number."
            )
            return
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
            "Svensson fits the bootstrapped zero rates on the 0.05y-10y grid (400 "
            "points), unweighted. Six parameters against the whole curve: the RMSE is "
            "the price of that parsimony, reported rather than described."
        )
