"""Tab 4 — three things a single government curve cannot do.

USD, not SEK, and the snapshot is why: it holds no SEK swap or basis quotes, and
`fred_treasury_cmt_history.csv` is the only time series in it. Each section opens by naming
the constraint that put it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import bar_figure, overlay_figure
from app.data import cmt_history, hullwhite_calibration, load_snapshot, usd_curves
from app.state import AppState
from yieldcurve.curves.build import government_swap_basis
from yieldcurve.models.hullwhite import HullWhite
from yieldcurve.risk.pca import daily_changes, fit_pca

_BASIS_TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
_GRID = np.linspace(0.05, 10.0, 300)
_BP = 10_000.0
_N_COMPONENTS = 3
_COMPONENT_NAMES = ("Level", "Slope", "Curvature")


def _section_a(state: AppState) -> None:
    st.subheader("A government curve is not a discount curve")
    st.markdown(
        "This section is in USD because the snapshot forces it: there are no SEK swap or "
        "basis quotes in it. After 2008, the curve that projects a floating coupon and the "
        "curve that discounts it stopped being the same curve. That separation is the "
        "entire content of multi-curve pricing."
    )
    curves = usd_curves(state.asof, state.method)
    forecast = curves.forecast_for("3M")
    st.plotly_chart(
        overlay_figure(
            {
                "OIS (discounting)": (
                    _GRID.tolist(),
                    [curves.discount.zero(float(t)) * 100.0 for t in _GRID],
                ),
                "3M forecast": (
                    _GRID.tolist(),
                    [forecast.zero(float(t)) * 100.0 for t in _GRID],
                ),
            },
            y_title="Zero rate (%)",
        ),
        width="stretch",
    )
    basis = government_swap_basis(load_snapshot(), state.asof, _BASIS_TENORS, method=state.method)
    st.plotly_chart(
        bar_figure(
            [f"{t:g}y" for t in basis],
            [v * _BP for v in basis.values()],
            y_title="Government-swap basis (bp)",
        ),
        width="stretch",
    )
    st.caption(
        "The basis chart is the price of the distinction: what a government curve says a "
        "cash flow is worth, minus what the swap market says. Both curves and the basis "
        "are **constructed** from the packaged snapshot (see `DATA_SOURCES.md`) — not "
        "observed live quotes."
    )
    with st.expander("Government-swap basis by tenor (data)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Tenor (y)": [f"{t:g}" for t in basis],
                    "Government-swap basis (bp)": [v * _BP for v in basis.values()],
                }
            ),
            width="stretch",
            hide_index=True,
        )


def _section_b() -> None:
    st.subheader("A curve has no dynamics")
    st.markdown(
        "A bootstrapped curve is one day's photograph. It cannot tell you how likely "
        "tomorrow's curve is. Principal components extracted from "
        "`fred_treasury_cmt_history.csv` — the only time series in the snapshot — give the "
        "three shapes that historically account for almost all of the movement."
    )
    changes, tenors = daily_changes(cmt_history())
    result = fit_pca(changes, tenors, n_components=_N_COMPONENTS)

    columns = st.columns(_N_COMPONENTS)
    for column, name, ratio in zip(
        columns, _COMPONENT_NAMES, result.explained_variance_ratio, strict=True
    ):
        column.metric(f"{name} — variance explained (% of centred variance)", f"{ratio * 100:.1f}%")

    st.plotly_chart(
        overlay_figure(
            {
                name: (list(result.tenors), result.loadings[i, :].tolist())
                for i, name in enumerate(_COMPONENT_NAMES)
            },
            y_title="Component loading (unit-norm, dimensionless)",
        ),
        width="stretch",
    )
    st.caption(
        f"{result.n_observations} daily observations of US Treasury CMT par-yield changes "
        "(a USD proxy — the snapshot holds no SEK rate history). PCA is a statistical "
        "decomposition, not an arbitrage-free model: it describes how the curve has moved, "
        "and it cannot price anything. That is the division of labour with the next "
        "section — Hull-White prices, PCA describes."
    )
    with st.expander("Component loadings (data)"):
        st.dataframe(
            pd.DataFrame(
                {name: result.loadings[i, :].tolist() for i, name in enumerate(_COMPONENT_NAMES)},
                index=[f"{t:g}y" for t in result.tenors],
            ),
            width="stretch",
        )


def _section_c(state: AppState) -> None:
    st.subheader("A curve prices linear products only")
    st.markdown(
        "A discount curve prices anything whose payoff is linear in rates. A swaption is "
        "not: its value depends on the distribution of future rates, not just their "
        "expectation. That needs a model, and Hull-White one-factor is the simplest one "
        "that stays arbitrage-free against the curve it was built on."
    )
    st.warning(
        "**The volatilities below are illustrative, not market data.** Real cleared-"
        "swaption settlement volatility surfaces are licensed and are not redistributed "
        "here. This grid is constructed from a closed form stated in the CSV header and "
        "in `DATA_SOURCES.md`. What follows therefore demonstrates how well a two-parameter "
        "model spans a surface; it is not a fit to traded prices."
    )

    curves = usd_curves(state.asof, state.method)
    fit = hullwhite_calibration(state.asof, state.method)

    a, b, c = st.columns(3)
    a.metric("Calibrated a (1/yr)", f"{fit.a:.5f}")
    b.metric("Calibrated sigma (1/√yr)", f"{fit.sigma:.5f}")
    c.metric("Residual (bp, illustrative grid)", f"{fit.rmse_vol_bp:.2f}")

    table = pd.DataFrame(
        {
            "Expiry": list(fit.expiries),
            "Swap maturity": list(fit.maturities),
            "Illustrative vol (bp)": [v * _BP for v in fit.market_vols],
            "Model vol (bp)": [v * _BP for v in fit.model_vols],
            "Difference (bp)": [
                (m - k) * _BP for m, k in zip(fit.model_vols, fit.market_vols, strict=True)
            ],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "The residual is a real number because the volatilities did not come from the model "
        "being fitted. Two parameters cannot span an expiry-by-tenor grid; the misses "
        "concentrate at the short expiries, which is the reason desks reach for two-factor "
        "models or a local-volatility overlay."
    )

    st.markdown("**What the two parameters do** — sliders below drive the illustration only.")
    slider_a = st.slider(
        "Mean reversion a (1/yr)",
        0.001,
        0.50,
        float(fit.a),
        0.001,
        help="Drives the illustration only; not part of the calibration above.",
    )
    slider_sigma = st.slider(
        "Volatility sigma (1/√yr)",
        0.0001,
        0.05,
        float(fit.sigma),
        0.0001,
        help="Drives the illustration only; not part of the calibration above.",
    )
    model = HullWhite(curve=curves.discount, a=slider_a, sigma=slider_sigma)
    times = [0.0, *np.linspace(0.25, 10.0, 40).tolist()]
    paths = model.simulate(times, n_paths=25, seed=20260803)
    st.plotly_chart(
        overlay_figure(
            {f"path {i}": (times, [v * 100.0 for v in paths[i]]) for i in range(paths.shape[0])},
            y_title="Short rate (%)",
        ),
        width="stretch",
    )
    st.caption(
        "These paths are illustrative and are driven by the sliders, not by the calibration "
        "above. The calibrated values are the metrics; the sliders are a toy."
    )


def render(state: AppState) -> None:
    _section_a(state)
    st.divider()
    _section_b()
    st.divider()
    _section_c(state)
