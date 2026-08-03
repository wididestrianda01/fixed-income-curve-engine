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
from app.data import cmt_history, load_snapshot, usd_curves
from app.state import AppState
from yieldcurve.curves.build import government_swap_basis
from yieldcurve.models.hullwhite import HullWhite, atm_swaption_grid, calibrate
from yieldcurve.risk.pca import daily_changes, fit_pca

_BASIS_TENORS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
_GRID = np.linspace(0.05, 10.0, 300)
_VOL_DATASET = "illustrative_swaption_vols"
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
        use_container_width=True,
    )
    basis = government_swap_basis(load_snapshot(), state.asof, _BASIS_TENORS, method=state.method)
    st.plotly_chart(
        bar_figure(
            [f"{t:g}y" for t in basis],
            [v * _BP for v in basis.values()],
            y_title="Government-swap basis (bp)",
        ),
        use_container_width=True,
    )
    st.caption(
        "The basis chart is the price of the distinction: what a government curve says a "
        "cash flow is worth, minus what the swap market says."
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
        column.metric(f"{name} — variance explained", f"{ratio * 100:.1f}%")

    st.plotly_chart(
        overlay_figure(
            {
                name: (list(result.tenors), result.loadings[:, i].tolist())
                for i, name in enumerate(_COMPONENT_NAMES)
            },
            y_title="Loading",
        ),
        use_container_width=True,
    )
    st.caption(
        f"{result.n_observations} daily observations. PCA is a statistical decomposition, "
        "not an arbitrage-free model: it describes how the curve has moved, and it cannot "
        "price anything. That is the division of labour with the next section — Hull-White "
        "prices, PCA describes."
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
        "**The volatilities below are illustrative, not market data.** Real CME cleared-"
        "swaption settlement volatilities require a CME Information License Agreement and "
        "may not be redistributed here — which is why `market/cme.py` raises rather than "
        "caches. This grid is constructed from a closed form stated in the CSV header and "
        "in `DATA_SOURCES.md`. What follows therefore demonstrates how well a two-parameter "
        "model spans a surface. It is not a fit to traded prices."
    )

    curves = usd_curves(state.asof, state.method)
    swaptions, vols = atm_swaption_grid(
        load_snapshot(), state.asof, curves.discount, dataset=_VOL_DATASET
    )
    result = calibrate(curves.discount, swaptions, vols, state.asof)

    a, b, c = st.columns(3)
    a.metric("Calibrated a", f"{result.a:.5f}")
    b.metric("Calibrated sigma", f"{result.sigma:.5f}")
    c.metric("Residual (bp)", f"{result.rmse_vol_bp:.2f}")

    table = pd.DataFrame(
        {
            "Expiry": [s.expiry for s in swaptions],
            "Swap maturity": [s.swap.maturity for s in swaptions],
            "Illustrative vol (bp)": [v * _BP for v in result.market_vols],
            "Model vol (bp)": [v * _BP for v in result.model_vols],
            "Difference (bp)": [
                (m - k) * _BP for m, k in zip(result.model_vols, result.market_vols, strict=True)
            ],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "The residual is a real number because the volatilities did not come from the model "
        "being fitted. Two parameters cannot span an expiry-by-tenor grid; the misses "
        "concentrate at the short expiries, which is the reason desks reach for two-factor "
        "models or a local-volatility overlay."
    )

    st.markdown("**What the two parameters do** — sliders below drive the illustration only.")
    slider_a = st.slider("Mean reversion a", 0.001, 0.50, float(result.a), 0.001)
    slider_sigma = st.slider("Volatility sigma", 0.0001, 0.05, float(result.sigma), 0.0001)
    model = HullWhite(curve=curves.discount, a=slider_a, sigma=slider_sigma)
    times = [0.0, *np.linspace(0.25, 10.0, 40).tolist()]
    paths = model.simulate(times, n_paths=25, seed=20260803)
    st.plotly_chart(
        overlay_figure(
            {f"path {i}": (times, paths[i].tolist()) for i in range(paths.shape[0])},
            y_title="Short rate",
        ),
        use_container_width=True,
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
