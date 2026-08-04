"""Tab 3 — risk on one bond, then risk on a book, then risk against history."""

from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from app.charts import bar_figure, histogram_figure
from app.data import (
    VAR_WINDOW,
    cmt_history,
    gov_bonds,
    pnl_sample,
    portfolio,
    sek_curveset,
    sek_quotes,
)
from app.state import AppState
from app.tabs.pricing import bond_label
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.pricing import price
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.risk.keyrate import SEK_KEY_RATES, krd
from yieldcurve.risk.ladder import par_delta_ladder
from yieldcurve.risk.portfolio import (
    eve_ladder,
    historical_pnl,
    present_value,
    var_es,
)
from yieldcurve.risk.scenarios import eu_scenarios
from yieldcurve.risk.sensitivities import (
    convexity,
    dv01,
    effective_convexity,
    effective_duration,
    modified_duration,
)

TIER1_CAPITAL = 4_000_000_000.0
"""Illustrative, and committed alongside the demo portfolio. It exists to give the 15%
outlier threshold something to divide by."""

_OUTLIER_FRACTION = 0.15
_SMOOTH_METHODS = (InterpMethod.CUBIC_LOG_DF, InterpMethod.LOG_LINEAR_DF)


def _selected_bond() -> FixedCouponBond:
    bonds = gov_bonds()
    return cast(FixedCouponBond, bonds[st.session_state.get("bond_index", 0)])


def _section_a(state: AppState) -> None:
    st.subheader("One bond's risk")
    bond = _selected_bond()
    st.caption(f"Showing {bond_label(bond)} — the bond selected on the Pricing tab.")
    curves = sek_curveset(state.asof, state.method)

    a, b, c = st.columns(3)
    a.metric("DV01", f"{dv01(bond, curves, state.asof):.6f}")
    b.metric("Modified duration", f"{modified_duration(bond, curves, state.asof):.4f}")
    c.metric("Effective duration", f"{effective_duration(bond, curves, state.asof):.4f}")
    st.caption(
        "The two durations differ because they answer different questions. Modified "
        "duration assumes a parallel move in the bond's own yield and is analytic in that "
        "yield. Effective duration reprices the bond off a bumped curve, so it carries "
        "whatever the interpolation scheme does between pillars. On a par bond under a flat "
        "curve they agree; away from that they need not."
    )

    d, e = st.columns(2)
    d.metric("Convexity", f"{convexity(bond, curves, state.asof):.4f}")
    e.metric("Effective convexity", f"{effective_convexity(bond, curves, state.asof):.4f}")
    st.caption(
        "Convexity conventions differ between vendors by factors of two and by whether the "
        "frequency scaling is included. This one is the library's own. Compare the price "
        "change it predicts, not the number."
    )


def _section_b(state: AppState) -> None:
    st.subheader("Two ladders that do not agree")
    bond = _selected_bond()
    curves = sek_curveset(state.asof, state.method)

    left, right = st.columns(2)
    ladder = krd(bond, curves, state.asof, SEK_KEY_RATES)
    with left:
        st.plotly_chart(
            bar_figure(
                [f"{k:g}y" for k in ladder],
                list(ladder.values()),
                y_title="Key-rate duration (years)",
            ),
            use_container_width=True,
        )
        st.caption(
            "The SEK 1y key rate is interpolated, not observed. The Riksbank publishes 6m "
            "bills and 2y benchmarks with nothing between."
        )

    par = par_delta_ladder(bond, sek_quotes(state.asof), state.asof, method=state.method)
    with right:
        st.plotly_chart(
            bar_figure(
                [f"{d:%Y-%m}" for d in par],
                list(par.values()),
                y_title="Par-rate delta",
            ),
            use_container_width=True,
        )

    if state.method in _SMOOTH_METHODS:
        additivity = (
            "Under this smooth scheme the ladder is additive to about 1e-4: the sum of the "
            "bucket sensitivities reconstructs a parallel bump."
        )
    else:
        additivity = (
            "Under monotone convex the ladder is additive only to about 1.4%. The scheme's "
            "amendment tests are branches, so the curve is not a linear function of its "
            "inputs and the buckets do not have to sum. Switch the sidebar to a smooth "
            "method and the gap closes to 1e-4."
        )
    st.caption(
        "These two ladders do not agree entry by entry, and they are not supposed to. A "
        "bump to the 5y par quote moves every zero out to five years; a key-rate hat moves "
        "a triangle centred on one maturity. They are different questions. " + additivity
    )


def _section_c(state: AppState) -> None:
    st.subheader("The IRRBB board")
    book = portfolio()
    curves = sek_curveset(state.asof, state.method)
    scenarios = eu_scenarios("SEK")
    ladder = eve_ladder(book, curves, state.asof, scenarios)
    base = present_value(book, curves, state.asof)

    worst_name = min(ladder, key=lambda name: ladder[name])
    worst = ladder[worst_name]
    left, right = st.columns(2)
    left.metric(f"Worst-case ΔEVE ({worst_name})", f"{worst:,.0f} SEK")
    right.metric("As % of Tier 1", f"{worst / TIER1_CAPITAL * 100:.2f}%")

    threshold = -_OUTLIER_FRACTION * TIER1_CAPITAL
    figure = bar_figure(list(ladder), list(ladder.values()), y_title="ΔEVE (SEK)")
    figure.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="-15% of Tier 1 (BCBS d368 outlier test)",
    )
    if ladder.values():
        figure.update_traces(
            marker_color=["#c0392b" if v <= threshold else "#1f4e79" for v in ladder.values()]
        )
    st.plotly_chart(figure, use_container_width=True)

    st.markdown(
        "The six shocks are 200bp parallel, 300bp short and 150bp long, read from "
        "`scenarios.toml`, which cites EBA GL/2018/02 Table 1 and BCBS d368 Annex 2 Table 2 "
        "on every row. SEK falls in the same shock bucket as USD under those tables, so the "
        "identical numbers are by design rather than by mistake. The -15% of Tier 1 line is "
        "the supervisory outlier test from BCBS d368.\n\n"
        "This is ΔEVE on a stylised proxy book. There are no deposits, no net interest "
        "income, and no behavioural assumptions about non-maturity accounts. A real IRRBB "
        "submission is a considerably larger object than this chart."
    )

    rows = [
        {
            "Position": p.label,
            "Type": type(p.instrument).__name__,
            "Notional": p.notional,
            "Base PV": p.notional
            / float(getattr(p.instrument, "face", 1.0))
            * price(p.instrument, curves, state.asof).dirty,
        }
        for p in book.positions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Base portfolio value {base:,.0f} SEK.")

    st.markdown(
        "**The issuer's side of the same table.** ΔEVE across a maturity-laddered "
        "government book is the bank's framing of a question a debt office asks in reverse. "
        "A bank holding fixed-rate government paper reads the parallel-up bar as a loss of "
        "economic value; the sovereign that issued that paper reads the same shock as the "
        "cost of having termed out its borrowing rather than funding short. Same discount "
        "factors, same six scenarios, opposite sign convention. Cost versus risk in debt "
        "composition — maturity exposure, and how that choice looks under a prescribed "
        "shock set — is the computation this board already performs."
    )


def _section_d(state: AppState) -> None:
    st.subheader("What rates actually did")
    st.markdown(
        "Section C is *prescribed*: a supervisor fixes the shock and the book is revalued "
        "under it. This section asks the complementary question — what did rates actually "
        "do, and what would that have done to this book. The two belong side by side "
        "because they fail differently. A prescribed shock cannot be too small by accident; "
        "an empirical distribution cannot contain a move the sample never saw."
    )
    st.warning(
        "**This is a volatility proxy, not a SEK VaR.** The snapshot holds no SEK rate "
        "history — `fred_treasury_cmt_history.csv` is the only time series in it. The book "
        "is SEK. So SEK bucket exposures are driven here by US Treasury daily changes. USD "
        "and SEK rate volatility are correlated but not equal, and this number moves if the "
        "real series is substituted. Reporting a VaR without naming the sample it came from "
        "is exactly the failure this repository documents everywhere else."
    )

    changes, tenors = pnl_sample()
    book = portfolio()
    curves = sek_curveset(state.asof, state.method)
    pnl = historical_pnl(book, curves, state.asof, changes, tenors)

    confidence = st.radio(
        "Confidence",
        options=(0.95, 0.99),
        index=1,
        format_func=lambda c: f"{c:.0%}",
        horizontal=True,
    )
    value_at_risk, shortfall = var_es(pnl, confidence=confidence)

    left, right = st.columns(2)
    left.metric(f"VaR ({confidence:.0%})", f"{value_at_risk:,.0f} SEK")
    right.metric(f"Expected shortfall ({confidence:.0%})", f"{shortfall:,.0f} SEK")

    st.plotly_chart(
        histogram_figure(
            pnl,
            markers={"VaR": -value_at_risk, "ES": -shortfall},
            x_title="Daily P&L (SEK)",
        ),
        use_container_width=True,
    )
    dates = sorted(cmt_history()["date"].unique())[-VAR_WINDOW:]
    st.caption(
        f"{pnl.size} daily observations, {dates[0]} to {dates[-1]}, from "
        "`fred_treasury_cmt_history.csv`. Expected shortfall is the mean of the tail beyond "
        "VaR, so it can never be the smaller of the two."
    )


def render(state: AppState) -> None:
    _section_a(state)
    st.divider()
    _section_b(state)
    st.divider()
    _section_c(state)
    st.divider()
    _section_d(state)
