"""Tab 3 — risk on one bond, then risk on a book, then risk against history."""

from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from app.charts import ADVERSE, PALETTE, bar_figure, histogram_figure
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
"""Illustrative, and committed alongside the demo portfolio. It exists only to
give the dashed reference line on the ΔEVE chart something to divide by. It is
an invented number — not regulatory capital — and no supervisory threshold is
applied anywhere in the app."""

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
    a.metric("DV01 (loss per 1 bp, per 100 face)", f"{dv01(bond, curves, state.asof):.6f}")
    b.metric("Modified duration (years)", f"{modified_duration(bond, curves, state.asof):.4f}")
    c.metric("Effective duration (years)", f"{effective_duration(bond, curves, state.asof):.4f}")
    st.caption(
        "DV01 is the positive loss a long position takes when rates rise 1 bp — base "
        "price minus the +1 bp price — in SEK per 100 face. The two durations differ "
        "because they answer different questions. Modified duration assumes a parallel "
        "move in the bond's own yield and is analytic in that yield. Effective duration "
        "reprices the bond off a bumped curve, so it carries whatever the interpolation "
        "scheme does between pillars. On a par bond under a flat curve they agree; away "
        "from that they need not."
    )

    d, e = st.columns(2)
    d.metric(
        "Convexity (per 1.0 rate², library convention)",
        f"{convexity(bond, curves, state.asof):.4f}",
    )
    e.metric(
        "Effective convexity (per 1.0 rate², library convention)",
        f"{effective_convexity(bond, curves, state.asof):.4f}",
    )
    st.caption(
        "Convexity units are per (1.0 decimal rate change)², and conventions differ "
        "between vendors by factors of two and by whether the frequency scaling is "
        "included. This one is the library's own. Compare the price change it predicts, "
        "not the number."
    )


def _section_b(state: AppState) -> None:
    st.subheader("Two ladders that do not agree")
    bond = _selected_bond()
    curves = sek_curveset(state.asof, state.method)

    left, right = st.columns(2)
    with left:
        try:
            ladder = krd(bond, curves, state.asof, SEK_KEY_RATES)
        except ValueError:
            # A par swap prices at zero, so its normalized KRD is undefined (the
            # library raises). The monetary ladders below stay defined — say so
            # instead of letting the tab fail.
            st.info(
                "Key-rate duration is undefined for this instrument: it prices at zero "
                "(a par swap), so normalizing by its present value is meaningless. The "
                "par-rate ladder and the portfolio bucket exposures stay defined."
            )
            ladder = {}
        st.plotly_chart(
            bar_figure(
                [f"{k:g}y" for k in ladder],
                list(ladder.values()),
                y_title="Key-rate duration (price-bp per yield-bp)",
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
                y_title="Par-rate delta (per 100 face per 1 bp)",
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
        "a triangle centred on one maturity. They are different questions. Units: key-rate "
        "duration is in price-bp per yield-bp — numerically equal to years of duration, "
        "not multiplied by 100 — and the par-rate ladder is in price per 100 face per 1 bp "
        "rise in the quoted instrument. " + additivity
    )

    with st.expander("Ladder data"):
        ladder_frame = pd.DataFrame(
            {
                "Key rate (y)": [f"{k:g}" for k in ladder],
                "KRD (price-bp per yield-bp)": [round(v, 6) for v in ladder.values()],
            }
        )
        par_frame = pd.DataFrame(
            {
                "Bucket (maturity)": [d.strftime("%Y-%m") for d in par],
                "Par-rate delta (per 100 face per 1 bp)": [round(v, 6) for v in par.values()],
            }
        )
        st.dataframe(ladder_frame, use_container_width=True, hide_index=True)
        st.dataframe(par_frame, use_container_width=True, hide_index=True)


def _section_c(state: AppState) -> None:
    st.subheader("Illustrative ΔEVE comparison (EU 2024/856 shocks)")
    book = portfolio()
    curves = sek_curveset(state.asof, state.method)
    scenarios = eu_scenarios("SEK")
    ladder = eve_ladder(book, curves, state.asof, scenarios)
    base = present_value(book, curves, state.asof)

    worst_name = min(ladder, key=lambda name: ladder[name])
    worst = ladder[worst_name]
    st.metric("Worst-case illustrative ΔEVE (SEK)", f"{worst:,.0f}")
    st.caption(f"Under the {worst_name} scenario.")

    threshold = -_OUTLIER_FRACTION * TIER1_CAPITAL
    figure = bar_figure(
        list(ladder),
        list(ladder.values()),
        y_title="Illustrative ΔEVE (SEK)",
        text_format="{:,.0f}",
    )
    figure.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=ADVERSE,
        annotation_text="Illustrative reference: -15% of the invented Tier 1 proxy",
    )
    if ladder.values():
        figure.update_traces(
            marker_color=[ADVERSE if v <= threshold else PALETTE[0] for v in ladder.values()]
        )
    st.plotly_chart(figure, use_container_width=True)

    st.markdown(
        "The six shocks are the **EU 2024/856** supervisory scenarios — parallel up/down, "
        "short-rate up/down, steepener and flattener — read from the packaged "
        "`scenarios.toml`, whose rows cite Commission Delegated Regulation (EU) 2024/856 "
        "Annex Part A for the USD/SEK parameters (200 bp parallel, 300 bp short, 150 bp "
        "long) and apply the Article 3(7) post-shock rate floor. USD and SEK share those "
        "parameters in the regulation, so the identical numbers are by design.\n\n"
        "This is an **illustrative ΔEVE comparison** on a stylised, single-currency SEK "
        "proxy book — an educational exhibit, not a regulatory EVE measure and not an "
        "IRRBB submission. There are no deposits, no net interest income, and no "
        "behavioural assumptions about non-maturity accounts. The dashed reference line "
        "sits at -15% of the portfolio file's `tier1_capital`, an **invented** number "
        "that is **not regulatory capital**; no capital is computed and no supervisory "
        "threshold is applied."
    )

    rows = [
        {
            "Position": p.label,
            "Type": type(p.instrument).__name__,
            "Notional (SEK)": p.notional,
            "Base PV (SEK)": p.notional
            / float(getattr(p.instrument, "face", 1.0))
            * price(p.instrument, curves, state.asof).dirty,
        }
        for p in book.positions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Base portfolio value {base:,.0f} SEK — a single-currency SEK book whose "
        "notionals are invented (see `demo_portfolio.toml`)."
    )

    with st.expander("ΔEVE by scenario (data)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Scenario": list(ladder),
                    "Illustrative ΔEVE (SEK)": [f"{v:,.0f}" for v in ladder.values()],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _section_d(state: AppState) -> None:
    st.subheader("What rates actually did (historical proxy)")
    st.markdown(
        "Section C is *prescribed*: a supervisor fixes the shock and the book is revalued "
        "under it. This section asks the complementary question — what did rates actually "
        "do, and what would that have done to this book. The two belong side by side "
        "because they fail differently: a prescribed shock is fixed by the regulation "
        "regardless of what markets did, while an empirical distribution cannot contain a "
        "move the sample never saw."
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
        help="Changes the VaR/ES confidence on this tab only.",
    )
    value_at_risk, shortfall = var_es(pnl, confidence=confidence)

    left, right = st.columns(2)
    left.metric(f"Linearized delta VaR ({confidence:.0%})", f"{value_at_risk:,.0f} SEK")
    right.metric(f"Linearized delta ES ({confidence:.0%})", f"{shortfall:,.0f} SEK")

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
        "`fred_treasury_cmt_history.csv` (US Treasury CMT par yields). P&L is first-order "
        "— bucket exposures revalued linearly, with no full revaluation — so both numbers "
        "are a **linearized delta proxy**, not a regulatory VaR. Expected shortfall is the "
        "mean of the tail beyond VaR, so it can never be the smaller of the two."
    )

    with st.expander("Daily P&L observations (data)"):
        st.dataframe(
            pd.DataFrame({"Daily P&L (SEK)": pnl.tolist()}),
            use_container_width=True,
            hide_index=True,
        )


def render(state: AppState) -> None:
    _section_a(state)
    st.divider()
    _section_b(state)
    st.divider()
    _section_c(state)
    st.divider()
    _section_d(state)
