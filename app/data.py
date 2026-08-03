"""Cached bridges from a Snapshot to the objects the tabs render.

Every function here is a @st.cache_data wrapper over something already tested in
src/yieldcurve/. Caching is keyed on the arguments, so a tab that asks for the same curve
twice pays for it once.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from yieldcurve.calendars import SwedenCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.bootstrap import Quote
from yieldcurve.curves.build import (
    sek_government_curve,
    sek_government_quotes,
    usd_curveset,
    usd_ois_quotes,
)
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.risk.portfolio import Portfolio

__all__ = [
    "SNAPSHOT_DATE",
    "cmt_history",
    "gov_bonds",
    "load_snapshot",
    "portfolio",
    "sek_curve",
    "sek_curveset",
    "sek_quotes",
    "usd_curves",
    "usd_ois_quote_table",
]

SNAPSHOT_DATE = date(2026, 7, 24)
"""The one committed snapshot. The as-of control displays this; it does not choose it."""

_PORTFOLIO_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_portfolio.toml"
_LAST_PILLAR_YEARS = 10.0


@st.cache_data(show_spinner=False)
def load_snapshot() -> Snapshot:
    """The committed offline snapshot. Failure here is fatal — the app has no fallback."""
    return Snapshot(date=SNAPSHOT_DATE)


@st.cache_data(show_spinner=False)
def sek_quotes(asof: date) -> tuple[Quote, ...]:
    """The bill and benchmark quotes the SEK curve is bootstrapped from."""
    return sek_government_quotes(load_snapshot(), asof)


@st.cache_resource(show_spinner=False)
def sek_curve(asof: date, method: InterpMethod) -> InterpolatedDiscountCurve:
    return sek_government_curve(load_snapshot(), asof, method=method)


@st.cache_resource(show_spinner=False)
def sek_curveset(asof: date, method: InterpMethod) -> CurveSet:
    """Single-curve mode: SEK government discounting, no separate forecast curve."""
    return CurveSet.single(sek_curve(asof, method))


@st.cache_resource(show_spinner=False)
def usd_curves(asof: date, method: InterpMethod) -> CurveSet:
    """OIS discounting with a 3M forecast curve — the post-2008 arrangement."""
    return usd_curveset(load_snapshot(), asof, method=method)


@st.cache_data(show_spinner=False)
def usd_ois_quote_table(asof: date) -> tuple[Quote, ...]:
    return usd_ois_quotes(load_snapshot(), asof)


@st.cache_data(show_spinner=False)
def cmt_history() -> pd.DataFrame:
    """Daily US Treasury constant-maturity yields, long form: date, tenor_years, rate."""
    return load_snapshot().load("fred_treasury_cmt_history")


@st.cache_data(show_spinner=False)
def portfolio() -> Portfolio:
    return Portfolio.from_toml(_PORTFOLIO_PATH)


@st.cache_data(show_spinner=False)
def gov_bonds() -> tuple[FixedCouponBond, ...]:
    """Riksgälden benchmarks that mature inside the bootstrapped curve.

    The 2039 and 2071 bonds are excluded on purpose: the SEK curve's last pillar is the 10y
    benchmark, and quoting a price beyond it would be extrapolation presented as a price.
    """
    frame = load_snapshot().load("riksgalden_gov_bonds")
    horizon = SNAPSHOT_DATE + timedelta(days=int(_LAST_PILLAR_YEARS * 365.25))
    bonds = []
    for row in frame.itertuples():
        maturity = pd.Timestamp(row.maturity_date).date()  # type: ignore[arg-type]
        if maturity > horizon:
            continue
        bonds.append(
            FixedCouponBond(
                issue=pd.Timestamp(row.issue_date).date(),  # type: ignore[arg-type]
                maturity=maturity,
                coupon=float(row.coupon),  # type: ignore[arg-type]
                frequency=1,
                day_count=DayCount.THIRTY_360_BOND,
                calendar=SwedenCalendar(),
                bdc=BusinessDayConvention.FOLLOWING,
            )
        )
    return tuple(sorted(bonds, key=lambda b: b.maturity))
