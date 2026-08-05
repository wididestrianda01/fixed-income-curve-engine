"""Cached bridges from a Snapshot to the objects the tabs render.

Every function here is a @st.cache_data wrapper over something already tested in
src/yieldcurve/. Caching is keyed on the arguments, so a tab that asks for the same curve
twice pays for it once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st

from yieldcurve.calendars import SwedenCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.bootstrap import Quote
from yieldcurve.curves.build import (
    sek_government_curve,
    sek_government_quotes,
    usd_curveset,
)
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.models.hullwhite import atm_swaption_grid, calibrate
from yieldcurve.risk.pca import daily_changes
from yieldcurve.risk.portfolio import Portfolio

__all__ = [
    "SNAPSHOT_DATE",
    "VAR_WINDOW",
    "HullWhiteCalibration",
    "cmt_history",
    "gov_bonds",
    "hullwhite_calibration",
    "load_snapshot",
    "pnl_sample",
    "portfolio",
    "sek_curve",
    "sek_curveset",
    "sek_quotes",
    "usd_curves",
]

SNAPSHOT_DATE = date(2026, 7, 24)
"""The one committed snapshot. The as-of control displays this; it does not choose it."""

_PORTFOLIO_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_portfolio.toml"
_LAST_PILLAR_YEARS = 10.0
_VOL_DATASET = "illustrative_swaption_vols"


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
    """Constructed USD curve set: OIS discounting from the constructed OIS
    par-rate grid plus a constructed 3M forecast basis. Both inputs are
    constructed/illustrative, not observed live quotes (see DATA_SOURCES.md)."""
    return usd_curveset(load_snapshot(), asof, method=method)


@dataclass(frozen=True)
class HullWhiteCalibration:
    """The calibrated two-parameter Hull-White fit on the illustrative vol grid.

    Carries the presentation-ready fields the Beyond tab shows: the fitted parameters,
    the residual, the per-swaption market/model vols, and the expiry and maturity grid
    they were calibrated on.
    """

    a: float
    sigma: float
    rmse_vol_bp: float
    market_vols: tuple[float, ...]
    model_vols: tuple[float, ...]
    expiries: tuple[date, ...]
    maturities: tuple[date, ...]


@st.cache_data(show_spinner=False)
def hullwhite_calibration(asof: date, method: InterpMethod) -> HullWhiteCalibration:
    """The illustrative-grid Hull-White fit, cached per curve choice.

    Calibration is the app's single expensive pure computation (about two seconds).
    Streamlit renders every tab on every rerun, so an interaction in any other tab would
    repeat it without this cache. The grid is illustrative, not market data — the Beyond
    tab discloses this on screen.
    """
    curves = usd_curves(asof, method)
    swaptions, vols = atm_swaption_grid(
        load_snapshot(), asof, curves.discount, dataset=_VOL_DATASET
    )
    result = calibrate(curves.discount, swaptions, vols, asof)
    return HullWhiteCalibration(
        a=result.a,
        sigma=result.sigma,
        rmse_vol_bp=result.rmse_vol_bp,
        market_vols=tuple(result.market_vols),
        model_vols=tuple(result.model_vols),
        expiries=tuple(s.expiry for s in swaptions),
        maturities=tuple(s.swap.maturity for s in swaptions),
    )


@st.cache_data(show_spinner=False)
def cmt_history() -> pd.DataFrame:
    """Daily US Treasury CMT par yields (public source), long form: date,
    tenor_years, rate."""
    return load_snapshot().load("fred_treasury_cmt_history")


@st.cache_data(show_spinner=False)
def portfolio() -> Portfolio:
    return Portfolio.from_toml(_PORTFOLIO_PATH)


VAR_WINDOW = 1000
"""Roughly four years of daily observations. Sized so the 99% confidence tail contains at
least the ten observations that yieldcurve.risk.portfolio requires for a stable expected
shortfall estimate."""


@st.cache_data(show_spinner=False)
def pnl_sample() -> tuple[npt.NDArray[np.float64], tuple[float, ...]]:
    """The most recent VAR_WINDOW daily changes of the CMT par-yield history,
    with their tenor grid.

    Windowing happens here rather than in yieldcurve.risk.portfolio so that the library
    function stays free of presentation choices. The changes are a CMT-implied history
    proxy, not an observed funding-rate history.
    """
    changes, tenors = daily_changes(cmt_history())
    return changes[-VAR_WINDOW:], tenors


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
