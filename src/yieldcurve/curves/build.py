"""Snapshot to curve. The single path from committed market data to a CurveSet.

Everything downstream — risk, Hull-White, notebooks, the Streamlit app — enters
here. Two callers bootstrapping the same market by two routes is how a project
ends up with two 10-year rates and no way to say which is the curve.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from yieldcurve.calendars import NullCalendar, SwedenCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.bootstrap import Quote, bootstrap
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import OIS, Bill, FixedCouponBond, VanillaSwap
from yieldcurve.market.snapshot import Snapshot

_FORECAST_TENOR = "3M"


class CurveDataError(ValueError):
    """The snapshot cannot support the requested curve."""


def _maturity(asof: date, years: float) -> date:
    return asof + timedelta(days=round(years * 365.0))


def usd_ois_quotes(snapshot: Snapshot, asof: date) -> tuple[Quote, ...]:
    frame = snapshot.load("usd_ois_swaps").sort_values("tenor_years")
    quotes = []
    for years, rate in zip(frame["tenor_years"], frame["par_rate"], strict=True):
        swap = OIS(
            start=asof,
            maturity=_maturity(asof, float(years)),
            fixed_rate=float(rate),
            fixed_frequency=1,
            fixed_day_count=DayCount.ACT_360,
            float_day_count=DayCount.ACT_360,
            calendar=NullCalendar(),
            bdc=BusinessDayConvention.UNADJUSTED,
        )
        quotes.append(Quote(instrument=swap, rate=float(rate)))
    return tuple(quotes)


def usd_ois_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> InterpolatedDiscountCurve:
    return bootstrap(usd_ois_quotes(snapshot, asof), asof=asof, method=method)


def _forecast_par_rates(snapshot: Snapshot) -> pd.DataFrame:
    available = set(snapshot.available())
    if "usd_term_sofr_swaps" in available:
        return snapshot.load("usd_term_sofr_swaps").sort_values("tenor_years")
    if {"usd_forecast_basis", "usd_ois_swaps"} <= available:
        ois = snapshot.load("usd_ois_swaps").sort_values("tenor_years")
        basis = snapshot.load("usd_forecast_basis").sort_values("tenor_years")
        ois["tenor_years"] = ois["tenor_years"].astype(float)
        merged = ois.merge(basis, on="tenor_years", how="inner", validate="one_to_one")
        if merged.empty:
            raise CurveDataError(
                "usd_forecast_basis shares no tenor with usd_ois_swaps; "
                "the two files were built on different grids"
            )
        merged["par_rate"] = merged["par_rate"] + merged["basis_bp"] / 1e4
        return merged[["tenor_years", "par_rate"]]
    raise CurveDataError(
        "No forecast source in the snapshot: expected either "
        "'usd_term_sofr_swaps' or 'usd_forecast_basis'. "
        "See DATA_SOURCES.md for which branch this snapshot was built on."
    )


def usd_forecast_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> InterpolatedDiscountCurve:
    frame = _forecast_par_rates(snapshot)
    quotes = [
        Quote(
            instrument=VanillaSwap(
                start=asof,
                maturity=_maturity(asof, float(years)),
                fixed_rate=float(rate),
                fixed_frequency=2,
                fixed_day_count=DayCount.THIRTY_360_BOND,
                float_tenor=_FORECAST_TENOR,
                float_day_count=DayCount.ACT_360,
                calendar=NullCalendar(),
                bdc=BusinessDayConvention.UNADJUSTED,
            ),
            rate=float(rate),
        )
        for years, rate in zip(frame["tenor_years"], frame["par_rate"], strict=True)
    ]
    return bootstrap(quotes, asof=asof, method=method)


def usd_curveset(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> CurveSet:
    return CurveSet(
        discount=usd_ois_curve(snapshot, asof, method=method),
        forecast={_FORECAST_TENOR: usd_forecast_curve(snapshot, asof, method=method)},
    )


def sek_government_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> InterpolatedDiscountCurve:
    bills = snapshot.load("riksbank_bills")
    benchmarks = snapshot.load("riksbank_gov_benchmarks")
    calendar = SwedenCalendar()

    quotes: list[Quote] = [
        Quote(
            instrument=Bill(
                maturity=date.fromisoformat(str(maturity)),
                day_count=DayCount.ACT_360,
            ),
            rate=float(rate),
        )
        for maturity, rate in zip(bills["maturity_date"], bills["rate"], strict=True)
    ]
    for maturity, quoted in zip(benchmarks["maturity_date"], benchmarks["yield"], strict=True):
        m = date.fromisoformat(str(maturity))
        quotes.append(
            Quote(
                instrument=FixedCouponBond(
                    issue=asof,
                    maturity=m,
                    coupon=float(quoted),
                    frequency=1,
                    day_count=DayCount.THIRTY_360_BOND,
                    calendar=calendar,
                    bdc=BusinessDayConvention.FOLLOWING,
                ),
                rate=float(quoted),
            )
        )
    return bootstrap(_sorted(quotes), asof=asof, method=method)


def usd_government_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> InterpolatedDiscountCurve:
    frame = snapshot.load("fred_treasury_cmt").sort_values("tenor_years")
    calendar = USGovernmentBondCalendar()
    quotes: list[Quote] = []
    for years, rate in zip(frame["tenor_years"], frame["rate"], strict=True):
        years, rate = float(years), float(rate)
        maturity = _maturity(asof, years)
        if years <= 1.0:
            quotes.append(
                Quote(instrument=Bill(maturity=maturity, day_count=DayCount.ACT_360), rate=rate)
            )
        else:
            quotes.append(
                Quote(
                    instrument=FixedCouponBond(
                        issue=asof,
                        maturity=maturity,
                        coupon=rate,
                        frequency=2,
                        day_count=DayCount.ACT_ACT_ICMA,
                        calendar=calendar,
                        bdc=BusinessDayConvention.FOLLOWING,
                    ),
                    rate=rate,
                )
            )
    return bootstrap(tuple(quotes), asof=asof, method=method)


def government_swap_basis(
    snapshot: Snapshot,
    asof: date,
    tenors: Sequence[float],
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
) -> dict[float, float]:
    swap = usd_ois_curve(snapshot, asof, method=method)
    government = usd_government_curve(snapshot, asof, method=method)
    return {float(t): swap.zero(float(t)) - government.zero(float(t)) for t in tenors}


def _sorted(quotes: Sequence[Quote]) -> tuple[Quote, ...]:
    return tuple(sorted(quotes, key=lambda q: q.instrument.maturity))  # type: ignore[attr-defined]
