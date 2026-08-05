"""Snapshot to curve. The single path from committed market data to a CurveSet.

Everything downstream — risk, Hull-White, notebooks, the Streamlit app — enters
here. Two callers bootstrapping the same market by two routes is how a project
ends up with two 10-year rates and no way to say which is the curve.

Builders default to the canonical method, log-linear discount-factor
interpolation, whose exactness contract is enforced: a canonical build that did
not reprice every quote within tolerance is a bug and fails loudly. Comparative
methods (cubic log-DF, monotone convex) remain available as overlays built on
the canonical nodes via ``yieldcurve.curves.interpolation.overlay_curve``;
their residuals are measured with
``yieldcurve.curves.bootstrap.repricing_report``, never asserted to vanish. A
A ``method=`` parameter also remains on the builders so the Streamlit app
can offer comparative interpolation overlays at runtime.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from yieldcurve.calendars import NullCalendar, SwedenCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.bootstrap import Quote, bootstrap
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.protocol import CurveSet, DiscountCurve
from yieldcurve.instruments import OIS, Bill, FixedCouponBond, VanillaSwap
from yieldcurve.market.snapshot import Snapshot

_FORECAST_TENOR = "3M"
_CANONICAL_TOLERANCE = 1e-10
_MONTH_ALIGNMENT_TOLERANCE = 1e-3
_MAX_TENOR_YEARS = 1000.0

__all__ = [
    "CurveDataError",
    "government_swap_basis",
    "sek_government_curve",
    "sek_government_quotes",
    "usd_curveset",
    "usd_forecast_curve",
    "usd_government_curve",
    "usd_ois_curve",
    "usd_ois_quotes",
]


class CurveDataError(ValueError):
    """The snapshot cannot support the requested curve."""


def _add_months(asof: date, months: int) -> date:
    """Calendar-month arithmetic from ``asof``, clamping an end-of-month anchor
    (e.g. 31 Aug -> Feb 28/29) so the anniversary stays a valid date."""
    total = asof.year * 12 + (asof.month - 1) + months
    year, zero_based_month = divmod(total, 12)
    month = zero_based_month + 1
    day = min(asof.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _maturity(asof: date, years: float) -> date:
    """Maturity for a year tenor, by calendar months from ``asof``.

    Integer years land on the anniversary (CORE-01: 10Y from 24 Jul is 24 Jul,
    not a rounded 365-day delta), and month-grid tenors such as 0.25Y or the
    rounded 1M representation 0.0833 land on the calendar-month date. Tenors
    that are not month-aligned fall back to whole days.
    """
    if not math.isfinite(years) or years <= 0.0:
        raise CurveDataError(f"Invalid tenor in years: {years!r}")
    if years > _MAX_TENOR_YEARS:
        raise CurveDataError(
            f"Tenor {years!r} years exceeds the supported maximum of {_MAX_TENOR_YEARS:g} years"
        )
    months = round(years * 12.0)
    if abs(months / 12.0 - years) <= _MONTH_ALIGNMENT_TOLERANCE:
        return _add_months(asof, months)
    return asof + timedelta(days=round(years * 365.0))


def _bootstrap_curve(
    quotes: Sequence[Quote],
    asof: date,
    *,
    method: InterpMethod,
    discount_curve: DiscountCurve | None = None,
) -> InterpolatedDiscountCurve:
    """Bootstrap and, for the canonical method, enforce the exactness contract.

    The canonical default (log-linear DF) is the only method that preserves the
    sequential bootstrap's exact repricing; with a comparative method the build
    is an overlay whose final residuals are measured, not enforced.
    """
    tolerance = _CANONICAL_TOLERANCE if method is InterpMethod.LOG_LINEAR_DF else None
    return bootstrap(
        quotes, asof=asof, method=method, discount_curve=discount_curve, tolerance=tolerance
    )


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
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> InterpolatedDiscountCurve:
    return _bootstrap_curve(usd_ois_quotes(snapshot, asof), asof=asof, method=method)


def _forecast_par_rates(snapshot: Snapshot) -> pd.DataFrame:
    available = set(snapshot.available())
    if "usd_term_sofr_swaps" in available:
        return snapshot.load("usd_term_sofr_swaps").sort_values("tenor_years")
    if {"usd_forecast_basis", "usd_ois_swaps"} <= available:
        ois = snapshot.load("usd_ois_swaps").sort_values("tenor_years")
        basis = snapshot.load("usd_forecast_basis").sort_values("tenor_years")
        ois["tenor_years"] = ois["tenor_years"].astype(float)
        basis["tenor_years"] = basis["tenor_years"].astype(float)
        merged = ois.merge(basis, on="tenor_years", how="inner", validate="one_to_one")
        if merged.empty:
            raise CurveDataError(
                "usd_forecast_basis shares no tenor with usd_ois_swaps; "
                "the two files were built on different grids"
            )
        # CORE-05: a basis that omits an OIS tenor inside its own covered span
        # would leave a hole in the forecast grid. Tenors beyond the covered
        # span are simply outside the basis data; holes are a defect.
        covered_min = float(basis["tenor_years"].min())
        covered_max = float(basis["tenor_years"].max())
        merged_tenors = set(merged["tenor_years"])
        missing = [
            float(t)
            for t in ois["tenor_years"]
            if covered_min <= t <= covered_max and t not in merged_tenors
        ]
        if missing:
            raise CurveDataError(
                "usd_forecast_basis omits OIS tenor(s) inside its covered span "
                f"[{covered_min:g}, {covered_max:g}]: {missing}. The forecast grid would "
                "have holes; refusing to build a truncated forecast curve."
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
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
    discount_curve: DiscountCurve | None = None,
) -> InterpolatedDiscountCurve:
    """Bootstrap the 3M projection curve, discounting the quoted swaps off OIS.

    ``discount_curve`` defaults to the snapshot's own OIS curve. Passing ``None``
    explicitly is not a way to opt out of OIS discounting; it is how the caller
    says "build the OIS curve for me too".

    The forecast grid is the OIS grid intersected with the committed basis data:
    the committed snapshot's basis covers 0.25Y to 10Y, so the forecast curve's
    covered horizon is 10Y (``curve.covered_horizon``). Beyond the covered
    horizon the curve extrapolates flat in the zero rate — a stated modelling
    rule, not observed market data, and extrapolated values are unobservable
    inputs.
    """
    if discount_curve is None:
        discount_curve = usd_ois_curve(snapshot, asof, method=method)
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
    return _bootstrap_curve(quotes, asof=asof, method=method, discount_curve=discount_curve)


def usd_curveset(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> CurveSet:
    ois = usd_ois_curve(snapshot, asof, method=method)
    return CurveSet(
        discount=ois,
        forecast={
            _FORECAST_TENOR: usd_forecast_curve(snapshot, asof, method=method, discount_curve=ois)
        },
    )


def sek_government_quotes(snapshot: Snapshot, asof: date) -> tuple[Quote, ...]:
    """Bills and benchmark yields from the Riksbank snapshot datasets.

    Riksbank benchmark yields are treated as *par-yield* inputs: the quoted
    yield is used as the coupon of a par ``FixedCouponBond`` that prices to 100.
    Payments fall on unadjusted anniversaries, so every cashflow lands on a
    bootstrap knot and the par-yield mapping is exact by construction. That is
    the documented model mapping for this educational pipeline, not a raw
    security-price bootstrap and not an official Riksbank curve construction.
    """
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
                    bdc=BusinessDayConvention.UNADJUSTED,
                ),
                rate=float(quoted),
            )
        )
    return tuple(sorted(quotes, key=lambda q: q.instrument.maturity))  # type: ignore[attr-defined]


def sek_government_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> InterpolatedDiscountCurve:
    return _bootstrap_curve(sek_government_quotes(snapshot, asof), asof=asof, method=method)


def _cmt_quote(asof: date, years: float, rate: float, calendar: USGovernmentBondCalendar) -> Quote:
    """One CMT tenor as a bill (<= 1Y) or a semiannual par bond (> 1Y).

    Payments fall on unadjusted anniversaries so every cashflow lands on a
    bootstrap knot and the CMT par-yield mapping stays exact (see
    ``usd_government_curve``).
    """
    maturity = _maturity(asof, years)
    if years <= 1.0:
        return Quote(instrument=Bill(maturity=maturity, day_count=DayCount.ACT_360), rate=rate)
    return Quote(
        instrument=FixedCouponBond(
            issue=asof,
            maturity=maturity,
            coupon=rate,
            frequency=2,
            day_count=DayCount.ACT_ACT_ICMA,
            calendar=calendar,
            bdc=BusinessDayConvention.UNADJUSTED,
        ),
        rate=rate,
    )


def usd_government_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> InterpolatedDiscountCurve:
    """The USD government curve from published Treasury CMT yields.

    The FRED CMT series are *approximate CMT-implied curve inputs*: each quoted
    yield is mapped onto a par instrument (a bill up to 1Y, a semiannual par
    bond beyond) purely so the snapshot can be bootstrapped. Bond payments fall
    on unadjusted anniversaries so every cashflow lands on a bootstrap knot and
    the mapping is exact. This is a labelled model mapping, not a set of raw
    Treasury security prices and not an official Treasury bootstrap.
    """
    frame = snapshot.load("fred_treasury_cmt").sort_values("tenor_years")
    calendar = USGovernmentBondCalendar()
    quotes = tuple(
        _cmt_quote(asof, float(years), float(rate), calendar)
        for years, rate in zip(frame["tenor_years"], frame["rate"], strict=True)
    )
    return _bootstrap_curve(quotes, asof=asof, method=method)


def _validate_tenors(tenors: Sequence[float]) -> None:
    if not tenors:
        raise CurveDataError("No basis tenors requested")
    for tenor in tenors:
        if not math.isfinite(tenor):
            raise CurveDataError(f"Non-finite basis tenor {tenor!r}")
        if tenor <= 0.0:
            raise CurveDataError(f"Basis tenor must be positive, got {tenor!r}")
    if any(later <= earlier for earlier, later in itertools.pairwise(tenors)):
        raise CurveDataError(f"Basis tenors must be strictly increasing, got {list(tenors)}")


def government_swap_basis(
    snapshot: Snapshot,
    asof: date,
    tenors: Sequence[float],
    *,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> dict[float, float]:
    swap = usd_ois_curve(snapshot, asof, method=method)
    government = usd_government_curve(snapshot, asof, method=method)
    _validate_tenors(tenors)
    return {float(t): swap.zero(float(t)) - government.zero(float(t)) for t in tenors}
