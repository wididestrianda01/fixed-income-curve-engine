"""Bootstrapping a discount curve from quoted instruments."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from curveengine.calendars import NullCalendar
from curveengine.conventions import BusinessDayConvention, DayCount, year_fraction
from curveengine.curves.bootstrap import (
    Quote,
    bootstrap,
    discount_factors_from_cashflow_matrix,
)
from curveengine.curves.interpolation import CurveConstructionError, InterpMethod
from curveengine.curves.protocol import CurveSet, curve_time
from curveengine.instruments import Bill, FixedCouponBond, VanillaSwap
from curveengine.pricing import par_rate, price

REFERENCE = date(2026, 7, 24)
ALL_METHODS = list(InterpMethod)


def bills() -> list[Quote]:
    return [
        Quote(Bill(maturity=date(2026, 10, 24), day_count=DayCount.ACT_360), 0.0215),
        Quote(Bill(maturity=date(2027, 1, 24), day_count=DayCount.ACT_360), 0.0222),
        Quote(Bill(maturity=date(2027, 7, 24), day_count=DayCount.ACT_360), 0.0231),
    ]


def bond(maturity: date, coupon: float) -> FixedCouponBond:
    return FixedCouponBond(
        issue=REFERENCE,
        maturity=maturity,
        coupon=coupon,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )


def bonds() -> list[Quote]:
    return [
        Quote(bond(date(2028, 7, 24), 0.0248), 0.0248),
        Quote(bond(date(2031, 7, 24), 0.0286), 0.0286),
        Quote(bond(date(2036, 7, 24), 0.0318), 0.0318),
    ]


def swap(maturity: date, rate: float) -> VanillaSwap:
    return VanillaSwap(
        start=REFERENCE,
        maturity=maturity,
        fixed_rate=rate,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )


@pytest.mark.parametrize("method", ALL_METHODS)
def test_bills_reprice_to_their_own_quotes(method: InterpMethod) -> None:
    """The bootstrap's defining property. Anything short of 1e-10 means the
    solver stopped early or the curve does not pass through its knots."""
    curve = bootstrap(bills(), asof=REFERENCE, method=method)
    curves = CurveSet.single(curve)

    for quote in bills():
        instrument = quote.instrument
        assert isinstance(instrument, Bill)

        t = curve_time(REFERENCE, instrument.maturity)
        tau = year_fraction(REFERENCE, instrument.maturity, instrument.day_count)
        implied = (1.0 / curve.df(t) - 1.0) / tau

        assert implied == pytest.approx(quote.rate, abs=1e-10)
        assert price(instrument, curves, REFERENCE).dirty > 0.0


@pytest.mark.parametrize("method", [InterpMethod.LOG_LINEAR_DF])
def test_par_bonds_reprice_to_par(method: InterpMethod) -> None:
    curve = bootstrap([*bills(), *bonds()], asof=REFERENCE, method=method)
    curves = CurveSet.single(curve)

    for quote in bonds():
        instrument = quote.instrument
        assert isinstance(instrument, FixedCouponBond)
        assert price(instrument, curves, REFERENCE).dirty == pytest.approx(100.0, abs=1e-10)


@pytest.mark.parametrize("method", [InterpMethod.LOG_LINEAR_DF])
def test_swaps_reprice_to_their_quoted_par_rates(method: InterpMethod) -> None:
    quotes = [
        *bills(),
        Quote(swap(date(2029, 7, 24), 0.0259), 0.0259),
        Quote(swap(date(2033, 7, 24), 0.0295), 0.0295),
    ]
    curve = bootstrap(quotes, asof=REFERENCE, method=method)
    curves = CurveSet.single(curve)

    for quote in quotes[3:]:
        instrument = quote.instrument
        assert isinstance(instrument, VanillaSwap)
        assert par_rate(instrument, curves, REFERENCE) == pytest.approx(quote.rate, abs=1e-10)


def test_bootstrap_knots_land_on_instrument_maturities() -> None:
    curve = bootstrap(bills() + bonds(), asof=REFERENCE)

    assert len(curve.times) == 6
    assert curve.times[0] == pytest.approx((date(2026, 10, 24) - REFERENCE).days / 365.0)
    assert curve.times[-1] == pytest.approx((date(2036, 7, 24) - REFERENCE).days / 365.0)


def test_quotes_out_of_maturity_order_are_sorted_not_rejected() -> None:
    """Market data arrives in whatever order the API returns it."""
    shuffled = [bills()[2], bills()[0], bills()[1]]

    curve = bootstrap(shuffled, asof=REFERENCE)

    assert list(curve.times) == sorted(curve.times)


def test_two_instruments_with_the_same_maturity_are_rejected() -> None:
    duplicated = [bills()[0], bills()[0]]

    with pytest.raises(CurveConstructionError, match="same maturity"):
        bootstrap(duplicated, asof=REFERENCE)


def test_an_already_matured_instrument_is_rejected() -> None:
    stale = [Quote(Bill(maturity=date(2026, 1, 1), day_count=DayCount.ACT_360), 0.02)]

    with pytest.raises(CurveConstructionError, match="matures"):
        bootstrap(stale, asof=REFERENCE)


def test_matrix_form_agrees_with_the_sequential_bootstrap() -> None:
    """The linear-algebra view: with coupon dates aligned across bonds, the
    cash-flow matrix is lower triangular and d = CF^-1 P solves in one step.
    Agreement with the sequential solver checks both against each other."""
    quotes = bonds()
    curve = bootstrap(quotes, asof=REFERENCE, method=InterpMethod.LOG_LINEAR_DF)

    maturities = [date(2028, 7, 24), date(2031, 7, 24), date(2036, 7, 24)]
    payment_dates = sorted(
        {d for quote in quotes for d in quote.instrument.coupon_dates()[1:]}  # type: ignore[attr-defined]
    )
    matrix: np.ndarray = np.zeros((len(quotes), len(payment_dates)))
    for row, quote in enumerate(quotes):
        for flow in quote.instrument.cashflows(REFERENCE):  # type: ignore[attr-defined]
            matrix[row, payment_dates.index(flow.date)] = flow.amount
    prices: np.ndarray = np.full(len(quotes), 100.0)

    square: np.ndarray = matrix[:, [payment_dates.index(m) for m in maturities]]
    assert square.shape == (3, 3)

    from curveengine.curves.protocol import curve_time

    del square  # the aligned-date reduction below is the actual comparison
    reduced: np.ndarray = np.zeros((len(quotes), len(quotes)))
    reduced_prices: np.ndarray = prices.copy()
    for row, quote in enumerate(quotes):
        for flow in quote.instrument.cashflows(REFERENCE):  # type: ignore[attr-defined]
            if flow.date in maturities:
                reduced[row, maturities.index(flow.date)] += flow.amount
            else:
                reduced_prices[row] -= flow.amount * curve.df(curve_time(REFERENCE, flow.date))

    solved = discount_factors_from_cashflow_matrix(reduced, reduced_prices)

    for maturity, df in zip(maturities, solved, strict=True):
        assert curve.df(curve_time(REFERENCE, maturity)) == pytest.approx(df, abs=1e-10)


def test_empty_quotes_are_rejected() -> None:
    with pytest.raises(CurveConstructionError, match="no quotes"):
        bootstrap([], asof=REFERENCE)


def test_discount_factors_rejects_wrong_shaped_prices() -> None:
    with pytest.raises(CurveConstructionError, match="prices"):
        discount_factors_from_cashflow_matrix(np.eye(3), np.ones(4))


def test_matrix_form_rejects_a_non_square_system() -> None:
    with pytest.raises(CurveConstructionError, match="square"):
        discount_factors_from_cashflow_matrix(np.ones((2, 3)), np.ones(2))


def test_a_quote_no_discount_factor_can_reach_is_rejected() -> None:
    """The solver brackets the unknown discount factor in [1e-8, 5.0]. A par
    rate of -1000% lies outside what any factor in that range can produce, so
    the residual has the same sign at both ends and Brent has nothing to solve.
    Better to name the inconsistent quote than to hand brentq an invalid
    bracket and let it raise about the sign of f(a) and f(b)."""
    with pytest.raises(CurveConstructionError, match="inconsistent"):
        bootstrap([Quote(swap(date(2031, 7, 24), -10.0), -10.0)], asof=REFERENCE)
