"""Bootstrapping a discount curve from quoted instruments."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from yieldcurve.calendars import NullCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount, year_fraction
from yieldcurve.curves.bootstrap import (
    Quote,
    bootstrap,
    discount_factors_from_cashflow_matrix,
    repricing_report,
)
from yieldcurve.curves.interpolation import (
    CurveConstructionError,
    InterpMethod,
    overlay_curve,
)
from yieldcurve.curves.pricing import par_rate, price
from yieldcurve.curves.protocol import CurveSet, curve_time
from yieldcurve.instruments import Bill, FixedCouponBond, VanillaSwap

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


def off_knot_quotes() -> list[Quote]:
    """3M/6M deposits plus 2Y/3Y annual swaps: the 1Y swap payment falls between
    knots, which is what makes overlay residuals measurable."""
    return [
        Quote(Bill(maturity=date(2026, 10, 24), day_count=DayCount.ACT_360), 0.0215),
        Quote(Bill(maturity=date(2027, 1, 24), day_count=DayCount.ACT_360), 0.0222),
        Quote(swap(date(2028, 7, 24), 0.0259), 0.0259),
        Quote(swap(date(2029, 7, 24), 0.0295), 0.0295),
    ]


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


def test_hand_derived_deposits_and_swaps_recover_canonical_discount_factors() -> None:
    """Independent oracle: deposits imply ``df = 1 / (1 + r*tau)`` by definition,
    and an annual par swap obeys ``p * (tau_1y*df_1y + tau_2y*df_2y) = 1 - df_2y``
    because its floating leg telescopes. Every expected value below is computed
    from the quotes alone, never with the library's own curve code."""
    asof = REFERENCE
    dep6m = Bill(maturity=date(2027, 1, 24), day_count=DayCount.ACT_360)
    dep12m = Bill(maturity=date(2027, 7, 24), day_count=DayCount.ACT_360)
    r6, r12, par = 0.0222, 0.0231, 0.0259
    quotes = [
        Quote(dep6m, r6),
        Quote(dep12m, r12),
        Quote(swap(date(2028, 7, 24), par), par),
    ]
    curve = bootstrap(quotes, asof=asof)  # canonical default: log-linear DF

    tau6 = year_fraction(asof, dep6m.maturity, DayCount.ACT_360)
    tau12 = year_fraction(asof, dep12m.maturity, DayCount.ACT_360)
    df6 = 1.0 / (1.0 + r6 * tau6)
    df12 = 1.0 / (1.0 + r12 * tau12)
    # Annual 30/360 swap paying at 1Y and 2Y: tau = 1 per period, so
    # par * (df_1y + df_2y) = 1 - df_2y  =>  df_2y = (1 - par*df_1y) / (1 + par).
    df2 = (1.0 - par * df12) / (1.0 + par)

    assert curve.df(curve_time(asof, dep6m.maturity)) == pytest.approx(df6, abs=1e-13)
    assert curve.df(curve_time(asof, dep12m.maturity)) == pytest.approx(df12, abs=1e-13)
    assert curve.df(curve_time(asof, date(2028, 7, 24))) == pytest.approx(df2, abs=1e-10)


def test_repricing_report_covers_every_quote_and_verdicts_the_tolerance() -> None:
    quotes = [*bills(), *bonds(), Quote(swap(date(2038, 7, 24), 0.0300), 0.0300)]
    curve = bootstrap(quotes, asof=REFERENCE)
    report = repricing_report(curve, quotes, asof=REFERENCE)

    assert len(report) == len(quotes)
    for row, quote in zip(report, quotes, strict=True):
        assert row.instrument is quote.instrument
        assert row.target_rate == quote.rate
        assert row.residual == pytest.approx(row.model_rate - row.target_rate, abs=1e-15)
        assert row.tolerance == 1e-10
        assert row.ok, f"canonical build left residual {row.residual} on {row.instrument}"

    # A tolerance tighter than the solver's precision must flip the verdicts:
    # bills are closed form, but bond and swap rates are solved and cannot hit
    # 1e-16 in rate terms.
    tight = repricing_report(curve, quotes, asof=REFERENCE, tolerance=1e-16)
    assert not all(row.ok for row in tight)


def test_bootstrap_enforces_the_selected_tolerance() -> None:
    quotes = off_knot_quotes()
    curve = bootstrap(quotes[:3], asof=REFERENCE, tolerance=1e-10)  # canonical: exact
    assert curve.df(2.0) > 0.0

    # With a global interpolator a later pillar moves an interpolated payment,
    # so the final residual exceeds 1e-10 and the build must say so.
    with pytest.raises(CurveConstructionError, match="reprice"):
        bootstrap(
            quotes,
            asof=REFERENCE,
            method=InterpMethod.CUBIC_LOG_DF,
            tolerance=1e-10,
        )


def test_comparative_overlays_measure_nonzero_final_residuals() -> None:
    quotes = off_knot_quotes()
    canonical = bootstrap(quotes, asof=REFERENCE)  # canonical log-linear default
    assert all(row.ok for row in repricing_report(canonical, quotes, asof=REFERENCE))

    for method in (InterpMethod.CUBIC_LOG_DF, InterpMethod.MONOTONE_CONVEX):
        overlay = overlay_curve(canonical, method)
        assert overlay.times == canonical.times
        assert overlay.dfs == canonical.dfs
        report = repricing_report(overlay, quotes, asof=REFERENCE)
        assert len(report) == len(quotes)
        failing = [row for row in report if not row.ok]
        assert failing, f"{method.name} overlay should leave measured residuals"
        assert all(abs(row.residual) > 1e-8 for row in failing), [row.residual for row in failing]


def test_adding_later_pillars_does_not_change_an_earlier_canonical_solve() -> None:
    quotes = off_knot_quotes()
    early = bootstrap(quotes[:3], asof=REFERENCE)
    full = bootstrap(quotes, asof=REFERENCE)

    assert full.times[: len(early.times)] == early.times
    for earlier_df, later_df in zip(early.dfs, full.dfs[: len(early.dfs)], strict=True):
        assert earlier_df == pytest.approx(later_df, abs=1e-15)


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


def test_quotes_out_of_maturity_order_are_rejected_before_solving() -> None:
    """Market data may arrive in any order, but a sequential bootstrap that
    silently re-sorted it could hide a caller's mistake. Ordering is the
    caller's job; the solver names the violation instead."""
    shuffled = [bills()[2], bills()[0], bills()[1]]

    with pytest.raises(CurveConstructionError, match="increasing maturity order"):
        bootstrap(shuffled, asof=REFERENCE)


def test_two_instruments_with_the_same_maturity_are_rejected() -> None:
    duplicated = [bills()[0], bills()[0]]

    with pytest.raises(CurveConstructionError, match="same maturity"):
        bootstrap(duplicated, asof=REFERENCE)


def test_an_already_matured_instrument_is_rejected() -> None:
    stale = [Quote(Bill(maturity=date(2026, 1, 1), day_count=DayCount.ACT_360), 0.02)]

    with pytest.raises(CurveConstructionError, match="matures"):
        bootstrap(stale, asof=REFERENCE)


def test_non_finite_quote_rates_are_rejected_before_solving() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CurveConstructionError, match="Non-finite"):
            bootstrap(
                [Quote(Bill(maturity=date(2026, 10, 24), day_count=DayCount.ACT_360), bad)],
                asof=REFERENCE,
            )


def test_an_unsupported_instrument_is_rejected_before_solving() -> None:
    from yieldcurve.instruments import FRN

    frn = FRN(
        issue=REFERENCE,
        maturity=date(2028, 7, 24),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        index_tenor="3M",
        spread=0.0,
    )
    with pytest.raises(CurveConstructionError, match="Cannot bootstrap from FRN"):
        bootstrap([Quote(frn, 0.03)], asof=REFERENCE)


def test_matrix_form_agrees_with_the_sequential_bootstrap() -> None:
    """The linear-algebra view: with coupon dates aligned across bonds, the
    cash-flow matrix is lower triangular and d = CF^-1 P solves in one step.
    Agreement with the sequential solver checks both against each other."""
    quotes = bonds()
    curve = bootstrap(quotes, asof=REFERENCE, method=InterpMethod.LOG_LINEAR_DF)

    maturities = [date(2028, 7, 24), date(2031, 7, 24), date(2036, 7, 24)]
    instruments = [quote.instrument for quote in quotes]
    assert all(isinstance(i, FixedCouponBond) for i in instruments)
    bonds_: list[FixedCouponBond] = [i for i in instruments if isinstance(i, FixedCouponBond)]
    payment_dates = sorted({d for bond in bonds_ for d in bond.coupon_dates()[1:]})
    matrix: np.ndarray = np.zeros((len(quotes), len(payment_dates)))
    for row, bond in enumerate(bonds_):
        for flow in bond.cashflows(REFERENCE):
            matrix[row, payment_dates.index(flow.date)] = flow.amount

    reduced: np.ndarray = np.zeros((len(quotes), len(quotes)))
    reduced_prices: np.ndarray = np.full(len(quotes), 100.0)
    for row, bond in enumerate(bonds_):
        for flow in bond.cashflows(REFERENCE):
            if flow.date in maturities:
                reduced[row, maturities.index(flow.date)] += flow.amount
            else:
                reduced_prices[row] -= flow.amount * curve.df(curve_time(REFERENCE, flow.date))

    solved = discount_factors_from_cashflow_matrix(reduced, reduced_prices)

    for maturity, df in zip(maturities, solved, strict=True):
        assert curve.df(curve_time(REFERENCE, maturity)) == pytest.approx(df, abs=1e-10)


def test_matrix_form_rejects_a_rank_deficient_system() -> None:
    cashflows = np.array([[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises(CurveConstructionError, match="rank"):
        discount_factors_from_cashflow_matrix(cashflows, np.array([1.0, 1.0]))


def test_matrix_form_rejects_an_ill_conditioned_system() -> None:
    # Full rank, but the 1e14 condition number leaves no usable precision.
    cashflows = np.diag([1.0, 1e-14])

    with pytest.raises(CurveConstructionError, match="condition"):
        discount_factors_from_cashflow_matrix(cashflows, np.array([1.0, 1.0]))


def test_matrix_form_rejects_zero_price_normalization() -> None:
    with pytest.raises(CurveConstructionError, match="normaliz"):
        discount_factors_from_cashflow_matrix(np.eye(2), np.zeros(2))


def test_matrix_form_rejects_non_finite_inputs() -> None:
    cashflows = np.array([[1.0, np.nan], [0.0, 1.0]])

    with pytest.raises(CurveConstructionError, match="finite"):
        discount_factors_from_cashflow_matrix(cashflows, np.ones(2))


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
