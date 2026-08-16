"""Inflation analytics: breakeven curves, real curves, linkers, ZC swaps.

The closed-form cross-checks are software verification of the implementation
against a hand-derived reference — the Fisher identity and the ZC-swap par
condition — not an empirical or regulatory validation of any model. The
reference derivations are written out next to each pinned value.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldcurve.conventions import add_months
from yieldcurve.curves.build import usd_ois_curve
from yieldcurve.curves.protocol import FlatCurve
from yieldcurve.inflation import (
    BreakevenCurve,
    InflationError,
    InflationLinkedBond,
    LinkerError,
    RealRateCurve,
    ZeroCouponInflationSwap,
    index_ratio,
    price_linker,
    zc_swap_legs,
    zc_swap_par_breakeven,
)
from yieldcurve.market.snapshot import Snapshot

ASOF = date(2026, 7, 24)

# A flat nominal curve at 4% and a flat breakeven at 2.5% (both continuously
# compounded decimals), so the real curve is flat at 1.5% and every closed form
# reduces to arithmetic with the constants below.
_NOMINAL = 0.04
_BREAKEVEN = 0.025
_REAL = _NOMINAL - _BREAKEVEN


def _flat_breakeven(rate: float = _BREAKEVEN) -> BreakevenCurve:
    return BreakevenCurve(reference_date=ASOF, tenors=(1.0, 30.0), breakevens=(rate, rate))


def _flat_real_curve(rate: float = _BREAKEVEN) -> RealRateCurve:
    return RealRateCurve(
        nominal=FlatCurve(reference_date=ASOF, rate=_NOMINAL), breakeven=_flat_breakeven(rate)
    )


# --- BreakevenCurve ----------------------------------------------------------


def test_breakeven_curve_interpolates_linearly_between_knots() -> None:
    curve = BreakevenCurve(reference_date=ASOF, tenors=(1.0, 3.0), breakevens=(0.02, 0.03))

    assert curve.breakeven(1.0) == pytest.approx(0.02)
    assert curve.breakeven(3.0) == pytest.approx(0.03)
    assert curve.breakeven(2.0) == pytest.approx(0.025)


def test_breakeven_curve_extrapolates_flat_beyond_the_knots() -> None:
    curve = BreakevenCurve(reference_date=ASOF, tenors=(1.0, 3.0), breakevens=(0.02, 0.03))

    assert curve.breakeven(0.25) == pytest.approx(0.02)
    assert curve.breakeven(10.0) == pytest.approx(0.03)


def test_breakeven_curve_allows_negative_breakevens() -> None:
    # Deflation is a legitimate market state; only non-finite values are rejected.
    curve = BreakevenCurve(reference_date=ASOF, tenors=(1.0, 2.0), breakevens=(-0.01, 0.0))

    assert curve.breakeven(1.5) == pytest.approx(-0.005)


@pytest.mark.parametrize(
    ("tenors", "breakevens", "message"),
    [
        ((1.0, 2.0), (0.02,), "breakevens"),
        ((), (), "at least one"),
        ((0.0, 2.0), (0.02, 0.03), "positive"),
        ((2.0, 1.0), (0.02, 0.03), "strictly increasing"),
        ((1.0, float("inf")), (0.02, 0.03), "finite"),
        ((1.0, 2.0), (0.02, float("nan")), "finite"),
    ],
)
def test_breakeven_curve_rejects_invalid_construction(
    tenors: tuple[float, ...], breakevens: tuple[float, ...], message: str
) -> None:
    with pytest.raises(InflationError, match=message):
        BreakevenCurve(reference_date=ASOF, tenors=tenors, breakevens=breakevens)


def test_breakeven_curve_rejects_non_positive_curve_time() -> None:
    curve = _flat_breakeven()

    with pytest.raises(InflationError, match="non-negative"):
        curve.breakeven(-0.5)


# --- RealRateCurve -----------------------------------------------------------


def test_real_zero_rate_is_nominal_minus_breakeven() -> None:
    curve = _flat_real_curve()

    # Fisher relation, continuous form: r(T) = n(T) - b(T) = 4% - 2.5% = 1.5%.
    assert curve.zero(5.0) == pytest.approx(_REAL)


def test_real_discount_factor_matches_the_hand_derived_closed_form() -> None:
    curve = _flat_real_curve()

    # df_real(T) = exp(-(n - b) T) = exp(-0.015 * 5).
    assert curve.df(5.0) == pytest.approx(math.exp(-_REAL * 5.0), rel=1e-15)


def test_real_discount_factor_equals_nominal_times_inflation() -> None:
    curve = _flat_real_curve()

    # df_real(T) = df_nominal(T) * exp(b T): the Fisher identity in discount factors.
    t = 7.0
    assert curve.df(t) == pytest.approx(
        math.exp(-_NOMINAL * t) * math.exp(_BREAKEVEN * t), rel=1e-15
    )


def test_real_curve_rejects_mismatched_reference_dates() -> None:
    other = BreakevenCurve(reference_date=date(2020, 1, 1), tenors=(1.0,), breakevens=(0.02,))

    with pytest.raises(InflationError, match="reference date"):
        RealRateCurve(nominal=FlatCurve(reference_date=ASOF, rate=_NOMINAL), breakeven=other)


def test_real_curve_forward_rate_is_flat_for_a_flat_input() -> None:
    curve = _flat_real_curve()

    # A flat real curve has a flat forward: fwd(1, 3) = r = 1.5%.
    assert curve.fwd(1.0, 3.0) == pytest.approx(_REAL)


def test_real_curve_fwd_rejects_a_non_positive_interval() -> None:
    curve = _flat_real_curve()

    with pytest.raises(InflationError, match="exceed"):
        curve.fwd(2.0, 2.0)


# --- InflationLinkedBond and pricing -----------------------------------------


def _flat_bond(lag_months: int) -> InflationLinkedBond:
    return InflationLinkedBond(
        base_date=ASOF,
        maturity=add_months(ASOF, 24),
        face=100.0,
        coupon=0.02,
        frequency=1,
        base_index=100.0,
        indexation_lag_months=lag_months,
    )


def test_linker_price_matches_hand_derived_reference() -> None:
    # Derivation (continuous compounding, all rates decimals):
    #   nominal n = 4%, breakeven b = 2.5% -> real r = n - b = 1.5%.
    #   Bond: face 100, 2% annual coupon (frequency 1), base = ASOF,
    #   maturity = base + 1y. That span is exactly 365 ACT/365F days, so
    #   t = 1.0 exactly and the single real cashflow at maturity is the final
    #   coupon plus the face: 2.0 + 100.0 = 102.0.
    #   real_price = 102 exp(-0.015 * 1) = 102 * 0.9851119396030626
    #              = 100.48141783951239
    bond = InflationLinkedBond(
        base_date=ASOF,
        maturity=add_months(ASOF, 12),
        face=100.0,
        coupon=0.02,
        frequency=1,
        base_index=100.0,
        indexation_lag_months=0,
    )
    result = price_linker(bond, _flat_real_curve(), ASOF)

    assert result.real_price == pytest.approx(100.48141783951239, rel=1e-13)


def test_linker_cashflows_bundle_the_final_coupon_with_the_face() -> None:
    bond = _flat_bond(lag_months=0)

    assert bond.real_cashflows() == (
        (add_months(ASOF, 12), 2.0),
        (add_months(ASOF, 24), 102.0),
    )


def test_linker_nominal_price_equals_real_price_without_the_lag() -> None:
    # With indexation_lag_months = 0 the index date is the payment date, so
    # nominal_price = sum real * exp(b t) * exp(-n t) = sum real * df_real = real_price.
    result = price_linker(_flat_bond(lag_months=0), _flat_real_curve(), ASOF)

    assert result.nominal_price == pytest.approx(result.real_price, rel=1e-13)


def test_indexation_lag_lowers_the_nominal_price_under_positive_inflation() -> None:
    # A positive breakeven means the projected index rises with time; observing
    # the index 3 months earlier lowers every indexation ratio, so the indexed
    # (nominal) price sits below the real price.
    lagged = price_linker(_flat_bond(lag_months=3), _flat_real_curve(), ASOF)
    unlagged = price_linker(_flat_bond(lag_months=0), _flat_real_curve(), ASOF)

    assert lagged.nominal_price < unlagged.real_price


def test_index_ratio_is_one_at_the_base_date() -> None:
    assert index_ratio(_flat_breakeven(), ASOF, ASOF) == pytest.approx(1.0)


def test_index_ratio_rejects_an_index_date_before_the_base() -> None:
    with pytest.raises(LinkerError, match="precedes"):
        index_ratio(_flat_breakeven(), ASOF, add_months(ASOF, -1))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("face", 0.0, "face"),
        ("coupon", -0.01, "coupon"),
        ("frequency", 5, "frequency"),
        ("base_index", 0.0, "base_index"),
        ("indexation_lag_months", -1, "indexation_lag_months"),
    ],
)
def test_linker_rejects_invalid_construction(
    field: str, bad_value: float | int, message: str
) -> None:
    kwargs: dict[str, date | float | int] = {
        "base_date": ASOF,
        "maturity": add_months(ASOF, 24),
        field: bad_value,
    }
    with pytest.raises(LinkerError, match=message):
        InflationLinkedBond(**kwargs)  # type: ignore[arg-type]


def test_linker_rejects_a_maturity_before_the_base_date() -> None:
    with pytest.raises(LinkerError, match="after base_date"):
        InflationLinkedBond(base_date=ASOF, maturity=add_months(ASOF, -1))


def test_zc_swap_rejects_a_non_finite_breakeven() -> None:
    with pytest.raises(LinkerError, match="fixed_breakeven"):
        ZeroCouponInflationSwap(
            start_date=ASOF, maturity=add_months(ASOF, 12), fixed_breakeven=float("nan")
        )


def test_zc_swap_rejects_a_maturity_before_the_start() -> None:
    with pytest.raises(LinkerError, match="after start_date"):
        ZeroCouponInflationSwap(start_date=ASOF, maturity=add_months(ASOF, -1))


def test_price_linker_requires_the_valuation_date_to_match_the_curve() -> None:
    with pytest.raises(LinkerError, match="asof"):
        price_linker(_flat_bond(lag_months=0), _flat_real_curve(), date(2026, 7, 25))


# --- Zero-coupon inflation swap ----------------------------------------------


def _flat_swap(breakeven: float = _BREAKEVEN) -> ZeroCouponInflationSwap:
    return ZeroCouponInflationSwap(
        start_date=ASOF, maturity=add_months(ASOF, 60), notional=1.0, fixed_breakeven=breakeven
    )


def test_zc_swap_par_breakeven_is_the_curve_breakeven() -> None:
    # At par the fixed and floating terminal amounts coincide, which forces
    # K = b(T): the par breakeven is exactly the zero-coupon breakeven at T.
    assert zc_swap_par_breakeven(_flat_breakeven(), ASOF, add_months(ASOF, 60)) == pytest.approx(
        _BREAKEVEN
    )


def test_zc_swap_fixed_leg_matches_hand_derived_reference() -> None:
    # Derivation: N = 1, T = 1y, K = b = 2.5%, nominal n = 4%.
    #   T = 1.0 exactly (the 2026-07-24 -> 2027-07-24 span is 365 ACT/365F days).
    #   fixed leg PV = (exp(K T) - 1) exp(-n T) = (exp(0.025) - 1) exp(-0.04)
    #                = (1.0253151205244289 - 1) * 0.9607894391523232
    #                = 0.0253151205244289 * 0.9607894391523232 = 0.024322500450739467
    swap = ZeroCouponInflationSwap(
        start_date=ASOF, maturity=add_months(ASOF, 12), notional=1.0, fixed_breakeven=_BREAKEVEN
    )
    legs = zc_swap_legs(swap, FlatCurve(reference_date=ASOF, rate=_NOMINAL), _flat_breakeven())

    assert legs.fixed_leg_pv == pytest.approx(0.024322500450739467, rel=1e-13)


def test_zc_swap_is_zero_at_par() -> None:
    legs = zc_swap_legs(
        _flat_swap(), FlatCurve(reference_date=ASOF, rate=_NOMINAL), _flat_breakeven()
    )

    assert legs.par_breakeven == pytest.approx(_BREAKEVEN)
    assert legs.net_pv == pytest.approx(0.0, abs=1e-15)


def test_zc_swap_rejects_a_forward_start() -> None:
    forward = ZeroCouponInflationSwap(
        start_date=add_months(ASOF, 12),
        maturity=add_months(ASOF, 60),
        fixed_breakeven=_BREAKEVEN,
    )

    with pytest.raises(LinkerError, match="spot-starting"):
        zc_swap_legs(forward, FlatCurve(reference_date=ASOF, rate=_NOMINAL), _flat_breakeven())


def test_zc_swap_rejects_a_non_positive_notional() -> None:
    with pytest.raises(LinkerError, match="notional"):
        ZeroCouponInflationSwap(start_date=ASOF, maturity=add_months(ASOF, 60), notional=0.0)


def test_zc_swap_par_breakeven_rejects_a_forward_start() -> None:
    with pytest.raises(LinkerError, match="spot-starting"):
        zc_swap_par_breakeven(_flat_breakeven(), add_months(ASOF, 12), add_months(ASOF, 60))


def test_zc_swap_legs_reject_mismatched_reference_dates() -> None:
    other = BreakevenCurve(reference_date=date(2020, 1, 1), tenors=(1.0,), breakevens=(0.02,))

    with pytest.raises(LinkerError, match="reference date"):
        zc_swap_legs(_flat_swap(), FlatCurve(reference_date=ASOF, rate=_NOMINAL), other)


# --- End-to-end off the packaged snapshot ------------------------------------


def test_packaged_breakevens_build_a_real_curve_off_the_usd_curve(snapshot: Snapshot) -> None:
    frame = snapshot.load("illustrative_inflation_breakevens")
    breakeven = BreakevenCurve(
        reference_date=ASOF,
        tenors=tuple(float(t) for t in frame["tenor_years"]),
        breakevens=tuple(float(b) for b in frame["breakeven"]),
    )
    nominal = usd_ois_curve(snapshot, ASOF)
    real = RealRateCurve(nominal=nominal, breakeven=breakeven)

    # A real zero rate must sit below the nominal zero rate when the breakeven
    # is positive, and the 10y real discount factor must be positive and < 1.
    t = 10.0
    assert real.zero(t) < nominal.zero(t)
    assert 0.0 < real.df(t) < 1.0


# --- Property checks ---------------------------------------------------------


@given(
    st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-0.05, max_value=0.10, allow_nan=False, allow_infinity=False),
)
def test_breakeven_and_real_df_are_positive_and_finite(tenor: float, rate: float) -> None:
    breakeven = _flat_breakeven(rate)
    real = RealRateCurve(nominal=FlatCurve(reference_date=ASOF, rate=_NOMINAL), breakeven=breakeven)

    b = breakeven.breakeven(tenor)
    df = real.df(tenor)
    assert math.isfinite(b)
    # A real discount factor is exp(-r t): always strictly positive and finite,
    # but not bounded above by 1 when the real rate is negative (r = n - b < 0).
    assert math.isfinite(df) and df > 0.0
