"""Cross-currency basis and CSA discounting.

The core closed forms are pinned against hand-derived references (each
derivation is in the comment above the assertion). There is no QuantLib parity
test here: the market basis is a synthetic *input* (a deviation from covered
interest parity, not a curve-implied quantity), so QuantLib's cross-currency
machinery — whose fair spread is zero under pure OIS curves because its engine
already applies CIP through the FX forward — does not provide an independent
check of the quantities this module computes. See notebook 10 for the
statement of what was and was not independently checked.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet, FlatCurve, curve_time
from yieldcurve.curves.xccy import (
    BasisCurve,
    CsaDiscountCurve,
    XccyError,
    basis_between,
    basis_curve_from_snapshot,
    basis_from_zeros,
    eur_discount_curve,
    swap_npv_difference,
    usd_swap,
)
from yieldcurve.instruments import Bill
from yieldcurve.market.snapshot import Snapshot

REFERENCE = date(2026, 7, 24)

# A constant illustrative EUR/USD basis: -20 bp (USD funding premium).
_BASIS = BasisCurve(tenors=(0.25, 30.0), basis_bp=(-20.0, -20.0))
_FLAT_04 = FlatCurve(reference_date=REFERENCE, rate=0.04)


def _constant_basis(bp: float) -> BasisCurve:
    return BasisCurve(tenors=(0.25, 30.0), basis_bp=(bp, bp))


# -- BasisCurve interpolation --------------------------------------------------


def test_basis_curve_interpolates_linearly_and_extrapolates_flat() -> None:
    curve = BasisCurve(tenors=(1.0, 3.0), basis_bp=(-10.0, -30.0))
    assert curve.basis_bp_at(1.0) == pytest.approx(-10.0)
    assert curve.basis_bp_at(3.0) == pytest.approx(-30.0)
    assert curve.basis_bp_at(2.0) == pytest.approx(-20.0)  # midpoint
    assert curve.basis_bp_at(0.0) == pytest.approx(-10.0)  # flat before first
    assert curve.basis_bp_at(10.0) == pytest.approx(-30.0)  # flat after last
    assert curve.basis(2.0) == pytest.approx(-0.0020)  # bp -> decimal


@pytest.mark.parametrize(
    ("tenors", "basis_bp", "message"),
    [
        ((1.0,), (-10.0,), "two tenor"),
        ((1.0, 2.0), (-10.0,), "2 tenors but 1 basis"),
        ((1.0, 2.0), (-10.0, -20.0, -30.0), "2 tenors but 3 basis"),
        ((1.0, 1.0), (-10.0, -20.0), "strictly increasing"),
        ((2.0, 1.0), (-10.0, -20.0), "strictly increasing"),
        ((0.0, 1.0), (-10.0, -20.0), "positive"),
        ((1.0, math.inf), (-10.0, -20.0), "finite"),
        ((1.0, 2.0), (-10.0, math.nan), "finite"),
    ],
)
def test_basis_curve_rejects_invalid_inputs(
    tenors: tuple[float, ...], basis_bp: tuple[float, ...], message: str
) -> None:
    with pytest.raises(XccyError, match=message):
        BasisCurve(tenors=tenors, basis_bp=basis_bp)


def test_basis_curve_rejects_non_positive_time() -> None:
    with pytest.raises(XccyError, match="non-negative"):
        _BASIS.basis(-0.1)


# -- CSA discount curve (hand-derived closed form) -----------------------------


def test_csa_discount_curve_closed_form() -> None:
    """df = base.df * exp(b * t), zero = base.zero - b, fwd = base.fwd - b.

    Hand-derived reference for a flat 4% base and a constant -20 bp basis at
    t = 2y: P = exp(-0.04 * 2), so df = exp(-0.04 * 2) * exp(-0.002 * 2)
    = exp(-0.084), and the zero rate is 0.04 - (-0.002) = 0.042.
    """
    csa = CsaDiscountCurve(base=_FLAT_04, basis=_BASIS)
    assert csa.df(2.0) == pytest.approx(math.exp(-0.084), rel=1e-14)
    assert csa.zero(2.0) == pytest.approx(0.042, rel=1e-14)
    assert csa.fwd(1.0, 2.0) == pytest.approx(0.042, rel=1e-14)
    assert csa.reference_date == REFERENCE


def test_csa_discount_curve_zero_rate_equals_base_minus_basis() -> None:
    csa = CsaDiscountCurve(base=_FLAT_04, basis=_BASIS)
    for t in (0.5, 1.0, 3.0, 10.0):
        assert csa.zero(t) == pytest.approx(_FLAT_04.zero(t) - _BASIS.basis(t), rel=1e-14)


@given(
    rate=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
    basis_bp=st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    t=st.floats(min_value=0.01, max_value=30.0, allow_nan=False, allow_infinity=False),
)
def test_csa_discount_factor_is_base_times_basis_shift(
    rate: float, basis_bp: float, t: float
) -> None:
    base = FlatCurve(reference_date=REFERENCE, rate=rate)
    basis = _constant_basis(basis_bp)
    csa = CsaDiscountCurve(base=base, basis=basis)
    expected = math.exp(-rate * t) * math.exp(basis.basis(t) * t)
    assert csa.df(t) == pytest.approx(expected, rel=1e-12)
    assert csa.zero(t) == pytest.approx(rate - basis.basis(t), rel=1e-12)


# -- The no-arbitrage identity ------------------------------------------------


def test_basis_between_recovers_the_input_basis() -> None:
    """b(t) = r_usd(t) - r_csa(t): the basis is the difference of the two curves."""
    csa = CsaDiscountCurve(base=_FLAT_04, basis=_BASIS)
    for t in (0.25, 2.0, 10.0, 25.0):
        assert basis_between(_FLAT_04, csa, t) == pytest.approx(_BASIS.basis(t), rel=1e-14)


def test_basis_from_zeros_closed_form() -> None:
    """The CIP fair spread s = (P_eur - P_usd) / (t * P_eur).

    For r_usd = 4%, r_eur = 2%, t = 1: s = 1 - exp(-0.02) ~ 0.01980, which is
    the rate differential r_usd - r_eur = 0.02 to first order.
    """
    result = basis_from_zeros(usd_zero=0.04, eur_zero=0.02, tenor=1.0)
    assert result == pytest.approx(1.0 - math.exp(-0.02), rel=1e-14)
    assert result == pytest.approx(0.02, abs=2.1e-4)  # ~ r_usd - r_eur


def test_basis_from_zeros_rejects_invalid_inputs() -> None:
    with pytest.raises(XccyError, match="tenor"):
        basis_from_zeros(0.04, 0.02, 0.0)
    with pytest.raises(XccyError, match="non-finite"):
        basis_from_zeros(math.inf, 0.02, 1.0)


# -- Snapshot wiring -----------------------------------------------------------


def test_basis_curve_from_snapshot_matches_packaged_data(snapshot: Snapshot) -> None:
    curve = basis_curve_from_snapshot(snapshot, REFERENCE)
    frame = snapshot.load("illustrative_xccy_basis").sort_values("tenor_years")
    tenors = tuple(float(t) for t in frame["tenor_years"])
    assert curve.tenors == tenors
    assert curve.basis_bp_at(tenors[0]) == pytest.approx(float(frame["basis_bp"].iloc[0]))
    assert curve.basis_bp_at(tenors[-1]) == pytest.approx(float(frame["basis_bp"].iloc[-1]))
    # Market-plausible sign and shape: negative and widening.
    assert all(b < 0.0 for b in curve.basis_bp)
    assert curve.basis_bp[-1] < curve.basis_bp[0]


def test_basis_curve_from_snapshot_rejects_a_wrong_asof(snapshot: Snapshot) -> None:
    with pytest.raises(XccyError, match="differs"):
        basis_curve_from_snapshot(snapshot, date(2020, 1, 1))


def test_eur_discount_curve_reproduces_ecb_zero_rates(snapshot: Snapshot) -> None:
    curve = eur_discount_curve(snapshot, REFERENCE)
    spot = snapshot.load("ecb_spot_curve").sort_values("tenor_years")
    for tenor, zero in zip(spot["tenor_years"], spot["zero_rate"], strict=True):
        assert curve.zero(float(tenor)) == pytest.approx(float(zero), rel=1e-12)


# -- Swap construction and NPV difference --------------------------------------


def test_usd_swap_lands_on_the_anniversary_with_a_3m_float_leg() -> None:
    swap = usd_swap(REFERENCE, 5.0, fixed_rate=0.04)
    assert swap.maturity == date(2031, 7, 24)  # CORE-01: 5Y anniversary
    assert swap.float_tenor == "3M"
    assert swap.notional == 1_000_000.0
    assert swap.fixed_rate == 0.04


def test_swap_npv_difference_is_zero_for_a_zero_basis() -> None:
    base = CurveSet.single(_FLAT_04)
    swap = usd_swap(REFERENCE, 5.0, fixed_rate=0.04)
    result = swap_npv_difference(swap, base, _constant_basis(0.0), REFERENCE)
    assert result.delta == pytest.approx(0.0, abs=1e-6)


def test_swap_npv_difference_magnitude_grows_with_the_basis() -> None:
    base = CurveSet.single(_FLAT_04)
    swap = usd_swap(REFERENCE, 5.0, fixed_rate=0.04)
    at_minus_20 = swap_npv_difference(swap, base, _constant_basis(-20.0), REFERENCE).delta
    at_minus_40 = swap_npv_difference(swap, base, _constant_basis(-40.0), REFERENCE).delta
    assert at_minus_20 != pytest.approx(0.0)
    assert abs(at_minus_40) > abs(at_minus_20)


def test_swap_npv_difference_only_shifts_the_discount_curve() -> None:
    """The forecast leg is untouched: the CSA valuation shares the base's forecast map."""
    base = CurveSet(
        discount=_FLAT_04, forecast={"3M": FlatCurve(reference_date=REFERENCE, rate=0.05)}
    )
    swap = usd_swap(REFERENCE, 5.0, fixed_rate=0.04)
    result = swap_npv_difference(swap, base, _constant_basis(0.0), REFERENCE)
    assert result.base_npv == pytest.approx(result.csa_npv, rel=1e-12)


def test_csa_discounting_shifts_a_bill_npv_by_the_basis_closed_form() -> None:
    """Hand-derived NPV reference for a single cash flow.

    A 100-face bill at T (5y anniversary), flat 4% base, constant -20 bp basis.
    With curve time ``tau = (T - asof).days / 365``: base NPV = 100 * exp(-0.04
    * tau); the CSA curve discounts at rate 0.042, so CSA NPV = base * exp(-0.002
    * tau); delta = base * (exp(-0.002 * tau) - 1).
    """
    bill = Bill(maturity=date(2031, 7, 24), face=100.0)
    tau = curve_time(REFERENCE, bill.maturity)
    base = CurveSet.single(_FLAT_04)
    csa = CurveSet.single(CsaDiscountCurve(base=_FLAT_04, basis=_constant_basis(-20.0)))

    base_npv = price(bill, base, REFERENCE).dirty
    csa_npv = price(bill, csa, REFERENCE).dirty

    expected_base = 100.0 * math.exp(-0.04 * tau)
    expected_delta = expected_base * (math.exp(-0.002 * tau) - 1.0)
    assert base_npv == pytest.approx(expected_base, rel=1e-12)
    assert csa_npv - base_npv == pytest.approx(expected_delta, rel=1e-12)
