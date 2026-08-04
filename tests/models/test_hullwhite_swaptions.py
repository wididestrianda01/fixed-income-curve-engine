"""Jamshidian swaption decomposition under Hull-White."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from yieldcurve.calendars import USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import par_rate
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import Swaption, VanillaSwap
from yieldcurve.models.hullwhite import HullWhite, ModelError

ASOF = date(2026, 7, 24)
SEED = 20260727


@pytest.fixture
def model() -> HullWhite:
    return HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=0.05, sigma=0.01)


@pytest.fixture
def curves() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.03))


def _swaption(strike: float, *, pay: bool = True) -> Swaption:
    swap = VanillaSwap(
        start=date(2028, 7, 24),
        maturity=date(2033, 7, 24),
        fixed_rate=strike,
        fixed_frequency=2,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        notional=1.0,
    )
    return Swaption(expiry=date(2028, 7, 24), swap=swap, strike=strike, pay_fixed=pay)


def test_zero_coupon_bond_option_matches_a_monte_carlo_estimate(model: HullWhite) -> None:
    expiry, maturity, strike = 2.0, 5.0, 0.90
    analytic = model.zbo(expiry, maturity, strike, call=True)

    paths = model.simulate(list(np.linspace(0.0, expiry, 25)), 200_000, seed=SEED)
    r_expiry = paths[:, -1]
    integral = np.trapezoid(paths, x=np.linspace(0.0, expiry, 25), axis=1)
    payoff = np.maximum(
        np.array([model.zcb(expiry, maturity, float(r)) for r in r_expiry]) - strike, 0.0
    )
    estimates = np.exp(-integral) * payoff

    standard_error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    assert abs(analytic - estimates.mean()) < 4 * standard_error


def test_zbo_put_call_parity(model: HullWhite) -> None:
    call = model.zbo(2.0, 5.0, 0.90, call=True)
    put = model.zbo(2.0, 5.0, 0.90, call=False)

    assert call - put == pytest.approx(model.curve.df(5.0) - 0.90 * model.curve.df(2.0), abs=1e-12)


def test_payer_and_receiver_swaptions_satisfy_parity(model: HullWhite, curves: CurveSet) -> None:
    strike = 0.035
    payer = model.swaption(_swaption(strike, pay=True), ASOF)
    receiver = model.swaption(_swaption(strike, pay=False), ASOF)

    from yieldcurve.curves.pricing import price

    forward_swap = price(_swaption(strike).swap, curves, ASOF).dirty

    assert payer - receiver == pytest.approx(forward_swap, abs=1e-4)


def test_an_atm_swaption_is_worth_more_than_zero_and_less_than_the_annuity(
    model: HullWhite, curves: CurveSet
) -> None:
    from yieldcurve.curves.pricing import annuity

    atm = par_rate(_swaption(0.0).swap, curves, ASOF)
    value = model.swaption(_swaption(atm), ASOF)

    assert 0.0 < value < annuity(_swaption(atm).swap, curves, ASOF)


def test_swaption_value_rises_with_volatility(curves: CurveSet) -> None:
    values = [
        HullWhite(curve=curves.discount, a=0.05, sigma=s).swaption(_swaption(0.03), ASOF)
        for s in (0.004, 0.008, 0.016)
    ]

    assert all(b > a for a, b in zip(values, values[1:], strict=False))  # noqa: RUF007


def test_zero_volatility_gives_the_forward_swap_intrinsic(curves: CurveSet) -> None:
    from yieldcurve.curves.pricing import price

    model = HullWhite(curve=curves.discount, a=0.05, sigma=0.0)
    strike = 0.02
    intrinsic = max(price(_swaption(strike).swap, curves, ASOF).dirty, 0.0)

    assert model.swaption(_swaption(strike), ASOF) == pytest.approx(intrinsic, abs=1e-4)


def test_swaption_matches_a_monte_carlo_estimate(model: HullWhite, curves: CurveSet) -> None:
    strike = par_rate(_swaption(0.0).swap, curves, ASOF)
    analytic = model.swaption(_swaption(strike), ASOF)

    expiry = 2.0
    grid = list(np.linspace(0.0, expiry, 25))
    paths = model.simulate(grid, 100_000, seed=SEED)
    integral = np.trapezoid(paths, x=np.asarray(grid), axis=1)
    payoff = np.array(
        [
            max(model.forward_swap_value(_swaption(strike).swap, expiry, float(r), ASOF), 0.0)
            for r in paths[:, -1]
        ]
    )
    estimates = np.exp(-integral) * payoff

    standard_error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    assert abs(analytic - estimates.mean()) < 4 * standard_error


def test_normal_vol_matches_an_independent_quantlib_price(
    model: HullWhite, curves: CurveSet
) -> None:
    """The model vol is checked against QuantLib's Hull-White engine.

    A same-model round trip cannot detect a systematic pricing error, so the
    market-implied normal vol is rebuilt from QuantLib's Jamshidian NPV and the
    textbook Bachelier inversion instead.
    """
    ql = pytest.importorskip("QuantLib")

    from scipy.optimize import brentq
    from scipy.stats import norm as scipy_norm

    from yieldcurve.curves.pricing import annuity, par_rate
    from yieldcurve.curves.protocol import curve_time

    strike = 0.03
    swaption = _swaption(strike)
    a, sigma = model.a, model.sigma

    asof = ql.Date(24, 7, 2026)
    ql.Settings.instance().evaluationDate = asof
    handle = ql.YieldTermStructureHandle(
        ql.FlatForward(asof, 0.03, ql.Actual365Fixed(), ql.Continuous)
    )
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    start, end = ql.Date(24, 7, 2028), ql.Date(24, 7, 2033)

    def _schedule(tenor):  # type: ignore[no-untyped-def]
        return ql.Schedule(
            start,
            end,
            tenor,
            calendar,
            ql.ModifiedFollowing,
            ql.ModifiedFollowing,
            ql.DateGeneration.Backward,
            False,
        )

    index = ql.IborIndex(
        "Float3M",
        ql.Period(3, ql.Months),
        0,
        ql.USDCurrency(),
        calendar,
        ql.ModifiedFollowing,
        False,
        ql.Actual360(),
        handle,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        1.0,
        _schedule(ql.Period(ql.Semiannual)),  # type: ignore[no-untyped-call]
        strike,
        ql.Thirty360(ql.Thirty360.BondBasis),
        _schedule(ql.Period(3, ql.Months)),  # type: ignore[no-untyped-call]
        index,
        0.0,
        ql.Actual360(),
    )
    theirs = ql.Swaption(swap, ql.EuropeanExercise(start))
    theirs.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(handle, a, sigma), handle))

    # the forward is a plain curve quantity; QuantLib's own fairRate() cannot
    # be queried on an engine-less swap in this build, so ours stands in
    forward = par_rate(swaption.swap, curves, ASOF)
    expiry = curve_time(ASOF, swaption.expiry)
    undiscounted = theirs.NPV() / (swaption.swap.notional * annuity(swaption.swap, curves, ASOF))

    def _bachelier(v: float) -> float:
        moneyness = forward - strike
        s = v * math.sqrt(expiry)
        d = moneyness / s
        return moneyness * float(scipy_norm.cdf(d)) + s * float(scipy_norm.pdf(d))

    independent_vol = float(brentq(lambda v: _bachelier(v) - undiscounted, 1e-12, 1.0, xtol=1e-14))

    ours = model.swaption_normal_vol(swaption, ASOF)

    assert ours == pytest.approx(independent_vol, rel=1e-2)


def test_a_deeply_out_of_the_money_swaption_is_near_zero_but_positive(
    model: HullWhite,
) -> None:
    value = model.swaption(_swaption(0.15), ASOF)

    assert 0.0 < value < 1e-4


def test_a_swaption_with_a_strike_different_from_the_swap_fixed_rate_is_rejected(
    model: HullWhite,
) -> None:
    from yieldcurve.models.hullwhite import SwaptionStrikeError

    swaption = _swaption(0.03)
    mismatched = Swaption(expiry=swaption.expiry, swap=swaption.swap, strike=0.04, pay_fixed=True)

    with pytest.raises(SwaptionStrikeError, match=r"0\.04"):
        model.swaption(mismatched, ASOF)


def test_jamshidian_finds_a_far_root_by_widening_the_bracket() -> None:
    # Forward rates near 150% push the breakeven short rate above the initial
    # bracket [-0.5, 1.0]; the root solver must widen the bracket dynamically.
    curve = FlatCurve(reference_date=ASOF, rate=1.5)
    model = HullWhite(curve=curve, a=0.05, sigma=0.01)
    swap = VanillaSwap(
        start=date(2027, 7, 24),
        maturity=date(2028, 7, 24),
        fixed_rate=2.0,
        fixed_frequency=2,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        notional=1.0,
    )
    swaption = Swaption(expiry=date(2027, 7, 24), swap=swap, strike=2.0, pay_fixed=True)

    value = model.swaption(swaption, ASOF)

    assert math.isfinite(value)
    assert value > 0.0


def test_jamshidian_rejects_negative_cash_flow_coefficients_with_trade_context(
    model: HullWhite,
) -> None:
    # strike == fixed_rate == -2% is internally consistent, but the negative
    # coupons break the Jamshidian decomposition, which needs non-negative
    # coefficients
    swaption = _swaption(-0.02)

    with pytest.raises(ModelError, match="non-negative"):
        model.swaption(swaption, ASOF)


def test_quantlib_swaption_parity() -> None:
    ql = pytest.importorskip("QuantLib")

    a, sigma, strike = 0.05, 0.01, 0.03

    asof = ql.Date(24, 7, 2026)
    ql.Settings.instance().evaluationDate = asof

    handle = ql.YieldTermStructureHandle(
        ql.FlatForward(asof, 0.03, ql.Actual365Fixed(), ql.Continuous)
    )

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    start, end = ql.Date(24, 7, 2028), ql.Date(24, 7, 2033)

    def _schedule(tenor):  # type: ignore[no-untyped-def]
        return ql.Schedule(
            start,
            end,
            tenor,
            calendar,
            ql.ModifiedFollowing,
            ql.ModifiedFollowing,
            ql.DateGeneration.Backward,
            False,
        )

    index = ql.IborIndex(
        "Float3M",
        ql.Period(3, ql.Months),
        0,
        ql.USDCurrency(),
        calendar,
        ql.ModifiedFollowing,
        False,
        ql.Actual360(),
        handle,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        1.0,
        _schedule(ql.Period(ql.Semiannual)),  # type: ignore[no-untyped-call]
        strike,
        ql.Thirty360(ql.Thirty360.BondBasis),
        _schedule(ql.Period(3, ql.Months)),  # type: ignore[no-untyped-call]
        index,
        0.0,
        ql.Actual360(),
    )
    theirs = ql.Swaption(swap, ql.EuropeanExercise(start))
    theirs.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(handle, a, sigma), handle))

    ours = HullWhite(curve=FlatCurve(reference_date=ASOF, rate=0.03), a=a, sigma=sigma).swaption(
        _swaption(strike), ASOF
    )

    assert ours == pytest.approx(theirs.NPV(), abs=1e-4)
