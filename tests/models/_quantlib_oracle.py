"""QuantLib Hull-White pricing oracle shared by the Hull-White model tests.

The swaption NPV comes from QuantLib's Jamshidian engine, so the price is
independent of this repository's Hull-White implementation. The conversion to
a normal vol reuses the production Bachelier inversion (``bachelier_vol``), so
the oracle and the model agree at the intrinsic boundary by construction: a
near-intrinsic price is inverted to its tiny real vol, never collapsed to
zero.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from yieldcurve.curves.pricing import annuity, par_rate
from yieldcurve.curves.protocol import CurveSet, DiscountCurve, curve_time
from yieldcurve.instruments import Swaption
from yieldcurve.models.bachelier import bachelier_vol


def quantlib_jamshidian_npv(
    ql: Any, swaption: Swaption, a: float, sigma: float, asof: date
) -> float:
    """QuantLib Jamshidian NPV of ``swaption`` under Hull-White(a, sigma).

    The flat 3% term structure matches the fixture curves the tests build; the
    engine is QuantLib's own, so the price is independent of the Hull-White
    implementation under test.
    """
    ql_date = ql.Date(asof.day, asof.month, asof.year)
    ql.Settings.instance().evaluationDate = ql_date
    handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_date, 0.03, ql.Actual365Fixed(), ql.Continuous)
    )
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    start = ql.Date(swaption.expiry.day, swaption.expiry.month, swaption.expiry.year)
    end = ql.Date(
        swaption.swap.maturity.day, swaption.swap.maturity.month, swaption.swap.maturity.year
    )

    def _schedule(tenor: Any) -> Any:
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
        _schedule(ql.Period(ql.Semiannual)),
        float(swaption.strike),
        ql.Thirty360(ql.Thirty360.BondBasis),
        _schedule(ql.Period(3, ql.Months)),
        index,
        0.0,
        ql.Actual360(),
    )
    theirs = ql.Swaption(swap, ql.EuropeanExercise(start))
    theirs.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(handle, a, sigma), handle))
    return float(theirs.NPV())


def quantlib_normal_vol(
    ql: Any, curve: DiscountCurve, swaption: Swaption, a: float, sigma: float, asof: date
) -> float:
    """Market normal vol for ``swaption`` from QuantLib's Hull-White engine.

    The undiscounted price is the QuantLib Jamshidian NPV divided by the
    annuity; the inversion reuses production's ``bachelier_vol`` boundary
    contract (a price within tolerance of intrinsic is inverted to its tiny
    real vol rather than collapsed, and impossible prices raise).
    """
    undiscounted = quantlib_jamshidian_npv(ql, swaption, a, sigma, asof) / (
        float(swaption.swap.notional) * annuity(swaption.swap, CurveSet.single(curve), asof)
    )
    # the forward is a plain curve quantity; QuantLib's own fairRate() cannot
    # be queried on an engine-less swap in this build, so ours stands in
    forward = par_rate(swaption.swap, CurveSet.single(curve), asof)
    expiry = curve_time(asof, swaption.expiry)
    return bachelier_vol(
        undiscounted, forward, float(swaption.strike), expiry, pay=swaption.pay_fixed
    )
