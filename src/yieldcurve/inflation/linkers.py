"""Inflation-linked bonds and zero-coupon inflation swaps.

An inflation-linked bond ("linker") pays coupons and principal that are *real*
(deflated) amounts, scaled into nominal currency by the ratio of a price index
at the payment date to the same index at the bond's base date:

    nominal_amount(d) = real_amount * I(d) / I(base)

where ``I(d) / I(base)`` is the indexation ratio. The index used for a payment
is observed with a lag — the CPI for a month is published several weeks after
the month closes, so a payment dated ``d`` is settled against the index
published ``indexation_lag_months`` earlier. This module applies the lag as a
*shift of the index observation date*: the ratio at payment ``d`` is projected
from the index date ``d - lag``. That is a documented simplification: a real
desk would interpolate the CPI fixing inside the lag, whereas here the
projected forward index is evaluated at the lagged date. The lag is the only
reason the real-curve price and the nominal-curve price of the indexed
cashflows differ; with ``indexation_lag_months = 0`` they coincide by the
Fisher identity.

A zero-coupon (ZC) inflation swap exchanges, at maturity ``T``, the fixed
return ``(1 + K)^T - 1`` on the fixed leg for the realised inflation
``I(T)/I(T0) - 1`` on the floating leg (continuous form: ``exp(K T) - 1``
against ``exp(b(T) T) - 1``). Its par breakeven is therefore exactly the
zero-coupon breakeven at ``T``, and the legs are priced by discounting those
terminal amounts on the nominal curve.

All projected index levels come from the breakeven curve, so every number here
is a *relative* inflation price, never a CPI forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from yieldcurve.conventions import add_months
from yieldcurve.curves.protocol import DiscountCurve, curve_time
from yieldcurve.inflation.curve import BreakevenCurve, RealRateCurve


class LinkerError(ValueError):
    """A linker or inflation swap was constructed with inputs it cannot support."""


@dataclass(frozen=True)
class InflationLinkedBond:
    """A bullet linker whose real cashflows are indexed by a CPI ratio.

    ``base_date`` is the indexation base (the date of ``base_index``);
    ``coupon`` is the annual *real* coupon as a decimal; ``frequency`` is
    coupons per year. Cashflows are real amounts scaled by ``I(d)/I(base)``.
    """

    base_date: date
    maturity: date
    face: float = 100.0
    coupon: float = 0.02
    frequency: int = 2
    base_index: float = 100.0
    indexation_lag_months: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.face) or self.face <= 0.0:
            raise LinkerError(f"face must be positive and finite, got {self.face}")
        if not math.isfinite(self.coupon) or self.coupon < 0.0:
            raise LinkerError(f"coupon must be non-negative and finite, got {self.coupon}")
        if self.frequency <= 0 or 12 % self.frequency != 0:
            raise LinkerError(f"frequency must divide 12 evenly, got {self.frequency}")
        if not math.isfinite(self.base_index) or self.base_index <= 0.0:
            raise LinkerError(f"base_index must be positive and finite, got {self.base_index}")
        if self.indexation_lag_months < 0:
            raise LinkerError(
                f"indexation_lag_months must be non-negative, got {self.indexation_lag_months}"
            )
        if self.maturity <= self.base_date:
            raise LinkerError(
                f"maturity {self.maturity} must fall after base_date {self.base_date}"
            )

    def real_cashflows(self) -> tuple[tuple[date, float], ...]:
        """``(payment_date, real_amount)`` pairs, real (deflated) terms.

        Coupons at each period end, with the final period's coupon bundled into
        the maturity flow together with the face, matching the repository's
        bullet-bond convention.
        """
        step = 12 // self.frequency
        coupon_amount = self.face * self.coupon / self.frequency
        flows: list[tuple[date, float]] = []
        d = add_months(self.base_date, step)
        while d < self.maturity:
            flows.append((d, coupon_amount))
            d = add_months(d, step)
        flows.append((self.maturity, coupon_amount + self.face))
        return tuple(flows)


@dataclass(frozen=True)
class LinkerPrice:
    """A linker's value in both conventions.

    ``real_price`` discounts the real cashflows on the real curve;
    ``nominal_price`` discounts the projected-indexed cashflows on the nominal
    curve. They coincide when the indexation lag is zero and differ by the lag
    otherwise. ``maturity_index_ratio`` is the projected ``I(T - lag)/I(base)``.
    """

    real_price: float
    nominal_price: float
    maturity_index_ratio: float


def index_ratio(breakeven: BreakevenCurve, base_date: date, index_date: date) -> float:
    """Projected indexation ratio ``I(index_date) / I(base_date)``.

    The forward index is ``exp(b(t) t)`` with ``t`` the curve time from the
    base date to the index date, so the ratio is ``exp(b(t) t)`` and equals 1
    at the base date. This is a relative inflation price implied by the
    breakeven curve, not a CPI forecast.
    """
    t = curve_time(base_date, index_date)
    if t < 0.0:
        raise LinkerError(
            f"index date {index_date} precedes base date {base_date} (curve time {t})"
        )
    return math.exp(breakeven.breakeven(t) * t)


def price_linker(bond: InflationLinkedBond, real_curve: RealRateCurve, asof: date) -> LinkerPrice:
    """Price ``bond`` off the real curve, with the indexation lag applied.

    ``asof`` is the valuation date (must equal the curve reference date). Real
    cashflows are discounted on the real curve; the indexed cashflows are
    discounted on the nominal curve with the lag shifted index date, so the two
    prices differ by exactly the lag's effect.
    """
    ref = real_curve.reference_date
    if asof != ref:
        raise LinkerError(f"asof {asof} must equal the curve reference date {ref}")
    nominal = real_curve.nominal
    breakeven = real_curve.breakeven

    real_price = 0.0
    nominal_price = 0.0
    for d, real_amount in bond.real_cashflows():
        t_d = curve_time(ref, d)
        real_price += real_amount * real_curve.df(t_d)
        index_date = add_months(d, -bond.indexation_lag_months)
        ratio = index_ratio(breakeven, bond.base_date, index_date)
        nominal_price += real_amount * ratio * nominal.df(t_d)

    maturity_index_ratio = index_ratio(
        breakeven, bond.base_date, add_months(bond.maturity, -bond.indexation_lag_months)
    )
    return LinkerPrice(
        real_price=real_price,
        nominal_price=nominal_price,
        maturity_index_ratio=maturity_index_ratio,
    )


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """A spot-starting ZC inflation swap.

    The fixed leg pays ``notional * (exp(K T) - 1)`` and the floating leg
    ``notional * (exp(b(T) T) - 1)`` at maturity, both in nominal currency,
    where ``K`` is ``fixed_breakeven`` and ``T`` the tenor in years.
    """

    start_date: date
    maturity: date
    notional: float = 1.0
    fixed_breakeven: float = 0.02

    def __post_init__(self) -> None:
        if not math.isfinite(self.notional) or self.notional <= 0.0:
            raise LinkerError(f"notional must be positive and finite, got {self.notional}")
        if not math.isfinite(self.fixed_breakeven):
            raise LinkerError(f"fixed_breakeven must be finite, got {self.fixed_breakeven}")
        if self.maturity <= self.start_date:
            raise LinkerError(
                f"maturity {self.maturity} must fall after start_date {self.start_date}"
            )


@dataclass(frozen=True)
class ZcSwapLegs:
    """The two legs of a ZC inflation swap and its par breakeven.

    ``fixed_leg_pv`` and ``floating_leg_pv`` are nominal present values;
    ``par_breakeven`` is the fixed rate making the swap worth zero;
    ``net_pv`` is ``fixed_leg_pv - floating_leg_pv`` (the payer of fixed).
    """

    fixed_leg_pv: float
    floating_leg_pv: float
    par_breakeven: float
    net_pv: float


def zc_swap_par_breakeven(breakeven: BreakevenCurve, start_date: date, maturity: date) -> float:
    """The fixed breakeven making a spot-starting ZC inflation swap worth zero.

    At par the fixed and floating terminal amounts coincide, which requires
    ``K = b(T)`` — the zero-coupon breakeven at the swap's tenor.
    """
    if start_date != breakeven.reference_date:
        raise LinkerError(
            "only spot-starting ZC inflation swaps are supported: start_date "
            f"{start_date} must equal the breakeven reference date {breakeven.reference_date}"
        )
    return breakeven.breakeven(curve_time(start_date, maturity))


def zc_swap_legs(
    swap: ZeroCouponInflationSwap, nominal: DiscountCurve, breakeven: BreakevenCurve
) -> ZcSwapLegs:
    """Value both legs of a spot-starting ZC inflation swap on the nominal curve."""
    ref = nominal.reference_date
    if breakeven.reference_date != ref:
        raise LinkerError(
            "nominal and breakeven curves must share one reference date: "
            f"{ref} != {breakeven.reference_date}"
        )
    if swap.start_date != ref:
        raise LinkerError(
            "only spot-starting ZC inflation swaps are supported: start_date "
            f"{swap.start_date} must equal the curve reference date {ref}"
        )
    t = curve_time(ref, swap.maturity)
    df = nominal.df(t)
    par = breakeven.breakeven(t)
    fixed = swap.notional * (math.exp(swap.fixed_breakeven * t) - 1.0) * df
    floating = swap.notional * (math.exp(par * t) - 1.0) * df
    return ZcSwapLegs(
        fixed_leg_pv=fixed,
        floating_leg_pv=floating,
        par_breakeven=par,
        net_pv=fixed - floating,
    )
