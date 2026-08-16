"""Inflation analytics: breakeven curves, real-rate curves, linkers, ZC swaps.

This subpackage demonstrates the *relative* pricing of inflation-linked
cashflows: the breakeven curve is the spread between a nominal and a real
curve, and every projected index level here is a relative price, never a CPI
forecast. Linkers and zero-coupon inflation swaps price off a real curve that
satisfies the repository's ``DiscountCurve`` contract.
"""

from yieldcurve.inflation.curve import BreakevenCurve, InflationError, RealRateCurve
from yieldcurve.inflation.linkers import (
    InflationLinkedBond,
    LinkerError,
    LinkerPrice,
    ZcSwapLegs,
    ZeroCouponInflationSwap,
    index_ratio,
    price_linker,
    zc_swap_legs,
    zc_swap_par_breakeven,
)

__all__ = [
    "BreakevenCurve",
    "InflationError",
    "InflationLinkedBond",
    "LinkerError",
    "LinkerPrice",
    "RealRateCurve",
    "ZcSwapLegs",
    "ZeroCouponInflationSwap",
    "index_ratio",
    "price_linker",
    "zc_swap_legs",
    "zc_swap_par_breakeven",
]
