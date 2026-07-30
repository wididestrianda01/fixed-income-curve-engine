"""Bachelier (normal) option pricing.

The SOFR swaption market quotes normal volatilities in basis points, and has
done since rates went through zero: a lognormal vol is undefined at a
non-positive forward, and both the forward and the strike can be non-positive
in this market. Bachelier is therefore not a simplification here, it is the
convention.

    payer = (F - K) N(d) + v n(d),  d = (F - K) / v,  v = sigma sqrt(T)
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

_MAX_VOL = 1.0


def bachelier_price(
    forward: float, strike: float, normal_vol: float, expiry: float, *, pay: bool
) -> float:
    """Undiscounted option value. Multiply by the annuity for a swaption."""
    if expiry < 0.0:
        raise ValueError(f"expiry must be non-negative, got {expiry}")
    if normal_vol < 0.0:
        raise ValueError(f"normal_vol must be non-negative, got {normal_vol}")
    moneyness = (forward - strike) if pay else (strike - forward)
    v = normal_vol * math.sqrt(expiry)
    if v == 0.0:
        return max(moneyness, 0.0)
    d = moneyness / v
    return moneyness * float(norm.cdf(d)) + v * float(norm.pdf(d))


def bachelier_vol(
    price: float, forward: float, strike: float, expiry: float, *, pay: bool
) -> float:
    """Implied normal volatility. Brent on a bracket."""
    intrinsic = max((forward - strike) if pay else (strike - forward), 0.0)
    if price < intrinsic - 1e-14:
        raise ValueError(
            f"price {price} is below intrinsic {intrinsic}; no non-negative volatility produces it"
        )
    if abs(price - intrinsic) < 1e-14:
        return 0.0
    return float(
        brentq(
            lambda v: bachelier_price(forward, strike, v, expiry, pay=pay) - price,
            1e-12,
            _MAX_VOL,
            xtol=1e-14,
        )
    )
