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
# Relative tolerance for judging a price against intrinsic or against the
# maximum price attainable at ``_MAX_VOL``. Float noise from a quotient (e.g.
# an NPV divided by an annuity) is absorbed; genuinely impossible prices are
# not.
_BOUNDARY_TOL_REL = 1e-12


class BachelierError(ValueError):
    """The option data cannot be represented under the Bachelier convention."""


def _check_finite(price: float, forward: float, strike: float, expiry: float) -> None:
    if not all(math.isfinite(x) for x in (price, forward, strike, expiry)):
        raise BachelierError(
            f"all inputs must be finite, got price={price}, forward={forward}, "
            f"strike={strike}, expiry={expiry}"
        )


def bachelier_price(
    forward: float, strike: float, normal_vol: float, expiry: float, *, pay: bool
) -> float:
    """Undiscounted option value. Multiply by the annuity for a swaption."""
    if not all(math.isfinite(x) for x in (forward, strike, normal_vol, expiry)):
        raise BachelierError(
            f"all inputs must be finite, got forward={forward}, strike={strike}, "
            f"normal_vol={normal_vol}, expiry={expiry}"
        )
    if expiry < 0.0:
        raise BachelierError(f"expiry must be non-negative, got {expiry}")
    if normal_vol < 0.0:
        raise BachelierError(f"normal_vol must be non-negative, got {normal_vol}")
    moneyness = (forward - strike) if pay else (strike - forward)
    v = normal_vol * math.sqrt(expiry)
    if v == 0.0:
        return max(moneyness, 0.0)
    d = moneyness / v
    return moneyness * float(norm.cdf(d)) + v * float(norm.pdf(d))


def bachelier_vol(
    price: float, forward: float, strike: float, expiry: float, *, pay: bool
) -> float:
    """Implied normal volatility, inverting the Bachelier price on [0, _MAX_VOL].

    Expiry is validated before any intrinsic shortcut, and the no-solution
    boundaries are reported explicitly: a price below intrinsic, a price above
    the maximum attainable at ``_MAX_VOL``, or a non-zero price at zero expiry
    raise ``BachelierError`` instead of failing inside the root finder.
    """
    _check_finite(price, forward, strike, expiry)
    if expiry < 0.0:
        raise BachelierError(f"expiry must be non-negative, got {expiry}")
    moneyness = (forward - strike) if pay else (strike - forward)
    intrinsic = max(moneyness, 0.0)
    scale = max(1.0, abs(price), abs(intrinsic))
    tolerance = _BOUNDARY_TOL_REL * scale
    if price < intrinsic - tolerance:
        raise BachelierError(
            f"price {price} is below intrinsic {intrinsic}; no non-negative volatility produces it"
        )
    if expiry == 0.0:
        if price > intrinsic + tolerance:
            raise BachelierError(
                f"expiry is 0, so every volatility prices the option at its intrinsic "
                f"value {intrinsic}; no volatility produces price {price}"
            )
        return 0.0
    max_price = bachelier_price(forward, strike, _MAX_VOL, expiry, pay=pay)
    if price > max_price + _BOUNDARY_TOL_REL * max(1.0, abs(price), abs(max_price)):
        raise BachelierError(
            f"price {price} exceeds the maximum price {max_price} attainable at the "
            f"largest supported normal vol {_MAX_VOL}; no solution"
        )
    # No near-intrinsic shortcut: a price just above intrinsic corresponds to a
    # tiny but real volatility, which must not be collapsed to zero. The root
    # finder starts at zero, so an exactly-intrinsic price still yields 0.0.
    return float(
        brentq(
            lambda v: bachelier_price(forward, strike, v, expiry, pay=pay) - price,
            0.0,
            _MAX_VOL,
            xtol=1e-14,
        )
    )
