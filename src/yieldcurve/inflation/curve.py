"""Zero-coupon breakeven and real-rate curves.

The repository's curves are continuously compounded zero rates: a nominal curve
``n(T)`` discounts ``exp(-n(T) T)``. An inflation market adds one spread, the
*breakeven* ``b(T)`` — the inflation rate that equalises the nominal and real
return over ``[0, T]``. In continuous form the Fisher relation is additive:

    (1 + n_ann)^T = (1 + r_ann)^T (1 + b_ann)^T          (annual compounding)
    n(T) = r(T) + b(T)                                    (continuous compounding)

so the *real* zero rate is ``r(T) = n(T) - b(T)`` and the real discount factor
is ``exp(-r(T) T) = exp(-n(T) T) exp(b(T) T)``. A breakeven is a *spread
between two curves* (a relative price), not a forecast of future CPI: a quoted
breakeven prices the relative value of nominal and inflation-linked cashflows
and can move without any change in the expected CPI level.

:class:`BreakevenCurve` interpolates quoted zero-coupon breakevens;
:class:`RealRateCurve` composes a nominal ``DiscountCurve`` with a breakeven
curve into a curve that satisfies the ``DiscountCurve`` contract, so linkers
and ZC inflation swaps price off it through the ordinary discount-factor API.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from yieldcurve.curves.protocol import DiscountCurve


class InflationError(ValueError):
    """An inflation curve was constructed with inputs it cannot support."""


def _require_time(t: float) -> None:
    if not math.isfinite(t) or t < 0.0:
        raise InflationError(f"curve time must be finite and non-negative, got {t}")


@dataclass(frozen=True)
class BreakevenCurve:
    """A zero-coupon breakeven-inflation term structure.

    ``tenors`` are ACT/365F years from ``reference_date``, strictly increasing
    and strictly positive; ``breakevens`` are the matching continuously
    compounded zero-coupon breakeven rates as decimals (0.023 = 2.3%). Between
    knots the rate is interpolated linearly in the rate itself; beyond them it
    extrapolates flat (the same rule ``InterpolatedDiscountCurve`` applies to
    its zero rates). Breakevens may be negative — deflation is a legitimate
    market state — so only non-finite values are rejected, never their sign.
    """

    reference_date: date
    tenors: tuple[float, ...]
    breakevens: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.breakevens):
            raise InflationError(f"{len(self.tenors)} tenors but {len(self.breakevens)} breakevens")
        if not self.tenors:
            raise InflationError("A breakeven curve needs at least one tenor")
        if any(not math.isfinite(t) for t in self.tenors):
            raise InflationError(f"tenors must be finite: {self.tenors}")
        if any(t <= 0.0 for t in self.tenors):
            raise InflationError(f"tenors must be positive, got {self.tenors}")
        if any(later <= earlier for earlier, later in pairwise(self.tenors)):
            raise InflationError(f"tenors must be strictly increasing, got {self.tenors}")
        if any(not math.isfinite(b) for b in self.breakevens):
            raise InflationError(f"breakevens must be finite: {self.breakevens}")

    def breakeven(self, t: float) -> float:
        """The continuously compounded zero-coupon breakeven (decimal) at ``t``.

        Piecewise-linear in the rate between knots, flat beyond them. ``t`` is
        ACT/365F years from ``reference_date``.
        """
        _require_time(t)
        if t <= self.tenors[0]:
            return self.breakevens[0]
        if t >= self.tenors[-1]:
            return self.breakevens[-1]
        i = bisect_left(self.tenors, t)
        t0, t1 = self.tenors[i - 1], self.tenors[i]
        b0, b1 = self.breakevens[i - 1], self.breakevens[i]
        w = (t - t0) / (t1 - t0)
        return b0 * (1.0 - w) + b1 * w


@dataclass(frozen=True)
class RealRateCurve:
    """A zero-coupon *real* curve: nominal zero rate minus the breakeven.

    Implements the ``DiscountCurve`` contract so linkers and ZC inflation
    swaps price off it with the ordinary discount-factor API. The nominal and
    breakeven curves must share one reference date, so a real discount factor
    is a ratio in absolute curve time.
    """

    nominal: DiscountCurve
    breakeven: BreakevenCurve

    def __post_init__(self) -> None:
        if self.nominal.reference_date != self.breakeven.reference_date:
            raise InflationError(
                "nominal and breakeven curves must share one reference date: "
                f"{self.nominal.reference_date} != {self.breakeven.reference_date}"
            )

    @property
    def reference_date(self) -> date:
        return self.nominal.reference_date

    def zero(self, t: float) -> float:
        """The continuously compounded real zero rate: ``n(t) - b(t)``."""
        _require_time(t)
        return self.nominal.zero(t) - self.breakeven.breakeven(t)

    def df(self, t: float) -> float:
        """The real discount factor: ``exp(-(n(t) - b(t)) t)``."""
        return math.exp(-self.zero(t) * t)

    def fwd(self, t1: float, t2: float) -> float:
        """The continuously compounded real forward rate over ``[t1, t2]``."""
        _require_time(t1)
        if not math.isfinite(t2) or t2 <= t1:
            raise InflationError(f"t2 {t2} must be finite and exceed t1 {t1}")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)
