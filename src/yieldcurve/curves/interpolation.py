"""Interpolated discount curves.

Interpolation happens on the log of the discount factor rather than on the zero
rate. Both pass through the same knots, so the difference is invisible in a plot
of zeros — and highly visible in a plot of forwards, which is where an
interpolation choice does its damage. Notebook 02 shows both plots side by side
for exactly that reason.

Three schemes are offered:

* ``LOG_LINEAR_DF`` — the market default. Cheap, always monotone, and produces a
  sawtooth forward curve.
* ``CUBIC_LOG_DF`` — smooth forwards, but a cubic spline can overshoot and is not
  guaranteed monotone, so a discount factor can in principle rise.
* ``MONOTONE_CONVEX`` — Hagan and West (2006). Continuous forwards and monotone
  discount factors at once.

One deliberate deviation from Hagan-West: their *positivity* amendment, which
clamps interpolated instantaneous forwards to be non-negative, is not
implemented. It was written for a world without negative rates. SEK and EUR
forwards have been negative within the sample period this project uses, and
clamping them would silently distort the curve. Their *monotonicity* amendments —
the four-region construction below — are implemented in full.
"""

from __future__ import annotations

import itertools
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np
from scipy.interpolate import CubicSpline


class CurveConstructionError(ValueError):
    """A curve was asked to exist with inputs that cannot define one."""


class InterpMethod(StrEnum):
    """Interpolation schemes, all acting on log discount factors."""

    LOG_LINEAR_DF = "Log-linear on discount factors"
    CUBIC_LOG_DF = "Cubic spline on log discount factors"
    MONOTONE_CONVEX = "Hagan-West monotone convex"


@dataclass(frozen=True)
class InterpolatedDiscountCurve:
    """A discount curve defined by knots and an interpolation scheme.

    ``times`` are ACT/365F years from ``reference_date``, strictly increasing and
    strictly positive. The point ``t = 0, df = 1`` is implicit.
    """

    reference_date: date
    times: tuple[float, ...]
    dfs: tuple[float, ...]
    method: InterpMethod

    # Derived once in __post_init__. The curve is immutable, so everything below is
    # a pure function of the knots — and the bootstrap rebuilds a curve on every
    # root-finding iteration, which makes recomputing any of it per call expensive.
    _knots: tuple[float, ...] = field(init=False, repr=False, compare=False, default=())
    _logs: tuple[float, ...] = field(init=False, repr=False, compare=False, default=())
    _zero_rates: tuple[float, ...] = field(init=False, repr=False, compare=False, default=())
    _spline: CubicSpline | None = field(init=False, repr=False, compare=False, default=None)
    _forwards: tuple[tuple[float, ...], tuple[float, ...]] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        if len(self.times) != len(self.dfs):
            raise CurveConstructionError(
                f"{len(self.times)} times but {len(self.dfs)} discount factors"
            )
        if not self.times:
            raise CurveConstructionError("A curve needs at least one knot")
        if any(t <= 0.0 for t in self.times):
            raise CurveConstructionError(
                "Knot times must be positive; t = 0 with df = 1 is implicit and must "
                "not be supplied"
            )
        if any(later <= earlier for earlier, later in itertools.pairwise(self.times)):
            raise CurveConstructionError(f"Knot times must be strictly increasing: {self.times}")
        if any(df <= 0.0 for df in self.dfs):
            raise CurveConstructionError(f"Discount factors must be positive: {self.dfs}")
        knots = (0.0, *self.times)
        logs = (0.0, *(math.log(df) for df in self.dfs))
        object.__setattr__(self, "_knots", knots)
        object.__setattr__(self, "_logs", logs)
        object.__setattr__(
            self, "_zero_rates", tuple(-lg / t for t, lg in zip(self.times, logs[1:], strict=True))
        )
        if self.method is InterpMethod.CUBIC_LOG_DF:
            object.__setattr__(
                self, "_spline", CubicSpline(np.array(knots), np.array(logs), bc_type="natural")
            )
        elif self.method is InterpMethod.MONOTONE_CONVEX:
            object.__setattr__(self, "_forwards", _monotone_convex_forwards(knots, logs))

    # --- the DiscountCurve contract -------------------------------------------

    def df(self, t: float) -> float:
        if t < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t}")
        if t == 0.0:
            return 1.0
        return math.exp(self._log_df_at(t))

    def zero(self, t: float) -> float:
        if t < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t}")
        if t == 0.0:
            return self._zero_rates[0]
        return -self._log_df_at(t) / t

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        if t1 < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t1}")
        return -(self._log_df_at(t2) - self._log_df_at(t1)) / (t2 - t1)

    def instantaneous_fwd(self, t: float) -> float:
        """The instantaneous forward at ``t``, by central difference.

        A numerical derivative rather than a closed form on purpose: it works
        identically for all three schemes, and the plots it feeds are visual.

        Computed as ``(zero(high)*high - zero(low)*low) / (high - low)`` rather
        than ``-(log(df(high)) - log(df(low))) / (high - low)`` to avoid the
        log(exp(x)) roundtrip that costs ~1e-10 in precision.
        """
        h = 1e-6
        low = max(t - h, 0.0)
        high = t + h
        return (self.zero(high) * high - self.zero(low) * low) / (high - low)

    # --- interpolation internals ----------------------------------------------

    def _log_df_at(self, t: float) -> float:
        """Log discount factor at any ``t > 0``, flat in the zero rate outside the knots."""
        if t <= self.times[0]:
            return -self._zero_rates[0] * t
        if t >= self.times[-1]:
            return -self._zero_rates[-1] * t
        return self._log_df(t)

    def _log_df(self, t: float) -> float:
        """Interpolated log discount factor, for ``t`` strictly inside the knots."""
        if self.method is InterpMethod.LOG_LINEAR_DF:
            return self._log_linear(t)
        if self.method is InterpMethod.CUBIC_LOG_DF:
            return self._cubic(t)
        return self._monotone_convex(t)

    def _log_linear(self, t: float) -> float:
        knots, logs = self._knots, self._logs
        i = bisect_left(knots, t)
        t0, t1 = knots[i - 1], knots[i]
        w = (t - t0) / (t1 - t0)
        return logs[i - 1] * (1.0 - w) + logs[i] * w

    def _cubic(self, t: float) -> float:
        assert self._spline is not None, "spline not built in __post_init__"
        return float(self._spline(t))

    def _monotone_convex(self, t: float) -> float:
        """Hagan-West monotone convex, integrated to a log discount factor.

        The construction, in the paper's notation:

        1. Discrete forwards over each interval,
           ``fd[i] = (r[i]*t[i] - r[i-1]*t[i-1]) / (t[i] - t[i-1])``.
        2. Instantaneous forwards at the knots, ``f[i]``, as the time-weighted
           average of the two adjacent discrete forwards, with one-sided
           end conditions.
        3. On each interval, an interpolant ``g(x)`` in ``x = (t - t[i-1])/dt``
           with ``g(0) = f[i-1] - fd[i]``, ``g(1) = f[i] - fd[i]`` and
           ``integral of g over [0,1] = 0``. The zero integral is what makes the
           scheme reprice its own knots exactly.
        4. Four regions select the functional form for ``g`` so that the
           resulting forward stays monotone between knots.
        """
        assert self._forwards is not None, "forwards not built in __post_init__"
        fd, f = self._forwards
        times = self._knots
        i = bisect_left(times, t)
        t0, t1 = times[i - 1], times[i]
        dt = t1 - t0
        x = (t - t0) / dt

        # The integral of the forward from 0 to t[i-1] is r[i-1]*t[i-1] = -log df[i-1],
        # already known at the knot; only the part inside the interval needs integrating.
        integral = dt * (fd[i] * x + _region_integral(f[i - 1] - fd[i], f[i] - fd[i], x))
        return self._logs[i - 1] - integral


def _monotone_convex_forwards(
    times: tuple[float, ...], logs: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Discrete interval forwards and instantaneous knot forwards, Hagan-West steps 1-2."""
    rt = [-lg for lg in logs]  # r[i] * t[i]
    n = len(times) - 1

    fd = [0.0] * (n + 1)
    for i in range(1, n + 1):
        fd[i] = (rt[i] - rt[i - 1]) / (times[i] - times[i - 1])

    f = [0.0] * (n + 1)
    for i in range(1, n):
        left = times[i] - times[i - 1]
        right = times[i + 1] - times[i]
        f[i] = (right * fd[i] + left * fd[i + 1]) / (left + right)
    f[0] = fd[1] - 0.5 * (f[1] - fd[1]) if n > 1 else fd[1]
    f[n] = fd[n] - 0.5 * (f[n - 1] - fd[n]) if n > 1 else fd[1]
    return tuple(fd), tuple(f)


def _region_integral(g0: float, g1: float, x: float) -> float:
    """``integral of g from 0 to x`` for the Hagan-West interpolant on one interval.

    ``g`` is chosen from four regions by the signs and ratio of ``g0`` and ``g1``.
    Every branch satisfies ``g(0) = g0``, ``g(1) = g1`` and
    ``integral of g over [0,1] = 0``; the branches differ in how they avoid a
    non-monotone forward in between.
    """
    if g0 == 0.0 and g1 == 0.0:
        return 0.0

    # Region (i): the plain quadratic, valid when g0 and g1 are compatible.
    if (g0 <= 0.0 and -0.5 * g0 <= g1 <= -2.0 * g0) or (g0 >= 0.0 and -0.5 * g0 >= g1 >= -2.0 * g0):
        return _region_one(g0, g1, x)

    same_sign = (g0 > 0.0 and g1 > 0.0) or (g0 < 0.0 and g1 < 0.0)
    if same_sign:
        # Region (iv): both ends on the same side; meet at an interior level A.
        eta = g1 / (g1 + g0)
        a = -g0 * g1 / (g0 + g1)
        if x <= eta:
            return a * x + (g0 - a) * (eta / 3.0) * (1.0 - ((eta - x) / eta) ** 3)
        head = a * eta + (g0 - a) * (eta / 3.0)
        return head + a * (x - eta) + (g1 - a) * (x - eta) ** 3 / (3.0 * (1.0 - eta) ** 2)

    if (g0 < 0.0 and g1 > -2.0 * g0) or (g0 > 0.0 and g1 < -2.0 * g0):
        # Region (ii): flat at g0, then a quadratic run-up to g1.
        eta = (g1 + 2.0 * g0) / (g1 - g0)
        if x <= eta:
            return g0 * x
        return g0 * x + (g1 - g0) * (x - eta) ** 3 / (3.0 * (1.0 - eta) ** 2)

    # Region (iii): a quadratic run-down from g0, then flat at g1.
    eta = 3.0 * g1 / (g1 - g0)
    if x <= eta:
        return g1 * x + (g0 - g1) * (eta / 3.0) * (1.0 - ((eta - x) / eta) ** 3)
    head = g1 * eta + (g0 - g1) * (eta / 3.0)
    return head + g1 * (x - eta)


def _region_one(g0: float, g1: float, x: float) -> float:
    """Integral of ``g(x) = g0*(1 - 4x + 3x^2) + g1*(-2x + 3x^2)``."""
    return g0 * (x - 2.0 * x**2 + x**3) + g1 * (-(x**2) + x**3)
