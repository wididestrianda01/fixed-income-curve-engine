"""Curve scenarios and the single shift primitive.

Everything in ``curveengine.risk`` that needs a moved curve calls
``shift_curve``. Effective duration, key-rate duration, PCA duration and the
regulatory scenario P&L are then the same computation with a different
``Scenario`` — which is what makes ``sum(krd) == effective_duration`` an
identity rather than a coincidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from curveengine.curves.protocol import CurveSet, DiscountCurve


@dataclass(frozen=True)
class Scenario:
    """A shift of the continuously compounded zero curve.

    ``shift(t)`` is in decimals at curve time ``t`` in ACT/365F years.
    ``citation`` carries the source for regulatory scenarios and is empty for
    ad-hoc ones; ``scenarios.toml`` requires it to be non-empty.
    """

    name: str
    shift: Callable[[float], float]
    citation: str = ""


@dataclass(frozen=True)
class _ShiftedCurve:
    """A DiscountCurve view over a base curve plus a zero-rate shift.

    Structural typing means this satisfies the DiscountCurve Protocol without
    inheriting from anything, so it composes with itself and with anything else
    the protocol accepts. Nothing is precomputed: shifting is O(1) and lazy,
    which matters when a PCA run shifts the same curve a few thousand times.
    """

    base: DiscountCurve
    scenario: Scenario

    @property
    def reference_date(self) -> date:
        return self.base.reference_date

    def df(self, t: float) -> float:
        return self.base.df(t) * math.exp(-self.scenario.shift(t) * t)

    def zero(self, t: float) -> float:
        return self.base.zero(t) + self.scenario.shift(t)

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"fwd requires t2 > t1, got t1={t1}, t2={t2}")
        return -(math.log(self.df(t2)) - math.log(self.df(t1))) / (t2 - t1)


def shift_curve(curve: DiscountCurve, scenario: Scenario) -> DiscountCurve:
    """Apply a scenario to one curve. Returns a new curve; nothing mutates."""
    return _ShiftedCurve(base=curve, scenario=scenario)


def shift_curveset(curves: CurveSet, scenario: Scenario) -> CurveSet:
    """Apply one scenario to the discount curve and every forecast curve.

    Shocking only the discount curve would hold the basis fixed in absolute
    terms, which is a *different* scenario — a rate shock plus an offsetting
    basis shock. Shifting both is the plain reading of a rate shock.
    """
    try:
        forecast = {tenor: shift_curve(curve, scenario) for tenor, curve in curves.forecast.items()}
    except NotImplementedError:
        # CurveSet.single: the forecast mapping answers every tenor with the
        # discount curve and refuses to enumerate. Shifting the discount curve
        # has already shifted the forecast curve, because they are one curve.
        return CurveSet.single(shift_curve(curves.discount, scenario))
    return CurveSet(discount=shift_curve(curves.discount, scenario), forecast=forecast)


def parallel(size: float) -> Scenario:
    """A flat shift of ``size`` decimals at every tenor."""
    sign = "+" if size >= 0 else "-"
    return Scenario(name=f"parallel {sign}{abs(size) * 1e4:.0f}bp", shift=lambda _t: size)
