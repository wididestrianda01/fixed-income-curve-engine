"""The one contract the whole library is built on.

A curve is anything that can produce a discount factor for a time. It is a
``Protocol`` rather than a base class deliberately: a bootstrapped curve, a
Svensson curve and a shocked curve all satisfy it without any of them importing
the others, and mypy checks the conformance structurally.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

_DAYS_PER_YEAR = 365.0


def curve_time(reference_date: date, d: date) -> float:
    """Curve time in ACT/365F years. The only date-to-time conversion in the library."""
    return (d - reference_date).days / _DAYS_PER_YEAR


@runtime_checkable
class DiscountCurve(Protocol):
    """A term structure of discount factors.

    ``t`` is always ACT/365F years from ``reference_date``; ``zero`` and ``fwd``
    return continuously compounded rates.
    """

    @property
    def reference_date(self) -> date: ...

    def df(self, t: float) -> float: ...

    def zero(self, t: float) -> float: ...

    def fwd(self, t1: float, t2: float) -> float: ...


@dataclass(frozen=True)
class FlatCurve:
    """A constant continuously compounded rate. Analytic, so tests can assert
    exact values and isolate pricing errors from interpolation errors."""

    reference_date: date
    rate: float

    def df(self, t: float) -> float:
        _check_time(t)
        return math.exp(-self.rate * t)

    def zero(self, t: float) -> float:
        _check_time(t)
        return self.rate

    def fwd(self, t1: float, t2: float) -> float:
        _check_time(t1)
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        return self.rate


def _check_time(t: float) -> None:
    if t < 0.0:
        raise ValueError(f"Curve time must be non-negative, got {t}")


@dataclass(frozen=True)
class CurveSet:
    """One discount curve plus a forecast curve per index tenor.

    This is where multi-curve pricing enters, and it enters in exactly one place:
    a pricer asks for ``discount`` to discount and ``forecast_for(tenor)`` to
    project. No other module needs to know that two curves exist.
    """

    discount: DiscountCurve
    forecast: Mapping[str, DiscountCurve]

    @classmethod
    def single(cls, curve: DiscountCurve) -> CurveSet:
        """The pre-2008 single-curve world: forecast and discount coincide."""
        return cls(discount=curve, forecast=defaultdict(lambda: curve))

    def forecast_for(self, tenor: str) -> DiscountCurve:
        try:
            return self.forecast[tenor]
        except KeyError as exc:
            raise KeyError(
                f"No forecast curve for tenor {tenor!r}; available: {sorted(self.forecast)}"
            ) from exc
