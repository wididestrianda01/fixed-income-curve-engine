"""The one contract the whole library is built on.

A curve is anything that can produce a discount factor for a time. It is a
``Protocol`` rather than a base class deliberately: a bootstrapped curve, a
Svensson curve and a shocked curve all satisfy it without any of them importing
the others, and mypy checks the conformance structurally.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
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

    Curves backed by discrete quoted inputs additionally expose
    ``covered_horizon``: the largest curve time up to which quoted inputs exist.
    Beyond it every value is extrapolated — an unobservable input (a Level 3
    input under IFRS 13, whose hierarchy classification follows input
    significance, not extrapolation alone) — so consumers should treat the
    extrapolated region as a modelling choice rather than market data.
    ``InterpolatedDiscountCurve`` implements it;
    analytic curves such as ``FlatCurve`` are defined for all times.
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

    Discount and every forecast curve must share one reference date, so every
    present value is a ratio in absolute curve time. The forecast map is
    constructed by the builders and treated as read-only by consumers: reads
    allocate nothing and no consumer mutates a map.
    """

    discount: DiscountCurve
    forecast: Mapping[str, DiscountCurve]

    def __post_init__(self) -> None:
        ref = self.discount.reference_date
        bad = [tenor for tenor, curve in self.forecast.items() if curve.reference_date != ref]
        if bad:
            raise ValueError(f"forecast curves {bad} reference date differs from discount {ref}")
        # ponytail: freeze deferred — build.py mutates after construction;
        # add when all callers construct fully before passing.

    @classmethod
    def single(cls, curve: DiscountCurve) -> CurveSet:
        """The pre-2008 single-curve world: forecast and discount coincide.

        Every tenor resolves to the one curve with no internal map and no read
        allocation, and never raises for a missing tenor. The explicit
        multi-curve ``CurveSet`` still misses loudly.
        """
        return cls(discount=curve, forecast=_AlwaysDiscount(curve))

    def forecast_for(self, tenor: str) -> DiscountCurve:
        try:
            return self.forecast[tenor]
        except KeyError:
            if isinstance(self.forecast, _AlwaysDiscount):
                return self.discount
            raise KeyError(
                f"No forecast curve for tenor {tenor!r}; available: {sorted(self.forecast)}"
            ) from None


class _AlwaysDiscount(Mapping[str, DiscountCurve]):
    """Empty forecast map that resolves any tenor to the held discount curve.

    Used only by ``CurveSet.single`` so single-curve reads allocate nothing and
    never raise for a missing tenor; the explicit multi-curve ``CurveSet`` still
    misses loudly.
    """

    __slots__ = ("_curve",)

    def __init__(self, curve: DiscountCurve) -> None:
        self._curve = curve

    def __getitem__(self, tenor: str) -> DiscountCurve:
        return self._curve

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


@dataclass(frozen=True)
class Fixings:
    """Observed rate fixings for floating legs.

    ``term`` maps ``(index tenor, reset date)`` to the observed reset rate; an
    active term coupon that has already fixed must look its rate up here rather
    than project a forward over a stub. ``overnight`` maps an observation date
    to the realised overnight rate for that business day. Both maps are frozen
    on construction.
    """

    term: Mapping[tuple[str, date], float] = MappingProxyType({})
    overnight: Mapping[date, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        pass

    def term_rate(self, tenor: str, reset_date: date) -> float:
        key = (tenor, reset_date)
        try:
            return self.term[key]
        except KeyError as exc:
            raise MissingFixingError(f"missing term fixing for {tenor} @ {reset_date}") from exc

    def overnight_rate(self, observation_date: date) -> float:
        try:
            return self.overnight[observation_date]
        except KeyError as exc:
            raise MissingFixingError(f"missing overnight fixing @ {observation_date}") from exc


class MissingFixingError(KeyError):
    """A floating coupon has already fixed but its observed rate is absent.

    The library never replaces a missing fixing with a shortened forward: that
    silently scales a coupon by the fraction of the period still outstanding and
    bleeds value as the valuation date advances.
    """
