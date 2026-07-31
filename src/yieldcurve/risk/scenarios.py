"""Curve scenarios and the single shift primitive.

Everything in ``yieldcurve.risk`` that needs a moved curve calls
``shift_curve``. Effective duration, key-rate duration, PCA duration and the
regulatory scenario P&L are then the same computation with a different
``Scenario`` — which is what makes ``sum(krd) == effective_duration`` an
identity rather than a coincidence.
"""

from __future__ import annotations

import math
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from yieldcurve.curves.protocol import CurveSet, DiscountCurve


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
    shifted = {tenor: shift_curve(curve, scenario) for tenor, curve in curves.forecast.items()}
    # CurveSet.single answers every tenor from one curve via a default factory,
    # and enumerates as empty until a tenor is asked for. Shift the factory too,
    # or the shocked set would have no forecast curve at all.
    factory = getattr(curves.forecast, "default_factory", None)
    forecast: Mapping[str, DiscountCurve] = (
        shifted
        if factory is None
        else defaultdict(lambda: shift_curve(factory(), scenario), shifted)
    )
    return CurveSet(discount=shift_curve(curves.discount, scenario), forecast=forecast)


def parallel(size: float) -> Scenario:
    """A flat shift of ``size`` decimals at every tenor."""
    sign = "+" if size >= 0 else "-"
    return Scenario(name=f"parallel {sign}{abs(size) * 1e4:.0f}bp", shift=lambda _t: size)


_BP = 1e-4


def _find_config(start: Path | None = None) -> Path:
    d = (start or Path(__file__).resolve().parent).resolve()
    for _ in range(10):
        candidate = d / "scenarios.toml"
        if candidate.exists():
            return candidate
        if (d / "pyproject.toml").exists():
            raise ScenarioConfigError("Reached project root without finding scenarios.toml")
        d = d.parent
    raise ScenarioConfigError("Exhausted 10 levels without finding scenarios.toml")


_CONFIG_PATH = _find_config()


class ScenarioConfigError(ValueError):
    """scenarios.toml is missing or does not describe the requested shock."""


def load_scenarios(path: Path | None = None) -> dict[str, object]:
    target = path or _CONFIG_PATH
    if not target.exists():
        raise ScenarioConfigError(f"No scenario configuration at {target}")
    with target.open("rb") as handle:
        return tomllib.load(handle)


def bcbs_scenarios(currency: str, *, path: Path | None = None) -> tuple[Scenario, ...]:
    config = load_scenarios(path)
    currencies: dict[str, dict[str, object]] = config.get("currency", {})  # type: ignore[assignment]
    if currency not in currencies:
        raise ScenarioConfigError(
            f"{currency} is not in scenarios.toml; add its row from BCBS d368 "
            "Annex 2 rather than reusing another currency's sizes"
        )
    block = currencies[currency]
    shape: dict[str, object] = config["shape"]  # type: ignore[assignment]
    decay = float(shape["short_decay_years"])  # type: ignore[arg-type]
    citation = str(block["citation"])

    parallel_size = float(block["parallel_bp"]) * _BP  # type: ignore[arg-type]
    short_size = float(block["short_bp"]) * _BP  # type: ignore[arg-type]
    long_size = float(block["long_bp"]) * _BP  # type: ignore[arg-type]

    def short_factor(t: float) -> float:
        return math.exp(-t / decay)

    def long_factor(t: float) -> float:
        return 1.0 - math.exp(-t / decay)

    steep: dict[str, object] = shape["steepener"]  # type: ignore[assignment]
    flat: dict[str, object] = shape["flattener"]  # type: ignore[assignment]
    steep_short, steep_long = float(steep["short_weight"]), float(steep["long_weight"])  # type: ignore[arg-type]
    flat_short, flat_long = float(flat["short_weight"]), float(flat["long_weight"])  # type: ignore[arg-type]

    return (
        Scenario("parallel_up", lambda t: parallel_size, citation),
        Scenario("parallel_down", lambda t: -parallel_size, citation),
        Scenario("short_up", lambda t: short_size * short_factor(t), citation),
        Scenario("short_down", lambda t: -short_size * short_factor(t), citation),
        Scenario(
            "steepener",
            lambda t: (
                steep_short * short_size * short_factor(t) + steep_long * long_size * long_factor(t)
            ),
            str(steep["citation"]),
        ),
        Scenario(
            "flattener",
            lambda t: (
                flat_short * short_size * short_factor(t) + flat_long * long_size * long_factor(t)
            ),
            str(flat["citation"]),
        ),
    )
