"""EU 2024/856 supervisory shock scenarios and the single shift primitive.

This module implements the six supervisory shock scenarios of Commission
Delegated Regulation (EU) 2024/856 (the RTS specifying the supervisory shock
scenarios under Article 98(5) of Directive 2013/36/EU) — parallel up/down,
short-rate up/down, steepener and flattener — with the USD and SEK shock
parameters of Annex Part A, the Article 2 parameterisations and the Article
3(7) post-shock, maturity-dependent rate floor, for educational analysis.

It does not claim to implement an institution-wide supervisory outlier test,
IRRBB compliance, capital, NII, behavioural modelling, currency aggregation
or regulatory reporting.

Everything in ``yieldcurve.risk`` that needs a moved curve calls
``shift_curve``. Effective duration, key-rate duration, PCA duration and the
EU scenario ΔEVE are then the same computation with a different
``Scenario`` — which is what makes ``sum(krd)`` approximate
``effective_duration`` within the numerical tolerance documented in
``yieldcurve.risk.keyrate`` (the central-difference O(bump²) truncation
error), not an exact identity.
"""

from __future__ import annotations

import importlib.resources
import math
import tomllib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NoReturn

from yieldcurve.curves.protocol import CurveSet, DiscountCurve, _AlwaysDiscount

_BP = 1e-4

# Article 3(7): the post-shock floor starts at -150 bp for immediate maturity
# and rises 3 bp per year, reaching 0 % for maturities of 50 years and more.
_FLOOR_START = -0.015
_FLOOR_SLOPE = 0.0003


@dataclass(frozen=True)
class Scenario:
    """A shift of the continuously compounded zero curve.

    ``shift(t)`` is in decimals at curve time ``t`` in ACT/365F years.
    ``citation`` carries the source for regulatory scenarios and is empty for
    ad-hoc ones; ``scenarios.toml`` requires it to be non-empty. ``floor``, when
    present, is the maturity-dependent post-shock rate floor in decimals
    (Commission Delegated Regulation (EU) 2024/856, Article 3(7)): the shocked
    zero rate never falls below ``min(floor(t), observed rate at t)``.
    """

    name: str
    shift: Callable[[float], float]
    citation: str = ""
    floor: Callable[[float], float] | None = None


@dataclass(frozen=True)
class _ShiftedCurve:
    """A DiscountCurve view over a base curve plus a zero-rate shift.

    Structural typing means this satisfies the DiscountCurve Protocol without
    inheriting from anything, so it composes with itself and with anything else
    the protocol accepts. Nothing is precomputed: shifting is O(1) and lazy,
    which matters when a PCA run shifts the same curve a few thousand times.

    When the scenario carries a post-shock floor, ``zero`` clamps the shocked
    rate to the effective floor (Article 3(7): the floor, or the lower observed
    rate where observed rates are already below it) and ``df`` stays consistent
    with that floored zero — keeping the exact ``df * exp(-shift*t)`` closed
    form whenever the floor does not bind, so a floored scenario that never
    clamps prices identically to an unfloored one. Without a floor the
    discount factor always uses that closed form.
    """

    base: DiscountCurve
    scenario: Scenario

    @property
    def reference_date(self) -> date:
        return self.base.reference_date

    def _zero_with_shift(self, t: float) -> tuple[float, float, bool]:
        """(floored zero rate, shift actually applied, floor bound) at ``t``.

        ``zero`` and ``df`` share this single computation so the closed-form
        decision in ``df`` is made by the same arithmetic that produced the
        floored rate, never by a separately recomputed sum. The applied shift
        is ``scenario.shift(t)`` exactly when the floor did not bind and
        ``rate - base.zero(t)`` when it did; a future reordering of the
        arithmetic inside ``zero`` therefore cannot silently flip the path.
        """
        base_zero = self.base.zero(t)
        applied = self.scenario.shift(t)
        shocked = base_zero + applied
        floor = self.scenario.floor
        if floor is None:
            return shocked, applied, False
        effective = min(floor(t), base_zero)
        floored = max(shocked, effective)
        if floored == shocked:
            # The floor did not bind: the floored rate is the shocked rate, so
            # ``df`` can keep the exact ``df * exp(-shift*t)`` closed form,
            # bit-identical to the unfloored path.
            return floored, applied, False
        return floored, floored - base_zero, True

    def zero(self, t: float) -> float:
        return self._zero_with_shift(t)[0]

    def df(self, t: float) -> float:
        if self.scenario.floor is None:
            return self.base.df(t) * math.exp(-self.scenario.shift(t) * t)
        rate, applied, bound = self._zero_with_shift(t)
        if not bound:
            return self.base.df(t) * math.exp(-applied * t)
        return math.exp(-rate * t)

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"fwd requires t2 > t1, got t1={t1}, t2={t2}")
        return -(math.log(self.df(t2)) - math.log(self.df(t1))) / (t2 - t1)


def shift_curve(curve: DiscountCurve, scenario: Scenario) -> DiscountCurve:
    """Apply a scenario to one curve. Returns a new curve; nothing mutates."""
    return _ShiftedCurve(base=curve, scenario=scenario)


def shift_curveset(curves: CurveSet, scenario: Scenario) -> CurveSet:
    """Apply one scenario to the discount curve and every forecast curve.

    The scenario is applied to the discount curve and each forecast curve, so
    the quoted basis spread between them is held fixed under the shock. That is
    the modelling choice used by this educational analysis; it is a documented
    convention, not a regulatory instruction.
    """
    if isinstance(curves.forecast, _AlwaysDiscount):
        return CurveSet.single(shift_curve(curves.discount, scenario))
    shifted = {tenor: shift_curve(curve, scenario) for tenor, curve in curves.forecast.items()}
    factory = getattr(curves.forecast, "default_factory", None)
    forecast = (
        shifted
        if factory is None
        else defaultdict(lambda: shift_curve(factory(), scenario), shifted)
    )
    return CurveSet(discount=shift_curve(curves.discount, scenario), forecast=forecast)


def parallel(size: float) -> Scenario:
    """A flat shift of ``size`` decimals at every tenor (unfloored)."""
    sign = "+" if size >= 0 else "-"
    return Scenario(name=f"parallel {sign}{abs(size) * 1e4:.0f}bp", shift=lambda _t: size)


def post_shock_floor(maturity: float) -> float:
    """The Article 3(7) maturity-dependent post-shock floor, in decimals.

    ``floor(t) = min(0, -150 bp + 3 bp per year * t)``: -150 bp at immediate
    maturity, rising to 0 % at 50 years and more.

    Source: Commission Delegated Regulation (EU) 2024/856, Article 3(7).
    """
    return min(0.0, _FLOOR_START + _FLOOR_SLOPE * maturity)


class ScenarioConfigError(ValueError):
    """scenarios.toml is missing, malformed, or does not describe the requested shock."""


@dataclass(frozen=True)
class _RotationShapes:
    short_weight: float
    long_weight: float
    citation: str


@dataclass(frozen=True)
class _CurrencyShocks:
    parallel_bp: int
    short_bp: int
    long_bp: int
    citation: str


@dataclass(frozen=True)
class _ScenarioConfig:
    short_decay_years: float
    shape_citation: str
    steepener: _RotationShapes
    flattener: _RotationShapes
    currencies: dict[str, _CurrencyShocks]


def _fail(source: str, message: str) -> NoReturn:
    raise ScenarioConfigError(f"{source}: {message}")


def _check_unknown(source: str, context: str, keys: set[str], allowed: set[str]) -> None:
    unknown = keys - allowed
    if unknown:
        _fail(source, f"{context}: unknown key(s) {sorted(unknown)}")


def _number(source: str, context: str, value: object, *, positive: bool = False) -> float:
    """A number field: ``int`` and ``float`` are accepted, with an ``int``
    widened to ``float``; ``bool``, ``str`` and anything else are rejected.
    Non-finite and (optionally) non-positive values are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(source, f"{context} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        _fail(source, f"{context} must be finite, got {value!r}")
    if positive and number <= 0.0:
        _fail(source, f"{context} must be positive, got {value!r}")
    return number


def _integer(source: str, context: str, value: object, *, positive: bool = True) -> int:
    """An exact integer field: ``bool``, floats, strings and other types are
    rejected rather than coerced (fractional integer fields are invalid)."""
    if isinstance(value, bool):
        _fail(source, f"{context} must be an integer, got bool")
    if isinstance(value, float) and not math.isfinite(value):
        _fail(source, f"{context} must be finite, got {value!r}")
    if type(value) is not int:
        _fail(source, f"{context} must be an integer, got {type(value).__name__}")
    if positive and value <= 0:
        _fail(source, f"{context} must be positive, got {value!r}")
    return value


def _text(source: str, context: str, value: object) -> str:
    if type(value) is not str or not value.strip():
        _fail(source, f"{context} must be a non-empty string, got {value!r}")
    return value


def _parse_config(document: dict[str, object], source: str) -> _ScenarioConfig:
    _check_unknown(source, "top level", set(document), {"shape", "currency"})
    shape_raw = document.get("shape")
    if not isinstance(shape_raw, dict):
        _fail(source, "missing [shape] table")
    _check_unknown(
        source,
        "shape",
        set(shape_raw),
        {"short_decay_years", "citation", "steepener", "flattener"},
    )
    if "short_decay_years" not in shape_raw:
        _fail(source, "shape: missing short_decay_years")
    decay = _number(
        source, "shape.short_decay_years", shape_raw["short_decay_years"], positive=True
    )
    shape_citation = _text(source, "shape.citation", shape_raw.get("citation"))

    rotations: dict[str, _RotationShapes] = {}
    for name in ("steepener", "flattener"):
        raw = shape_raw.get(name)
        if not isinstance(raw, dict):
            _fail(source, f"shape: missing [{name}] table")
        _check_unknown(
            source, f"shape.{name}", set(raw), {"short_weight", "long_weight", "citation"}
        )
        if "short_weight" not in raw or "long_weight" not in raw:
            _fail(source, f"shape.{name}: missing short_weight or long_weight")
        rotations[name] = _RotationShapes(
            short_weight=_number(source, f"shape.{name}.short_weight", raw["short_weight"]),
            long_weight=_number(source, f"shape.{name}.long_weight", raw["long_weight"]),
            citation=_text(source, f"shape.{name}.citation", raw.get("citation")),
        )

    currency_raw = document.get("currency")
    if not isinstance(currency_raw, dict) or not currency_raw:
        _fail(source, "missing or empty [currency] table")
    currencies: dict[str, _CurrencyShocks] = {}
    for code, block in currency_raw.items():
        if not code:
            _fail(source, "currency code must be a non-empty string")
        if not isinstance(block, dict):
            _fail(source, f"currency {code!r} must be a table")
        _check_unknown(
            source,
            f"currency {code!r}",
            set(block),
            {"parallel_bp", "short_bp", "long_bp", "citation"},
        )
        for field in ("parallel_bp", "short_bp", "long_bp"):
            if field not in block:
                _fail(source, f"currency {code!r}: missing {field}")
        currencies[code] = _CurrencyShocks(
            parallel_bp=_integer(source, f"currency {code!r}.parallel_bp", block["parallel_bp"]),
            short_bp=_integer(source, f"currency {code!r}.short_bp", block["short_bp"]),
            long_bp=_integer(source, f"currency {code!r}.long_bp", block["long_bp"]),
            citation=_text(source, f"currency {code!r}.citation", block.get("citation")),
        )
    return _ScenarioConfig(
        short_decay_years=decay,
        shape_citation=shape_citation,
        steepener=rotations["steepener"],
        flattener=rotations["flattener"],
        currencies=currencies,
    )


def _load_validated(path: Path | None) -> tuple[_ScenarioConfig, str]:
    """Read and strictly validate the scenario configuration from the packaged
    resource (``importlib.resources``) or from an explicit caller path."""
    if path is None:
        source = "packaged scenarios.toml (yieldcurve.risk)"
        resource = importlib.resources.files("yieldcurve.risk").joinpath("scenarios.toml")
        try:
            handle = resource.open("rb")
        except FileNotFoundError:
            _fail(source, "packaged scenario configuration is missing")
        with handle:
            try:
                document = tomllib.load(handle)
            except tomllib.TOMLDecodeError as exc:
                _fail(source, f"invalid TOML: {exc}")
        return _parse_config(document, source), source
    if path.is_dir():
        _fail(str(path), "expected a TOML file, got a directory")
    if not path.exists():
        _fail(str(path), "no such file")
    source = f"scenarios.toml at {path}"
    with path.open("rb") as handle:
        try:
            document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            _fail(source, f"invalid TOML: {exc}")
    return _parse_config(document, source), source


def load_scenarios(path: Path | None = None) -> dict[str, object]:
    """The validated scenario configuration as a plain mapping.

    ``path`` is an explicit caller-supplied configuration; when omitted the
    packaged ``scenarios.toml`` resource is used, so an installed wheel works
    with no checkout. Raises :class:`ScenarioConfigError` with file and key
    context on any malformed input.
    """
    config, _ = _load_validated(path)
    return {
        "shape": {
            "short_decay_years": config.short_decay_years,
            "citation": config.shape_citation,
            "steepener": {
                "short_weight": config.steepener.short_weight,
                "long_weight": config.steepener.long_weight,
                "citation": config.steepener.citation,
            },
            "flattener": {
                "short_weight": config.flattener.short_weight,
                "long_weight": config.flattener.long_weight,
                "citation": config.flattener.citation,
            },
        },
        "currency": {
            code: {
                "parallel_bp": shocks.parallel_bp,
                "short_bp": shocks.short_bp,
                "long_bp": shocks.long_bp,
                "citation": shocks.citation,
            }
            for code, shocks in config.currencies.items()
        },
    }


def eu_scenarios(currency: str, *, path: Path | None = None) -> tuple[Scenario, ...]:
    """The six EU 2024/856 supervisory shock scenarios for ``currency``.

    USD and SEK are shipped in the packaged configuration with their Annex
    Part A rows (200/300/150 bp each). Every returned scenario carries the
    Article 3(7) post-shock floor. ``path`` overrides the packaged resource
    with a caller-supplied configuration file.

    Raises:
        ScenarioConfigError: if ``currency`` is empty or has no row, or the
            configuration is missing or malformed.
    """
    config, source = _load_validated(path)
    if not currency:
        _fail(source, f"currency must be a non-empty string, got {currency!r}")
    block = config.currencies.get(currency)
    if block is None:
        _fail(
            source,
            f"{currency!r} has no shock row; the packaged configuration ships USD and SEK "
            "from Annex Part A of Commission Delegated Regulation (EU) 2024/856. Add the "
            "currency's Part A row rather than reusing another currency's sizes.",
        )

    parallel_size = block.parallel_bp * _BP
    short_size = block.short_bp * _BP
    long_size = block.long_bp * _BP

    def short_factor(t: float) -> float:
        return math.exp(-t / config.short_decay_years)

    def long_factor(t: float) -> float:
        return 1.0 - math.exp(-t / config.short_decay_years)

    steep = config.steepener
    flat = config.flattener

    return (
        Scenario(
            "parallel_up",
            lambda _t: parallel_size,
            citation=block.citation,
            floor=post_shock_floor,
        ),
        Scenario(
            "parallel_down",
            lambda _t: -parallel_size,
            citation=block.citation,
            floor=post_shock_floor,
        ),
        Scenario(
            "short_up",
            lambda t: short_size * short_factor(t),
            citation=block.citation,
            floor=post_shock_floor,
        ),
        Scenario(
            "short_down",
            lambda t: -short_size * short_factor(t),
            citation=block.citation,
            floor=post_shock_floor,
        ),
        Scenario(
            "steepener",
            lambda t: (
                steep.short_weight * short_size * short_factor(t)
                + steep.long_weight * long_size * long_factor(t)
            ),
            citation=steep.citation,
            floor=post_shock_floor,
        ),
        Scenario(
            "flattener",
            lambda t: (
                flat.short_weight * short_size * short_factor(t)
                + flat.long_weight * long_size * long_factor(t)
            ),
            citation=flat.citation,
            floor=post_shock_floor,
        ),
    )
