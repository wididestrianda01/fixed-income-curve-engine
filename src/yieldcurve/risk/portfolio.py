"""Aggregation of instrument-level analytics into portfolio-level ones.

Everything here is a thin sum over :mod:`yieldcurve.curves.pricing` and
:mod:`yieldcurve.risk.scenarios`. No new financial mathematics lives in this module; if a
number looks wrong, the fault is upstream in the pricer or the scenario definition.

Scope: the supported portfolio is explicitly single-currency. A portfolio file
must declare its ``currency``; positions carry no currency of their own and no
FX mapping exists, so every notional is in the declared currency.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from yieldcurve.calendars import SwedenCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import (
    FixedCouponBond,
    Instrument,
    VanillaSwap,
    tenor_to_frequency,
)
from yieldcurve.risk.keyrate import SEK_KEY_RATES, hat
from yieldcurve.risk.scenarios import Scenario, shift_curveset
from yieldcurve.risk.sensitivities import instrument_scale

__all__ = [
    "Portfolio",
    "PortfolioError",
    "Position",
    "bucket_exposure",
    "delta_eve",
    "eve_ladder",
    "historical_pnl",
    "present_value",
    "var_es",
]

_BASIS_POINT = 1e-4

_MIN_TAIL_OBSERVATIONS = 10
"""Below this many observations beyond the quantile, expected shortfall is noise."""

_DAY_COUNTS = {
    "30/360": DayCount.THIRTY_360_BOND,
    "ACT/360": DayCount.ACT_360,
    "ACT/365F": DayCount.ACT_365F,
    "ACT/ACT": DayCount.ACT_ACT_ICMA,
}


class PortfolioError(ValueError):
    """Raised when a portfolio definition or a risk request is malformed."""


@dataclass(frozen=True)
class Position:
    """One line of a book: an instrument and how much of it is held.

    ``notional`` is a currency amount, positive for long and negative for
    short. The position value divides by the instrument's own scale — ``face``
    for bonds and bills, ``notional`` for swaps (see
    :func:`yieldcurve.risk.sensitivities.instrument_scale`) — so the same
    contract serves both families.
    """

    label: str
    instrument: Instrument
    notional: float


@dataclass(frozen=True)
class Portfolio:
    """An ordered collection of positions valued against one curve set."""

    positions: tuple[Position, ...]

    @classmethod
    def from_toml(cls, path: Path) -> Portfolio:
        """Read a committed portfolio definition.

        Every field is validated for its exact TOML type, finiteness, date
        order, frequency, tenor grid, enum membership, and the single-currency
        declaration before any instrument is constructed (error policy: reject
        invalid data before arithmetic).

        Raises:
            PortfolioError: with position context, on any malformed input.
        """
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        entries = document.get("position", [])
        if not isinstance(entries, list):
            raise PortfolioError(f"{path}: 'position' must be an array of tables")
        if not entries:
            raise PortfolioError(f"{path} declares no [[position]] entries")
        if (
            "currency" not in document
            or not isinstance(document["currency"], str)
            or not document["currency"]
        ):
            raise PortfolioError(
                f"{path} must declare a single non-empty 'currency' string "
                "(the supported portfolio is single-currency; no FX mapping exists)"
            )
        positions = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise PortfolioError(f"{path}: each [[position]] entry must be a table")
            positions.append(_position_from_entry(entry))
        return cls(positions=tuple(positions))


def _day_count(name: str) -> DayCount:
    try:
        return _DAY_COUNTS[name]
    except KeyError:
        raise PortfolioError(f"unknown day_count {name!r}") from None


def _required(entry: dict[str, Any], field: str) -> Any:
    if field not in entry:
        raise PortfolioError(f"position {entry.get('label', '<unlabelled>')!r} is missing {field}")
    return entry[field]


def _require_string(entry: dict[str, Any], field: str, label: str) -> str:
    value = _required(entry, field)
    if not isinstance(value, str):
        raise PortfolioError(
            f"position {label!r}: {field} must be a string, got {type(value).__name__}"
        )
    return value


def _require_number(entry: dict[str, Any], field: str, label: str) -> float:
    value = _required(entry, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioError(
            f"position {label!r}: {field} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PortfolioError(f"position {label!r}: {field} must be finite, got {value!r}")
    return number


def _require_int(entry: dict[str, Any], field: str, label: str) -> int:
    value = _required(entry, field)
    if isinstance(value, bool) or type(value) is not int:
        raise PortfolioError(
            f"position {label!r}: {field} must be an integer, got {type(value).__name__}"
        )
    return value


def _require_date(entry: dict[str, Any], field: str, label: str) -> date:
    value = _required(entry, field)
    if type(value) is not date:
        raise PortfolioError(
            f"position {label!r}: {field} must be a TOML date, got {type(value).__name__}"
        )
    return value


def _require_bool(entry: dict[str, Any], field: str, label: str) -> bool:
    value = _required(entry, field)
    if type(value) is not bool:
        raise PortfolioError(
            f"position {label!r}: {field} must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_frequency(entry: dict[str, Any], field: str, label: str) -> int:
    frequency = _require_int(entry, field, label)
    if frequency <= 0 or 12 % frequency != 0:
        raise PortfolioError(
            f"position {label!r}: {field} must divide 12 evenly "
            f"(one of 1, 2, 3, 4, 6, 12), got {frequency}"
        )
    return frequency


def _position_from_entry(entry: dict[str, Any]) -> Position:
    kind = _require_string(entry, "kind", "<unlabelled>")
    label = _require_string(entry, "label", "<unlabelled>")
    notional = _require_number(entry, "notional", label)
    if notional == 0.0:
        raise PortfolioError(f"position {label!r}: notional must be non-zero")
    instrument: Instrument
    if kind == "bond":
        issue = _require_date(entry, "issue", label)
        maturity = _require_date(entry, "maturity", label)
        if maturity <= issue:
            raise PortfolioError(
                f"position {label!r}: maturity {maturity} must fall after issue {issue}"
            )
        instrument = FixedCouponBond(
            issue=issue,
            maturity=maturity,
            coupon=_require_number(entry, "coupon", label),
            frequency=_require_frequency(entry, "frequency", label),
            day_count=_day_count(_require_string(entry, "day_count", label)),
            calendar=SwedenCalendar(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
    elif kind == "swap":
        start = _require_date(entry, "start", label)
        maturity = _require_date(entry, "maturity", label)
        if maturity <= start:
            raise PortfolioError(
                f"position {label!r}: maturity {maturity} must fall after start {start}"
            )
        float_tenor = _require_string(entry, "float_tenor", label)
        try:
            tenor_to_frequency(float_tenor)
        except ValueError as exc:
            raise PortfolioError(f"position {label!r}: float_tenor: {exc}") from None
        instrument = VanillaSwap(
            start=start,
            maturity=maturity,
            fixed_rate=_require_number(entry, "fixed_rate", label),
            fixed_frequency=_require_frequency(entry, "fixed_frequency", label),
            fixed_day_count=_day_count(_require_string(entry, "fixed_day_count", label)),
            float_tenor=float_tenor,
            float_day_count=_day_count(_require_string(entry, "float_day_count", label)),
            calendar=SwedenCalendar(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
            notional=1.0,
            pay_fixed=_require_bool(entry, "pay_fixed", label),
        )
    else:
        raise PortfolioError(f"unknown instrument kind {kind!r}")
    return Position(label=label, instrument=instrument, notional=notional)


def _position_value(position: Position, curves: CurveSet, asof: date) -> float:
    unit = price(position.instrument, curves, asof).dirty
    return position.notional / instrument_scale(position.instrument) * unit


def present_value(portfolio: Portfolio, curves: CurveSet, asof: date) -> float:
    """Total dirty present value of the book, in the currency of its notionals."""
    return sum(_position_value(p, curves, asof) for p in portfolio.positions)


def delta_eve(portfolio: Portfolio, curves: CurveSet, asof: date, scenario: Scenario) -> float:
    """Change in economic value of equity under ``scenario``.

    Signed like :func:`yieldcurve.risk.sensitivities.dv01`: negative when the shock destroys
    value.
    """
    base = present_value(portfolio, curves, asof)
    shocked = present_value(portfolio, shift_curveset(curves, scenario), asof)
    return shocked - base


def eve_ladder(
    portfolio: Portfolio,
    curves: CurveSet,
    asof: date,
    scenarios: Sequence[Scenario],
) -> dict[str, float]:
    """ΔEVE for each scenario, keyed by name, in the order given."""
    return {s.name: delta_eve(portfolio, curves, asof, s) for s in scenarios}


def bucket_exposure(
    portfolio: Portfolio,
    curves: CurveSet,
    asof: date,
    keys: Sequence[float],
    *,
    bump: float = _BASIS_POINT,
) -> dict[float, float]:
    """Monetary change in book value per 1bp rise in each key rate, by central
    difference. Units: currency per 1bp, in the currency of the portfolio's
    notionals.

    This is the monetary BPV ladder: unlike
    :func:`yieldcurve.risk.keyrate.krd` it never divides by the base present
    value, so it stays defined for a par swap that prices at zero.

    Because the Ho (1992) hats partition unity, the values sum to the
    portfolio's parallel monetary sensitivity — which is what makes the ladder
    checkable.

    Raises:
        PortfolioError: if ``bump`` is not a positive finite rate, or ``keys``
            is not a strictly ascending finite grid of at least two tenors.
    """
    if not math.isfinite(bump) or bump <= 0.0:
        raise PortfolioError(f"bucket_exposure bump must be a positive finite rate, got {bump}")
    base_up = {}
    for index, key in enumerate(keys):
        up = present_value(portfolio, shift_curveset(curves, hat(keys, index, bump)), asof)
        down = present_value(portfolio, shift_curveset(curves, hat(keys, index, -bump)), asof)
        base_up[float(key)] = (up - down) / (2.0 * bump)
    return base_up


def historical_pnl(
    portfolio: Portfolio,
    curves: CurveSet,
    asof: date,
    changes: npt.NDArray[np.float64],
    tenors: Sequence[float],
    keys: Sequence[float] = SEK_KEY_RATES,
) -> npt.NDArray[np.float64]:
    """Linearized delta P&L proxy: one number per row of observed daily rate
    changes, by bucket exposure.

    Each row of ``changes`` holds absolute zero-rate moves at ``tenors``. Those
    are linearly interpolated onto ``keys`` (flat beyond the ends) and
    contracted against the portfolio's bucket exposures. This is a first-order
    proxy — bucket exposures are recomputed at the valuation date and moved
    linearly with the rates — so it is *not* full revaluation and carries no
    FRTB or regulatory VaR implication.

    Raises:
        PortfolioError: if ``changes`` is not 2-D with one column per tenor,
            contains non-finite values, or has no rows; if ``tenors`` is empty,
            non-finite, or not strictly increasing.
    """
    if changes.ndim != 2 or changes.shape[1] != len(tenors):
        raise PortfolioError(
            f"changes must have {len(tenors)} columns to match tenors, got shape {changes.shape}"
        )
    if changes.shape[0] == 0:
        raise PortfolioError("changes must contain at least one observation")
    if not np.isfinite(changes).all():
        raise PortfolioError("changes must be finite")
    grid = np.asarray(tenors, dtype=np.float64)
    if grid.size == 0 or not np.isfinite(grid).all():
        raise PortfolioError(f"tenors must be finite and non-empty, got {tenors}")
    if np.any(np.diff(grid) <= 0.0):
        raise PortfolioError(f"tenors must be strictly increasing, got {tenors}")
    exposure = bucket_exposure(portfolio, curves, asof, keys)
    targets = np.asarray(list(exposure), dtype=np.float64)
    projected = np.column_stack(
        [np.interp(targets, grid, row) for row in np.asarray(changes, dtype=np.float64)]
    ).T
    return projected @ np.asarray(list(exposure.values()), dtype=np.float64)


def var_es(pnl: npt.NDArray[np.float64], *, confidence: float = 0.99) -> tuple[float, float]:
    """Historical value-at-risk and expected shortfall of the linearized delta
    P&L, as positive loss magnitudes.

    A first-order (bucket-exposure) proxy on historical rate changes: no full
    revaluation, no FRTB or other regulatory measure.

    Raises:
        PortfolioError: if ``confidence`` is outside ``(0, 1)``, if ``pnl`` is
            empty or non-finite, or if fewer than ten observations fall beyond
            the quantile — an expected shortfall averaged over a handful of
            points is not a measurement.
    """
    if not 0.0 < confidence < 1.0:
        raise PortfolioError(f"confidence must lie in (0, 1), got {confidence}")
    pnl_arr = np.asarray(pnl, dtype=np.float64).ravel()
    if pnl_arr.size == 0:
        raise PortfolioError("pnl must contain at least one observation")
    if not np.isfinite(pnl_arr).all():
        raise PortfolioError("pnl must be finite")
    losses = -pnl_arr
    threshold = float(np.quantile(losses, confidence))
    tail = losses[losses >= threshold]
    if tail.size < _MIN_TAIL_OBSERVATIONS:
        raise PortfolioError(
            f"only {tail.size} observations in the {confidence:.0%} tail; "
            f"at least {_MIN_TAIL_OBSERVATIONS} are needed for an expected shortfall"
        )
    return threshold, float(tail.mean())
