"""Aggregation of instrument-level analytics into portfolio-level ones.

Everything here is a thin sum over :mod:`yieldcurve.curves.pricing` and
:mod:`yieldcurve.risk.scenarios`. No new financial mathematics lives in this module; if a
number looks wrong, the fault is upstream in the pricer or the scenario definition.
"""

from __future__ import annotations

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
from yieldcurve.instruments import FixedCouponBond, Instrument, VanillaSwap
from yieldcurve.risk.keyrate import SEK_KEY_RATES, hat
from yieldcurve.risk.scenarios import Scenario, shift_curveset

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

    ``notional`` is a currency amount, positive for long and negative for short. It is not a
    multiplier on the instrument's own notional — see :func:`_position_value`.
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

        Raises:
            PortfolioError: if a position has an unknown ``kind`` or is missing a field.
        """
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        entries = document.get("position", [])
        if not entries:
            raise PortfolioError(f"{path} declares no [[position]] entries")
        return cls(positions=tuple(_position_from_entry(entry) for entry in entries))


def _day_count(name: str) -> DayCount:
    try:
        return _DAY_COUNTS[name]
    except KeyError:
        raise PortfolioError(f"unknown day_count {name!r}") from None


def _required(entry: dict[str, Any], field: str) -> Any:
    if field not in entry:
        raise PortfolioError(f"position {entry.get('label', '<unlabelled>')!r} is missing {field}")
    return entry[field]


def _position_from_entry(entry: dict[str, Any]) -> Position:
    kind = _required(entry, "kind")
    label = str(_required(entry, "label"))
    notional = float(_required(entry, "notional"))
    instrument: Instrument
    if kind == "bond":
        instrument = FixedCouponBond(
            issue=_required(entry, "issue"),
            maturity=_required(entry, "maturity"),
            coupon=float(_required(entry, "coupon")),
            frequency=int(_required(entry, "frequency")),
            day_count=_day_count(str(_required(entry, "day_count"))),
            calendar=SwedenCalendar(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
    elif kind == "swap":
        instrument = VanillaSwap(
            start=_required(entry, "start"),
            maturity=_required(entry, "maturity"),
            fixed_rate=float(_required(entry, "fixed_rate")),
            fixed_frequency=int(_required(entry, "fixed_frequency")),
            fixed_day_count=_day_count(str(_required(entry, "fixed_day_count"))),
            float_tenor=str(_required(entry, "float_tenor")),
            float_day_count=_day_count(str(_required(entry, "float_day_count"))),
            calendar=SwedenCalendar(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
            notional=1.0,
            pay_fixed=bool(_required(entry, "pay_fixed")),
        )
    else:
        raise PortfolioError(f"unknown instrument kind {kind!r}")
    return Position(label=label, instrument=instrument, notional=notional)


def _quote_scale(instrument: Instrument) -> float:
    """The face amount ``price()`` quotes this instrument against.

    Bonds and bills are quoted per ``face`` (100 by default); swaps carry their own notional,
    which the TOML loader pins to 1.0 so the position notional is the only size that matters.
    """
    return float(getattr(instrument, "face", 1.0))


def _position_value(position: Position, curves: CurveSet, asof: date) -> float:
    unit = price(position.instrument, curves, asof).dirty
    return position.notional / _quote_scale(position.instrument) * unit


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
    """Change in book value per unit move in each key rate, by central difference.

    This is :func:`yieldcurve.risk.keyrate.krd` without its ``/ base`` normalisation. The
    division is dropped on purpose: a swap struck at its own par rate has a base price of
    zero, so a duration is undefined for it while a money sensitivity is not.

    Because the Ho (1992) hats partition unity, the values sum to the portfolio's parallel
    sensitivity — which is what makes the ladder checkable.
    """
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
    """P&L for each row of observed daily rate changes, by bucket exposure.

    Each row of ``changes`` holds absolute zero-rate moves at ``tenors``. Those are linearly
    interpolated onto ``keys`` (flat beyond the ends) and contracted against the portfolio's
    bucket exposures. This is a first-order approximation: it is fast enough to run over a
    multi-year sample, and the convexity it omits is second order in a daily move.

    Raises:
        PortfolioError: if ``changes`` is not 2-D with one column per tenor.
    """
    if changes.ndim != 2 or changes.shape[1] != len(tenors):
        raise PortfolioError(
            f"changes must have {len(tenors)} columns to match tenors, got shape {changes.shape}"
        )
    exposure = bucket_exposure(portfolio, curves, asof, keys)
    grid = np.asarray(tenors, dtype=np.float64)
    targets = np.asarray(list(exposure), dtype=np.float64)
    projected = np.column_stack(
        [np.interp(targets, grid, row) for row in np.asarray(changes, dtype=np.float64)]
    ).T
    return projected @ np.asarray(list(exposure.values()), dtype=np.float64)


def var_es(pnl: npt.NDArray[np.float64], *, confidence: float = 0.99) -> tuple[float, float]:
    """Historical value-at-risk and expected shortfall, as positive loss magnitudes.

    Raises:
        PortfolioError: if ``confidence`` is outside ``(0, 1)``, or if fewer than ten
            observations fall beyond the quantile — an expected shortfall averaged over a
            handful of points is not a measurement.
    """
    if not 0.0 < confidence < 1.0:
        raise PortfolioError(f"confidence must lie in (0, 1), got {confidence}")
    losses = -np.asarray(pnl, dtype=np.float64).ravel()
    threshold = float(np.quantile(losses, confidence))
    tail = losses[losses >= threshold]
    if tail.size < _MIN_TAIL_OBSERVATIONS:
        raise PortfolioError(
            f"only {tail.size} observations in the {confidence:.0%} tail; "
            f"at least {_MIN_TAIL_OBSERVATIONS} are needed for an expected shortfall"
        )
    return threshold, float(tail.mean())
