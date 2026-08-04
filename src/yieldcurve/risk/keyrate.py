"""Key-rate durations following Ho (1992).

The curve is shocked one key at a time by a triangular shift that is full size
at its own key, falls linearly to zero at the neighbouring keys, and stays flat
beyond the first and last keys. Those flat tails make the shifts a partition of
unity, which is what makes the key-rate durations sum to the parallel-shift
duration — up to the O(bump^2) truncation error of the central finite
differences, not exactly (the module previously claimed exactness).

Grids come from the design specification section 2. The SEK 1y point is
interpolated rather than observed — the Riksbank publishes 6m bills and 2y
benchmarks with nothing between — and that has to be said wherever the SEK
key-rate profile is reported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Final

from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import Instrument
from yieldcurve.risk.scenarios import Scenario, shift_curveset
from yieldcurve.risk.sensitivities import MIN_UNIT_PRICE, instrument_scale

SEK_KEY_RATES: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)
USD_KEY_RATES: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

_BASIS_POINT = 1e-4

KRD_UNITS = "price-bp per yield-bp"
"""KRD units: a 1bp rise in a key rate moves the price by that many price basis
points (1 price bp = 1e-4 of price), numerically equal to years of duration.
Not multiplied by 100."""


def _validate(keys: Sequence[float]) -> None:
    if len(keys) < 2:
        raise ValueError("At least two key rates are required")
    if any(not math.isfinite(k) for k in keys):
        raise ValueError(f"Key rates must be finite, got {tuple(keys)}")
    if any(b <= a for a, b in pairwise(keys)):
        raise ValueError(f"Key rates must be strictly ascending, got {tuple(keys)}")


def _require_bump(bump: float) -> None:
    if not math.isfinite(bump) or bump <= 0.0:
        raise ValueError(f"krd bump must be a positive finite rate, got {bump}")


def _require_unit_price(base: float, instrument: Instrument) -> None:
    """Normalized KRD must not divide by a materially zero present value."""
    if abs(base / instrument_scale(instrument)) < MIN_UNIT_PRICE:
        raise ValueError(
            f"KRD is undefined for {type(instrument).__name__} with base price "
            f"{base:.6g}: materially zero present value; "
            "use bucket_exposure for the monetary sensitivity"
        )


def hat(keys: Sequence[float], index: int, size: float) -> Scenario:
    """The Ho (1992) triangular shift centred on ``keys[index]``."""
    _validate(keys)
    if not math.isfinite(size):
        raise ValueError(f"hat size must be finite, got {size}")
    if not 0 <= index < len(keys):
        raise IndexError(f"index {index} outside 0..{len(keys) - 1}")
    centre = keys[index]
    left = keys[index - 1] if index > 0 else None
    right = keys[index + 1] if index < len(keys) - 1 else None

    def shift(t: float) -> float:
        if t == centre:
            return size
        if t < centre:
            if left is None:
                return size
            if t <= left:
                return 0.0
            return size * (t - left) / (centre - left)
        if right is None:
            return size
        if t >= right:
            return 0.0
        return size * (right - t) / (right - centre)

    return Scenario(name=f"key {centre:g}y", shift=shift)


def piecewise_linear(
    keys: Sequence[float], shifts: Mapping[float, float]
) -> Callable[[float], float]:
    """A continuous, piecewise-linear shift built from Ho (1992) hats.

    The shift at each key equals *shifts[key]*, with linear interpolation
    between neighbouring keys and flat extrapolation beyond the endpoints.
    """
    _validate(keys)
    missing = [k for k in keys if k not in shifts]
    if missing:
        raise ValueError(f"No shift given for key rates {missing}")
    bad = [k for k in keys if not math.isfinite(shifts[k])]
    if bad:
        raise ValueError(f"Non-finite shift at key rates {bad}")

    def shift(t: float) -> float:
        return sum(hat(keys, i, shifts[key]).shift(t) for i, key in enumerate(keys))

    return shift


def krd(
    instrument: Instrument,
    curves: CurveSet,
    asof: date,
    keys: Sequence[float],
    *,
    bump: float = _BASIS_POINT,
) -> dict[float, float]:
    """Key-rate durations via central finite difference.

    Each key rate is bumped +/- *bump* (a triangular Ho 1992 shift). The result
    maps tenor → duration in years, positive for a long position — equivalently
    ``{KRD_UNITS}``: a 1bp rise in the key rate moves the price by that many
    price basis points (1 price bp = 1e-4 of price). The number is numerically
    consistent with effective duration and is not multiplied by 100.

    Because the hats partition unity, ``sum(result.values())`` equals the
    parallel-shift duration up to the O(bump^2) central-difference error.

    Raises:
        ValueError: if ``keys`` is not a strictly ascending finite grid of at
            least two tenors, if ``bump`` is not a positive finite rate, or if
            the base price is materially zero — a par swap prices at zero, so
            its normalized KRD is undefined; use portfolio ``bucket_exposure``
            for the monetary ladder.
    """
    _validate(keys)
    _require_bump(bump)
    base = price(instrument, curves, asof).dirty
    _require_unit_price(base, instrument)
    result: dict[float, float] = {}
    for index, key in enumerate(keys):
        up = shift_curveset(curves, hat(keys, index, bump))
        down = shift_curveset(curves, hat(keys, index, -bump))
        p_up = price(instrument, up, asof).dirty
        p_down = price(instrument, down, asof).dirty
        result[float(key)] = (p_down - p_up) / (2.0 * bump * base)
    return result


def bucket_pnl(
    instrument: Instrument,
    curves: CurveSet,
    asof: date,
    keys: Sequence[float],
    shifts: Mapping[float, float],
) -> float:
    """First-order P&L from key-rate shifts, in base-currency units.

    *shifts* maps key tenors to curve changes in decimals.  The approximation is
    ``-price * sum(krd * shift)`` and is exact only for infinitesimal moves;
    for larger moves the bucket error should be checked against a full reprice.
    Raises via :func:`krd` when the base price is materially zero.
    """
    durations = krd(instrument, curves, asof, keys)
    base = price(instrument, curves, asof).dirty
    return -base * sum(durations[k] * shifts[k] for k in durations)
