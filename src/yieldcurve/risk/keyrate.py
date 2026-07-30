"""Key-rate durations following Ho (1992).

The curve is shocked one key at a time by a triangular shift that is full size
at its own key, falls linearly to zero at the neighbouring keys, and stays flat
beyond the first and last keys. Those flat tails make the shifts a partition of
unity, which is what makes the key-rate durations sum to the parallel-shift
duration exactly rather than approximately.

Grids come from the design specification section 2. The SEK 1y point is
interpolated rather than observed — the Riksbank publishes 6m bills and 2y
benchmarks with nothing between — and that has to be said wherever the SEK
key-rate profile is reported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Final

from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import Instrument
from yieldcurve.risk.scenarios import Scenario, shift_curveset

SEK_KEY_RATES: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)
USD_KEY_RATES: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)

_BASIS_POINT = 1e-4


def _validate(keys: Sequence[float]) -> None:
    if len(keys) < 2:
        raise ValueError("At least two key rates are required")
    if any(b <= a for a, b in pairwise(keys)):
        raise ValueError(f"Key rates must be strictly ascending, got {tuple(keys)}")


def hat(keys: Sequence[float], index: int, size: float) -> Scenario:
    """The Ho (1992) triangular shift centred on ``keys[index]``."""
    _validate(keys)
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

    Each key rate is bumped +/- *bump* (a triangular Ho 1992 shift).  The
    result is tenors → duration in years (positive for a long position).

    Returns ``{key_tenor: duration}``.
    """
    _validate(keys)
    base = price(instrument, curves, asof).dirty
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
    """
    durations = krd(instrument, curves, asof, keys)
    base = price(instrument, curves, asof).dirty
    return -base * sum(durations[k] * shifts[k] for k in durations)
