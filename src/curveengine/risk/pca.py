"""Principal components of daily curve changes.

Level, slope and curvature, fitted to the committed Treasury history. Reported
as durations: the price sensitivity to a one-standard-deviation daily move in
each component, which is a number a risk report can use directly, unlike a
loading vector.

One limitation to carry into the README. These components describe how the
curve has moved historically. They are a *statistical* decomposition, not an
arbitrage-free model, and they cannot price anything. That is the division of
labour with Phase 5: Hull-White prices, PCA describes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from curveengine.curves.protocol import CurveSet
from curveengine.instruments import Instrument
from curveengine.pricing import price
from curveengine.risk.keyrate import piecewise_linear
from curveengine.risk.scenarios import Scenario, shift_curveset

_COMPONENT_NAMES = ("level", "slope", "curvature")
_MIN_OBSERVATIONS_PER_COMPONENT = 30


@dataclass(frozen=True)
class PCAResult:
    """Fitted components. ``loadings[k]`` is a unit vector over ``tenors``."""

    tenors: tuple[float, ...]
    loadings: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    component_sd: tuple[float, ...]
    n_observations: int


def daily_changes(history: pd.DataFrame) -> tuple[np.ndarray, tuple[float, ...]]:
    wide = history.pivot(index="date", columns="tenor_years", values="rate").sort_index()  # noqa: PD010
    if wide.isna().to_numpy().any():
        raise ValueError(
            "History has gaps; the adapter is expected to intersect dates across "
            "tenors so the matrix is rectangular"
        )
    return np.diff(wide.to_numpy(), axis=0), tuple(float(c) for c in wide.columns)


def fit_pca(changes: np.ndarray, tenors: Sequence[float], *, n_components: int = 3) -> PCAResult:
    if changes.ndim != 2:
        raise ValueError(f"changes must be 2-D, got shape {changes.shape}")
    if changes.shape[0] < _MIN_OBSERVATIONS_PER_COMPONENT * n_components:
        raise ValueError(
            f"{changes.shape[0]} observations is too few for {n_components} "
            f"components; want at least {_MIN_OBSERVATIONS_PER_COMPONENT * n_components}"
        )
    if changes.shape[1] != len(tenors):
        raise ValueError(
            f"changes has {changes.shape[1]} columns but {len(tenors)} tenors were given"
        )

    centred = changes - changes.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)

    variances = singular**2 / (centred.shape[0] - 1)
    total = float(variances.sum())
    loadings = right[:n_components]
    scores = centred @ loadings.T

    return PCAResult(
        tenors=tuple(float(t) for t in tenors),
        loadings=loadings,
        explained_variance_ratio=tuple(float(v / total) for v in variances[:n_components]),
        component_sd=tuple(float(s) for s in scores.std(axis=0, ddof=1)),
        n_observations=int(centred.shape[0]),
    )


def pca_durations(
    instrument: Instrument, curves: CurveSet, asof: date, result: PCAResult
) -> dict[str, float]:
    base = price(instrument, curves, asof).dirty
    durations: dict[str, float] = {}
    for index, name in enumerate(_COMPONENT_NAMES[: result.loadings.shape[0]]):
        move = result.loadings[index] * result.component_sd[index]
        shifts = dict(zip(result.tenors, (float(v) for v in move), strict=True))
        up = Scenario(name=name, shift=piecewise_linear(result.tenors, shifts))
        down = Scenario(
            name=f"{name} down",
            shift=piecewise_linear(result.tenors, {k: -v for k, v in shifts.items()}),
        )
        p_up = price(instrument, shift_curveset(curves, up), asof).dirty
        p_down = price(instrument, shift_curveset(curves, down), asof).dirty
        scale = float(np.linalg.norm(move))
        durations[name] = (p_down - p_up) / (2.0 * scale * base)
    return durations
