"""Principal components of daily zero-rate changes.

This module fits a principal-component decomposition to a history of daily
zero-rate changes observed on a common tenor grid, and derives two curve-risk
quantities per component, with explicit units:

- :func:`pca_durations` — the modified duration along each component's
  *unit-norm* loading direction: the fractional price change per unit (1.0
  decimal) shift of the zero curve along that direction, in years. This is a
  direction-only measure: the component's empirical volatility is not
  involved.
- :func:`pca_exposure` — the fractional price change for a
  *one-standard-deviation* move along the component: the direction duration
  scaled by the component's empirical standard deviation
  (``PCAResult.component_sd``), so the empirical scale is retained.

Signs are deterministic: every loading is oriented so its largest-magnitude
entry is positive (ties resolve to the earliest tenor index), so repeated fits
on the same history return identical components and the sign does not flip
with the overall sign of the history.

Components are named ``PC1``, ``PC2``, ... unless the loading's sign pattern
matches the documented economic criterion for that position — no sign change
across tenors for PC1 (level), exactly one for PC2 (slope), exactly two for
PC3 (curvature) — in which case the economic label is used. The diagnostic
behind each decision is available as ``PCAResult.loading_shape``.

Degenerate histories — constant, rank-deficient, or non-finite — are rejected
before any arithmetic.

These components describe how the curve has moved historically. They are a
*statistical* decomposition, not an arbitrage-free model, and they cannot
price anything. That is the division of labour with the Hull-White module:
Hull-White prices, PCA describes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import Instrument
from yieldcurve.risk._validators import _require_unit_price
from yieldcurve.risk.keyrate import piecewise_linear
from yieldcurve.risk.scenarios import Scenario, shift_curveset

_COMPONENT_NAMES = ("level", "slope", "curvature")
_EXPECTED_SIGN_CHANGES = (0, 1, 2)
_MIN_OBSERVATIONS_PER_COMPONENT = 30
_DIRECTION_BUMP = 1e-4
"""Probe for the direction duration: a 1bp shift along the unit-norm loading
direction, small enough for the central difference to be a faithful derivative,
and normalized out so the result is per unit (1.0) rate."""


class PCAError(ValueError):
    """Raised when a PCA input or a derived risk request is malformed."""


@dataclass(frozen=True)
class PCAResult:
    """Fitted principal components.

    ``loadings[k]`` is the unit-norm direction of component ``k`` over
    ``tenors`` (dimensionless; each row has Euclidean norm 1 and is
    deterministically oriented). ``component_sd[k]`` is the sample standard
    deviation of the component's scores — the empirical scale of a
    one-standard-deviation move. ``explained_variance_ratio[k]`` is the share
    of total centred variance the component accounts for.
    ``component_names[k]`` is ``PC{k+1}`` unless the loading-shape criterion
    for that position passes, in which case it is the economic label
    (level/slope/curvature); ``loading_shape[k]`` carries the diagnostic the
    naming decision was based on.
    """

    tenors: tuple[float, ...]
    loadings: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    component_sd: tuple[float, ...]
    n_observations: int
    component_names: tuple[str, ...]
    loading_shape: tuple[str, ...]


def _sign_changes(loading: np.ndarray) -> int:
    """Sign changes across tenors; a zero loading counts as a transition."""
    return int(np.sum(np.diff(np.sign(loading)) != 0))


def _orient(loadings: np.ndarray) -> np.ndarray:
    """Deterministic sign orientation: flip each row so its largest-magnitude
    entry is positive. ``argmax`` returns the first maximum, so an exact tie
    resolves to the earliest tenor index — fully deterministic."""
    oriented = loadings.copy()
    for row in oriented:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return oriented


def _component_names(loadings: np.ndarray) -> tuple[str, ...]:
    """Economic labels only when the loading-shape criterion for the component's
    position passes; otherwise the component stays ``PC{k+1}``."""
    names = []
    for index, row in enumerate(loadings):
        if index < len(_COMPONENT_NAMES) and _sign_changes(row) == _EXPECTED_SIGN_CHANGES[index]:
            names.append(_COMPONENT_NAMES[index])
        else:
            names.append(f"PC{index + 1}")
    return tuple(names)


def _shape_description(loading: np.ndarray) -> str:
    changes = _sign_changes(loading)
    if changes == 0:
        return "sign-consistent (no sign change)"
    if changes == 1:
        return "one sign change"
    if changes == 2:
        return "two sign changes"
    return f"{changes} sign changes"


def _require_base_price(base: float, instrument: Instrument, measure: str) -> None:
    """Reject normalizing by a materially zero present value, as a PCAError."""
    try:
        _require_unit_price(base, instrument, measure)
    except ValueError as exc:
        raise PCAError(str(exc)) from None


def daily_changes(history: pd.DataFrame) -> tuple[np.ndarray, tuple[float, ...]]:
    """Wide daily first differences of a long-form rate history.

    ``history`` must be long form with ``date``, ``tenor_years`` and ``rate``
    columns and every tenor observed on every date, so the rows align by date
    across tenors (a complete rectangular grid).

    Raises:
        PCAError: if the required columns are missing, the history is empty,
            or any cell is missing or non-finite (a misaligned history).
    """
    try:
        wide = history.pivot(index="date", columns="tenor_years", values="rate").sort_index()  # noqa: PD010
    except (KeyError, ValueError) as exc:
        raise PCAError(
            "history must be long form with 'date', 'tenor_years' and 'rate' "
            f"columns and unique (date, tenor) pairs: {exc}"
        ) from None
    if wide.empty:
        raise PCAError("history has no observations")
    try:
        values = wide.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PCAError(f"rates must be numeric: {exc}") from None
    if not np.isfinite(values).all():
        raise PCAError(
            "history is not a complete rectangular grid: every tenor must be "
            "observed on every date with a finite rate"
        )
    return np.diff(values, axis=0), tuple(float(c) for c in wide.columns)


def fit_pca(changes: np.ndarray, tenors: Sequence[float], *, n_components: int = 3) -> PCAResult:
    """Fit PCA to centred daily zero-rate changes.

    The empirical covariance is decomposed by SVD of the centred history; the
    loadings are the right singular vectors, deterministically oriented (see
    the module docstring). ``component_sd`` is the sample standard deviation
    of each component's scores over the history.

    Raises:
        PCAError: if ``changes`` is not a numeric finite 2-D array with one
            column per tenor, if ``tenors`` is not a strictly ascending finite
            grid of at least two values, if too few observations are present
            for ``n_components``, if ``n_components`` cannot be identified
            from the shape, or if the history is degenerate (zero variance —
            constant — or rank-deficient).
    """
    try:
        arr = np.asarray(changes, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PCAError(f"changes must be numeric, got: {exc}") from None
    if arr.ndim != 2:
        raise PCAError(f"changes must be 2-D, got shape {arr.shape}")
    if arr.shape[1] != len(tenors):
        raise PCAError(f"changes has {arr.shape[1]} columns but {len(tenors)} tenors were given")
    ten = np.asarray(tenors, dtype=np.float64)
    if ten.size < 2 or not np.isfinite(ten).all():
        raise PCAError(f"tenors must be finite and contain at least two values, got {tenors}")
    if np.any(np.diff(ten) <= 0.0):
        raise PCAError(f"tenors must be strictly increasing, got {tenors}")
    if n_components < 1 or n_components > min(arr.shape[0] - 1, arr.shape[1]):
        raise PCAError(
            f"cannot fit {n_components} components from shape {arr.shape}: at most "
            f"{min(arr.shape[0] - 1, arr.shape[1])} are identifiable"
        )
    if arr.shape[0] < _MIN_OBSERVATIONS_PER_COMPONENT * n_components:
        raise PCAError(
            f"{arr.shape[0]} observations is too few for {n_components} "
            f"components; want at least {_MIN_OBSERVATIONS_PER_COMPONENT * n_components}"
        )
    if not np.isfinite(arr).all():
        raise PCAError("changes must be finite, got non-finite entries")

    centred = arr - arr.mean(axis=0, keepdims=True)
    total = float((centred**2).sum() / (arr.shape[0] - 1))
    if not math.isfinite(total) or total <= 0.0:
        raise PCAError(
            "history is degenerate: zero variance (all rows identical); "
            "no principal components exist"
        )
    rank = int(np.linalg.matrix_rank(centred))
    if rank < n_components:
        raise PCAError(
            f"history is rank-deficient: rank {rank} is below the {n_components} "
            "components requested; the trailing singular values are numerically zero"
        )

    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    variances = singular**2 / (arr.shape[0] - 1)
    loadings = _orient(right[:n_components])
    scores = centred @ loadings.T

    return PCAResult(
        tenors=tuple(float(t) for t in tenors),
        loadings=loadings,
        explained_variance_ratio=tuple(float(v / total) for v in variances[:n_components]),
        component_sd=tuple(float(s) for s in scores.std(axis=0, ddof=1)),
        n_observations=int(centred.shape[0]),
        component_names=_component_names(loadings),
        loading_shape=tuple(_shape_description(loadings[k]) for k in range(loadings.shape[0])),
    )


def _price_change_per_component(
    instrument: Instrument,
    curves: CurveSet,
    asof: date,
    result: PCAResult,
    bump: Sequence[float],
) -> dict[str, float]:
    """Central-difference dirty-price change per component for a move of
    ``bump[k] * loadings[k]``, keyed by component name."""
    changes: dict[str, float] = {}
    for index, name in enumerate(result.component_names):
        move = result.loadings[index] * bump[index]
        shifts = dict(zip(result.tenors, (float(v) for v in move), strict=True))
        up = Scenario(name=name, shift=piecewise_linear(result.tenors, shifts))
        down = Scenario(
            name=f"{name} down",
            shift=piecewise_linear(result.tenors, {k: -v for k, v in shifts.items()}),
        )
        p_up = price(instrument, shift_curveset(curves, up), asof).dirty
        p_down = price(instrument, shift_curveset(curves, down), asof).dirty
        changes[name] = p_down - p_up
    return changes


def pca_durations(
    instrument: Instrument, curves: CurveSet, asof: date, result: PCAResult
) -> dict[str, float]:
    """Modified duration along each component's unit-norm loading direction.

    The fractional price change for a unit (1.0 decimal) shift of the zero
    curve along the component's direction, measured with a 1bp probe and
    normalized out. Units: years (per 1.0 rate shift along the direction).
    This is the direction-only measure: the component's empirical standard
    deviation (``PCAResult.component_sd``) is not involved — the
    one-standard-deviation exposure is :func:`pca_exposure`.

    Raises:
        PCAError: if the instrument's base price is materially zero (a
            normalized measure; error policy).
    """
    base = price(instrument, curves, asof).dirty
    _require_base_price(base, instrument, "pca_durations")
    changes = _price_change_per_component(
        instrument, curves, asof, result, (_DIRECTION_BUMP,) * result.loadings.shape[0]
    )
    return {name: change / (2.0 * _DIRECTION_BUMP * base) for name, change in changes.items()}


def pca_exposure(
    instrument: Instrument, curves: CurveSet, asof: date, result: PCAResult
) -> dict[str, float]:
    """Fractional price change per one-standard-deviation move along a component.

    The central-difference price change for a shift of ``component_sd[k]``
    along the component's unit-norm direction, divided by the base price.
    Dimensionless; the empirical component scale (``component_sd``) enters the
    move explicitly, so the exposure retains the empirical scale — unlike
    :func:`pca_durations`, which is direction-only. Up to central-difference
    curvature, ``pca_exposure[k] == pca_durations[k] * component_sd[k]``.

    Raises:
        PCAError: if the instrument's base price is materially zero (a
            normalized measure; error policy).
    """
    base = price(instrument, curves, asof).dirty
    _require_base_price(base, instrument, "pca_exposure")
    changes = _price_change_per_component(instrument, curves, asof, result, result.component_sd)
    return {name: change / (2.0 * base) for name, change in changes.items()}
