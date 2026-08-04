"""Nelson-Siegel and Svensson parametric curves.

Svensson rather than plain Nelson-Siegel because central banks publish Svensson:
the ECB, the Bundesbank and the Riksbank have all used the six-parameter form, so
a Svensson implementation can be checked against somebody else's published
parameters. That external check is the strongest validation in this project.

The parametric fit is the bootstrap's counterpart, not its competitor. The
bootstrap reprices every quote exactly and is therefore the pricing curve; the
parametric fit smooths through the quotes, and is therefore what you use to say
something about the *shape* of the curve — level, slope, curvature — and to
extrapolate past the last quote without a discontinuity.

Fit contract: inputs are validated before any arithmetic (finite, positive,
strictly increasing times; finite zero rates; finite positive weights; at least
one observation per parameter). The fit result reports the optimizer's own
verdict, boundary saturation, Jacobian rank/condition, and residual metrics as
explicit typed fields, and a fit that is unsuccessful, boundary-saturated, or
whose Jacobian cannot identify the parameters is rejected with ``FitError``
instead of being silently accepted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

_SEED = 20260727
_MIN_TAU = 1e-3
_CONSTRAINT_FLOOR = 1e-6
_BOUNDARY_TOL = 1e-8
_MAX_JACOBIAN_CONDITION = 1e12
_JACOBIAN_STEP = 1e-6

_SV_PARAM_NAMES = ("b0", "b1", "b2", "b3", "tau1", "tau2")
_NS_PARAM_NAMES = ("b0", "b1", "b2", "tau")

_SV_BOUNDS: tuple[tuple[float, float], ...] = (
    (_CONSTRAINT_FLOOR, 0.25),
    (-0.25, 0.25),
    (-0.25, 0.25),
    (-0.25, 0.25),
    (_MIN_TAU, 30.0),
    (_MIN_TAU, 30.0),
)
_NS_BOUNDS: tuple[tuple[float, float], ...] = (
    (_CONSTRAINT_FLOOR, 0.25),
    (-0.25, 0.25),
    (-0.25, 0.25),
    (_MIN_TAU, 30.0),
)


class FitError(ValueError):
    """A parametric curve cannot be constructed or fitted as requested."""


@dataclass(frozen=True)
class ParametricFitResult[CurveT]:
    """The outcome of a parametric fit, with the optimizer's verdict explicit.

    ``success`` is informational: the fit either succeeds or raises
    :class:`FitError` (optimizer failure, boundary saturation, or a Jacobian
    that cannot identify the parameters), so a returned result always has
    ``success is True`` and an empty ``saturated_parameters``. The Jacobian
    rank and condition describe how well
    the data identifies the parameters (scipy's differential evolution provides
    no covariance, so the Jacobian state is the reported uncertainty
    diagnostic). Residual metrics are unweighted and computed on the fit data.
    """

    curve: CurveT
    success: bool
    optimizer_status: str
    saturated_parameters: tuple[str, ...]
    jacobian_rank: int
    jacobian_condition: float
    rmse: float
    max_abs_error: float


def _factor(t: float, tau: float) -> float:
    """The Nelson-Siegel slope loading ``(1 - exp(-t/tau)) / (t/tau)``."""
    if t <= 0.0:
        return 1.0
    x = t / tau
    return (1.0 - math.exp(-x)) / x


def _require_curve_time(t: float) -> None:
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"Curve time must be finite and non-negative, got {t}")


def _svensson_zero(
    t: float, b0: float, b1: float, b2: float, b3: float, tau1: float, tau2: float
) -> float:
    loading1 = _factor(t, tau1)
    loading2 = _factor(t, tau2)
    return (
        b0
        + b1 * loading1
        + b2 * (loading1 - math.exp(-t / tau1))
        + b3 * (loading2 - math.exp(-t / tau2))
    )


def _ns_zero(t: float, b0: float, b1: float, b2: float, tau: float) -> float:
    loading = _factor(t, tau)
    return b0 + b1 * loading + b2 * (loading - math.exp(-t / tau))


def _svensson_zeros(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3, tau1, tau2 = (float(v) for v in p)
    x1 = t / tau1
    x2 = t / tau2
    loading1 = (1.0 - np.exp(-x1)) / x1
    loading2 = (1.0 - np.exp(-x2)) / x2
    return b0 + b1 * loading1 + b2 * (loading1 - np.exp(-x1)) + b3 * (loading2 - np.exp(-x2))


def _ns_zeros(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    b0, b1, b2, tau = (float(v) for v in p)
    x = t / tau
    loading = (1.0 - np.exp(-x)) / x
    return b0 + b1 * loading + b2 * (loading - np.exp(-x))


def _numerical_jacobian(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray], t: np.ndarray, p: np.ndarray
) -> np.ndarray:
    n = len(p)
    jac = np.zeros((len(t), n))
    for j in range(n):
        plus = p.copy()
        minus = p.copy()
        plus[j] += _JACOBIAN_STEP
        minus[j] -= _JACOBIAN_STEP
        jac[:, j] = (model(t, plus) - model(t, minus)) / (2.0 * _JACOBIAN_STEP)
    return jac


def _validate_fit_inputs(
    times: Sequence[float],
    zeros: Sequence[float],
    weights: Sequence[float] | None,
    *,
    n_params: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate fit inputs before any arithmetic; returns (times, zeros, weights)."""
    if len(times) != len(zeros):
        raise FitError(f"{len(times)} times but {len(zeros)} zero rates")
    if weights is not None and len(weights) != len(times):
        raise FitError(f"{len(weights)} weights but {len(times)} observations")
    try:
        t = np.asarray(times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FitError(f"{label} times must be numeric, got {times!r}") from exc
    try:
        z = np.asarray(zeros, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FitError(f"{label} zero rates must be numeric, got {zeros!r}") from exc
    try:
        w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise FitError(f"{label} weights must be numeric, got {weights!r}") from exc
    if t.size == 0:
        raise FitError(f"{label} got no observations")
    if t.size < n_params:
        raise FitError(
            f"{label} has {n_params} parameters; {t.size} observations cannot identify it"
        )
    if not np.all(np.isfinite(t)):
        bad = [float(x) for x in t if not math.isfinite(float(x))]
        raise FitError(f"{label} times must be finite, got non-finite {bad}")
    if not np.all(np.isfinite(z)):
        bad = [float(x) for x in z if not math.isfinite(float(x))]
        raise FitError(f"{label} zero rates must be finite, got non-finite {bad}")
    if not np.all(np.isfinite(w)):
        bad = [float(x) for x in w if not math.isfinite(float(x))]
        raise FitError(f"{label} weights must be finite, got non-finite {bad}")
    if np.any(t <= 0.0):
        raise FitError(f"{label} times must be positive (t = 0 is the reference date), got {times}")
    if np.any(w <= 0.0):
        raise FitError(f"{label} weights must be strictly positive, got {weights}")
    if np.any(t[1:] <= t[:-1]):
        raise FitError(f"{label} times must be strictly increasing with no duplicates, got {times}")
    return t, z, w


def _saturated_parameters(
    p: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    param_names: tuple[str, ...],
    extra: tuple[tuple[str, bool], ...],
) -> tuple[str, ...]:
    """Names of parameters pressed against a bound or a fit constraint."""
    saturated = [
        name
        for name, (lo, hi), value in zip(param_names, bounds, p, strict=True)
        if value <= lo + _BOUNDARY_TOL or value >= hi - _BOUNDARY_TOL
    ]
    saturated.extend(name for name, pressed in extra if pressed)
    return tuple(saturated)


def _fit_and_report(
    *,
    label: str,
    param_names: tuple[str, ...],
    bounds: tuple[tuple[float, float], ...],
    objective: Callable[[np.ndarray], float],
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    extra_constraints: tuple[tuple[str, Callable[[np.ndarray], bool]], ...],
    t: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, str, int, float, float, float]:
    """Run the optimizer and assemble the fit diagnostics.

    Returns (best parameters, optimizer status, jacobian rank, jacobian
    condition, rmse, max absolute error). A fit that reports optimizer failure,
    presses a parameter against its bounds, or whose Jacobian is singular or
    ill-conditioned is rejected with ``FitError``.
    """
    result: OptimizeResult = differential_evolution(
        objective,
        list(bounds),
        seed=seed,
        polish=True,
        strategy="best1bin",
        maxiter=5000,
        tol=1e-8,
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.7,
    )
    if not result.success:
        raise FitError(
            f"{label} fit did not converge ({result.message}); refusing to report a curve "
            "from an unsuccessful optimization"
        )
    fun = float(result.fun)
    if not math.isfinite(fun):
        raise FitError(
            f"{label} fit found no feasible parameter vector (objective value {fun}); "
            "refusing to report a curve that does not exist"
        )

    best = np.asarray(result.x, dtype=float)
    saturated = _saturated_parameters(
        best,
        bounds,
        param_names,
        tuple((name, predicate(best)) for name, predicate in extra_constraints),
    )
    if saturated:
        raise FitError(
            f"{label} fit saturates parameter bound(s) {saturated}; the optimum lies on "
            "the boundary of the supported domain — refusing to report a boundary fit"
        )

    jac = _numerical_jacobian(model, t, best)
    jacobian_rank = int(np.linalg.matrix_rank(jac))
    # Column-normalize before the condition number so the rejection threshold
    # is unit-free: the raw Jacobian mixes unitless rate parameters (~1e-2)
    # with tau parameters in years (~1), and that unit mismatch alone would
    # inflate the condition number. Rank is invariant under column scaling, so
    # the rank check below is unaffected; a zero column (kept zero here) is
    # rank-deficient and rejected there regardless.
    column_norms = np.linalg.norm(jac, axis=0)
    normalized_jac = jac / np.where(column_norms > 0.0, column_norms, 1.0)
    jacobian_condition = float(np.linalg.cond(normalized_jac))
    if jacobian_rank < len(param_names):
        raise FitError(
            f"{label} fit is underidentified: Jacobian rank {jacobian_rank} < "
            f"{len(param_names)} parameters; the data cannot identify the fitted parameters"
        )
    if not math.isfinite(jacobian_condition) or jacobian_condition > _MAX_JACOBIAN_CONDITION:
        raise FitError(
            f"{label} fit has an ill-conditioned Jacobian (condition "
            f"{jacobian_condition:.3g}); parameter estimates are numerically unreliable"
        )

    residuals = model(t, best) - z
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_abs_error = float(np.max(np.abs(residuals)))
    return best, result.message, jacobian_rank, jacobian_condition, rmse, max_abs_error


def _result_from[CurveT](
    curve: CurveT,
    status: str,
    jacobian_rank: int,
    jacobian_condition: float,
    rmse: float,
    max_abs_error: float,
) -> ParametricFitResult[CurveT]:
    return ParametricFitResult(
        curve=curve,
        success=True,
        optimizer_status=status,
        saturated_parameters=(),
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        rmse=rmse,
        max_abs_error=max_abs_error,
    )


@dataclass(frozen=True)
class Svensson:
    """The six-parameter Svensson zero curve, continuously compounded.

    ``zero(t) = b0
              + b1 * L(t/tau1)
              + b2 * (L(t/tau1) - exp(-t/tau1))
              + b3 * (L(t/tau2) - exp(-t/tau2))``

    where ``L(x) = (1 - exp(-x)) / x``. ``b0`` is the long-run level, ``b0 + b1``
    the instantaneous short rate, and the two hump terms carry curvature at
    separate horizons — which is what lets Svensson fit a twin-humped curve that
    Nelson-Siegel cannot.
    """

    beta: tuple[float, float, float, float]
    tau: tuple[float, float]
    reference_date: date

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (*self.beta, *self.tau)):
            raise FitError(f"Parameters must be finite, got beta={self.beta} tau={self.tau}")
        if self.tau[0] <= 0.0 or self.tau[1] <= 0.0:
            raise FitError(f"Both tau must be positive, got {self.tau}")

    def zero(self, t: float) -> float:
        _require_curve_time(t)
        b0, b1, b2, b3 = self.beta
        tau1, tau2 = self.tau
        return _svensson_zero(t, b0, b1, b2, b3, tau1, tau2)

    def instantaneous_fwd(self, t: float) -> float:
        """Closed-form instantaneous forward rate f(t) = d/dt[t * zero(t)].

        f(t) = b0 + b1*e^{-t/tau1} + b2*(t/tau1)*e^{-t/tau1} + b3*(t/tau2)*e^{-t/tau2}
        """
        _require_curve_time(t)
        b0, b1, b2, b3 = self.beta
        tau1, tau2 = self.tau
        return (
            b0
            + b1 * math.exp(-t / tau1)
            + b2 * (t / tau1) * math.exp(-t / tau1)
            + b3 * (t / tau2) * math.exp(-t / tau2)
        )

    def df(self, t: float) -> float:
        _require_curve_time(t)
        if t == 0.0:
            return 1.0
        return math.exp(-self.zero(t) * t)

    def fwd(self, t1: float, t2: float) -> float:
        if not (math.isfinite(t1) and math.isfinite(t2)):
            raise ValueError(f"Curve times must be finite, got ({t1}, {t2})")
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        if t1 < 0.0:
            raise ValueError(f"Curve time must be finite and non-negative, got {t1}")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def rmse(self, times: Sequence[float], zeros: Sequence[float]) -> float:
        t, z, _ = _validate_fit_inputs(times, zeros, None, n_params=1, label="rmse")
        return float(np.sqrt(np.mean((np.array([self.zero(float(x)) for x in t]) - z) ** 2)))

    @classmethod
    def fit(
        cls,
        times: Sequence[float],
        zeros: Sequence[float],
        reference_date: date,
        *,
        seed: int = _SEED,
        weights: Sequence[float] | None = None,
    ) -> ParametricFitResult[Svensson]:
        t, z, w = _validate_fit_inputs(times, zeros, weights, n_params=6, label="Svensson")

        def objective(p: np.ndarray) -> float:
            b0, b1, b2, b3, tau1, tau2 = (float(v) for v in p)
            if b0 + b1 <= _CONSTRAINT_FLOOR or tau2 - tau1 <= _MIN_TAU:
                return np.inf
            model = np.array([_svensson_zero(float(x), b0, b1, b2, b3, tau1, tau2) for x in t])
            return float(np.sum(w * (model - z) ** 2))

        best, status, rank, cond, rmse, max_abs = _fit_and_report(
            label="Svensson",
            param_names=_SV_PARAM_NAMES,
            bounds=_SV_BOUNDS,
            objective=objective,
            model=_svensson_zeros,
            extra_constraints=(
                ("b0+b1", lambda p: p[0] + p[1] <= _CONSTRAINT_FLOOR + _BOUNDARY_TOL),
                ("tau2-tau1", lambda p: p[5] - p[4] <= _MIN_TAU + _BOUNDARY_TOL),
            ),
            t=t,
            z=z,
            w=w,
            seed=seed,
        )
        curve = cls(
            beta=(float(best[0]), float(best[1]), float(best[2]), float(best[3])),
            tau=(float(best[4]), float(best[5])),
            reference_date=reference_date,
        )
        return _result_from(curve, status, rank, cond, rmse, max_abs)


@dataclass(frozen=True)
class NelsonSiegel:
    """The four-parameter Nelson-Siegel curve: Svensson without the second hump.

    Evaluation is closed form and never constructs a ``Svensson`` object: the
    fourth term of the Svensson form is identically zero, so the two taus of
    the six-parameter form would be pure overhead per evaluation.
    """

    beta: tuple[float, float, float]
    tau: float
    reference_date: date

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (*self.beta, self.tau)):
            raise FitError(f"Parameters must be finite, got beta={self.beta} tau={self.tau}")
        if self.tau <= 0.0:
            raise FitError(f"tau must be positive, got {self.tau}")

    def zero(self, t: float) -> float:
        _require_curve_time(t)
        b0, b1, b2 = self.beta
        return _ns_zero(t, b0, b1, b2, self.tau)

    def df(self, t: float) -> float:
        _require_curve_time(t)
        if t == 0.0:
            return 1.0
        return math.exp(-self.zero(t) * t)

    def fwd(self, t1: float, t2: float) -> float:
        if not (math.isfinite(t1) and math.isfinite(t2)):
            raise ValueError(f"Curve times must be finite, got ({t1}, {t2})")
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        if t1 < 0.0:
            raise ValueError(f"Curve time must be finite and non-negative, got {t1}")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def rmse(self, times: Sequence[float], zeros: Sequence[float]) -> float:
        t, z, _ = _validate_fit_inputs(times, zeros, None, n_params=1, label="rmse")
        return float(np.sqrt(np.mean((np.array([self.zero(float(x)) for x in t]) - z) ** 2)))

    @classmethod
    def fit(
        cls,
        times: Sequence[float],
        zeros: Sequence[float],
        reference_date: date,
        *,
        seed: int = _SEED,
        weights: Sequence[float] | None = None,
    ) -> ParametricFitResult[NelsonSiegel]:
        t, z, w = _validate_fit_inputs(times, zeros, weights, n_params=4, label="Nelson-Siegel")

        def objective(p: np.ndarray) -> float:
            b0, b1, b2, tau = (float(v) for v in p)
            if b0 + b1 <= _CONSTRAINT_FLOOR:
                return np.inf
            model = np.array([_ns_zero(float(x), b0, b1, b2, tau) for x in t])
            return float(np.sum(w * (model - z) ** 2))

        best, status, rank, cond, rmse, max_abs = _fit_and_report(
            label="Nelson-Siegel",
            param_names=_NS_PARAM_NAMES,
            bounds=_NS_BOUNDS,
            objective=objective,
            model=_ns_zeros,
            extra_constraints=(
                ("b0+b1", lambda p: p[0] + p[1] <= _CONSTRAINT_FLOOR + _BOUNDARY_TOL),
            ),
            t=t,
            z=z,
            w=w,
            seed=seed,
        )
        curve = cls(
            beta=(float(best[0]), float(best[1]), float(best[2])),
            tau=float(best[3]),
            reference_date=reference_date,
        )
        return _result_from(curve, status, rank, cond, rmse, max_abs)
