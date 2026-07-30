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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import differential_evolution

_SEED = 20260727
_MIN_TAU = 1e-3
_CONSTRAINT_FLOOR = 1e-6


class FitError(ValueError):
    """A parametric curve cannot be constructed or fitted as requested."""


def _factor(t: float, tau: float) -> float:
    """The Nelson-Siegel slope loading ``(1 - exp(-t/tau)) / (t/tau)``."""
    if t <= 0.0:
        return 1.0
    x = t / tau
    return (1.0 - math.exp(-x)) / x


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
        if self.tau[0] <= 0.0 or self.tau[1] <= 0.0:
            raise FitError(f"Both tau must be positive, got {self.tau}")

    def zero(self, t: float) -> float:
        if t < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t}")
        b0, b1, b2, b3 = self.beta
        tau1, tau2 = self.tau
        l1 = _factor(t, tau1)
        l2 = _factor(t, tau2)
        return b0 + b1 * l1 + b2 * (l1 - math.exp(-t / tau1)) + b3 * (l2 - math.exp(-t / tau2))

    def instantaneous_fwd(self, t: float) -> float:
        """Closed-form instantaneous forward rate f(t) = d/dt[t * zero(t)].

        f(t) = b0 + b1*e^{-t/tau1} + b2*(t/tau1)*e^{-t/tau1} + b3*(t/tau2)*e^{-t/tau2}
        """
        if t < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t}")
        b0, b1, b2, b3 = self.beta
        tau1, tau2 = self.tau
        return (
            b0
            + b1 * math.exp(-t / tau1)
            + b2 * (t / tau1) * math.exp(-t / tau1)
            + b3 * (t / tau2) * math.exp(-t / tau2)
        )

    def df(self, t: float) -> float:
        if t < 0.0:
            raise ValueError(f"Curve time must be non-negative, got {t}")
        if t == 0.0:
            return 1.0
        return math.exp(-self.zero(t) * t)

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def rmse(self, times: Sequence[float], zeros: Sequence[float]) -> float:
        errors = np.array([self.zero(t) - z for t, z in zip(times, zeros, strict=True)])
        return float(np.sqrt(np.mean(errors**2)))

    @classmethod
    def fit(
        cls,
        times: Sequence[float],
        zeros: Sequence[float],
        reference_date: date,
        *,
        seed: int = _SEED,
    ) -> Svensson:
        t_arr: np.ndarray = np.asarray(times, dtype=float)
        z_arr: np.ndarray = np.asarray(zeros, dtype=float)
        if len(t_arr) != len(z_arr):
            raise FitError(f"{len(t_arr)} times but {len(z_arr)} zero rates")
        if len(t_arr) < 6:
            raise FitError(f"Svensson has 6 parameters; {len(t_arr)} observations cannot fit it")

        def objective(p: np.ndarray) -> float:
            b0, b1, b2, b3, tau1, tau2 = (
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(p[5]),
            )
            if b0 + b1 <= _CONSTRAINT_FLOOR or tau2 - tau1 <= _MIN_TAU:
                return np.inf
            candidate = cls(
                beta=(b0, b1, b2, b3),
                tau=(tau1, tau2),
                reference_date=reference_date,
            )
            model = np.array([candidate.zero(float(x)) for x in t_arr])
            return float(np.sum((model - z_arr) ** 2))

        bounds = [
            (_CONSTRAINT_FLOOR, 0.25),
            (-0.25, 0.25),
            (-0.25, 0.25),
            (-0.25, 0.25),
            (_MIN_TAU, 30.0),
            (_MIN_TAU, 30.0),
        ]

        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            polish=True,
            strategy="best1bin",
            maxiter=5000,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
        )
        if not result.success and result.fun == np.inf:
            raise FitError("Differential evolution found no feasible Svensson fit")

        best: np.ndarray = np.asarray(result.x, dtype=float)
        return cls(
            beta=(float(best[0]), float(best[1]), float(best[2]), float(best[3])),
            tau=(float(best[4]), float(best[5])),
            reference_date=reference_date,
        )


@dataclass(frozen=True)
class NelsonSiegel:
    """The four-parameter Nelson-Siegel curve: Svensson without the second hump."""

    beta: tuple[float, float, float]
    tau: float
    reference_date: date

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise FitError(f"tau must be positive, got {self.tau}")

    def _as_svensson(self) -> Svensson:
        b0, b1, b2 = self.beta
        return Svensson(
            beta=(b0, b1, b2, 0.0),
            tau=(self.tau, self.tau * 2.0),
            reference_date=self.reference_date,
        )

    def zero(self, t: float) -> float:
        return self._as_svensson().zero(t)

    def df(self, t: float) -> float:
        return self._as_svensson().df(t)

    def fwd(self, t1: float, t2: float) -> float:
        return self._as_svensson().fwd(t1, t2)

    def rmse(self, times: Sequence[float], zeros: Sequence[float]) -> float:
        return self._as_svensson().rmse(times, zeros)

    @classmethod
    def fit(
        cls,
        times: Sequence[float],
        zeros: Sequence[float],
        reference_date: date,
        *,
        seed: int = _SEED,
    ) -> NelsonSiegel:
        t: np.ndarray = np.asarray(times, dtype=float)
        z: np.ndarray = np.asarray(zeros, dtype=float)
        if len(t) != len(z):
            raise FitError(f"{len(t)} times but {len(z)} zero rates")
        if len(t) < 4:
            raise FitError(f"Nelson-Siegel has 4 parameters; {len(t)} observations cannot fit it")

        def objective(p: np.ndarray) -> float:
            b0, b1, b2, tau = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            if b0 + b1 <= _CONSTRAINT_FLOOR:
                return np.inf
            candidate = cls(
                beta=(b0, b1, b2),
                tau=tau,
                reference_date=reference_date,
            )
            model = np.array([candidate.zero(float(x)) for x in t])
            return float(np.sum((model - z) ** 2))

        bounds = [
            (_CONSTRAINT_FLOOR, 0.25),
            (-0.25, 0.25),
            (-0.25, 0.25),
            (_MIN_TAU, 30.0),
        ]

        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            polish=True,
            strategy="best1bin",
            maxiter=5000,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
        )
        if not result.success and result.fun == np.inf:
            raise FitError("Differential evolution found no feasible NS fit")

        best: np.ndarray = np.asarray(result.x, dtype=float)
        return cls(
            beta=(float(best[0]), float(best[1]), float(best[2])),
            tau=float(best[3]),
            reference_date=reference_date,
        )
