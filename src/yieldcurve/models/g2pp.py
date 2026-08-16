"""Two-factor Gaussian short-rate model (G2++).

Brigo & Mercurio, *Interest Rate Models*, section 4.2. The short rate is the
sum of two correlated Gaussian factors plus a deterministic shift:

    r(t) = x(t) + y(t) + phi(t)

    dx_t = -a x_t dt + sigma dW1_t,   x(0) = 0
    dy_t = -b y_t dt + eta   dW2_t,   y(0) = 0
    E[dW1 dW2] = rho dt

with ``a, b, sigma, eta > 0`` and ``rho in (-1, 1)``.

``phi`` is the deterministic shift that fits the initial discount curve
*exactly*, mirroring how the Hull-White ``theta`` does the same for one factor:

    phi(t) = f^M(0,t) + (sigma^2/(2a^2))(1 - e^{-at})^2
           + (eta^2/(2b^2))(1 - e^{-bt})^2
           + (rho sigma eta/(a b))(1 - e^{-at})(1 - e^{-bt})

where ``f^M(0,t)`` is the market instantaneous forward. The zero-coupon bond
price is exponential-affine in the two state variables:

    P(t,T) = A(t,T) exp(-B(a,t,T) x(t) - B(b,t,T) y(t)),
    B(z,t,T) = (1 - e^{-z(T-t)}) / z,

    A(t,T) = (P^M(0,T)/P^M(0,t)) exp(0.5 [V(t,T) - V(0,T) + V(0,t)])

with ``V(t,T)`` the Brigo-Mercurio eq (4.12) variance of the integral of the
two factors:

    V(t,T) = (sigma^2/a^2)[T-t + (2/a)e^{-a(T-t)} - (1/(2a))e^{-2a(T-t)} - 3/(2a)]
           + (eta^2/b^2)[T-t + (2/b)e^{-b(T-t)} - (1/(2b))e^{-2b(T-t)} - 3/(2b)]
           + 2 rho sigma eta/(a b)[ T-t - B(a,t,T) - B(b,t,T)
                                    + (1 - e^{-(a+b)(T-t)})/(a+b) ].

**phi is only evaluated for the short-rate path and its mean.** Bond prices
need only ``P^M(0,.)`` and ``V`` (no derivative of the curve), and exact
simulation needs only the Gaussian transition of ``(x, y)``; ``phi`` enters
only as ``x + y + phi`` for the short rate itself. This routes around ``phi``
exactly the way ``yieldcurve.models.hullwhite`` routes around ``theta``.

The key improvement over one-factor Hull-White is decorrelation: the
instantaneous forward rates ``f(t,T1)`` and ``f(t,T2)`` for ``T1 != T2`` have
correlation strictly below 1, whereas a one-factor model forces it to 1 (see
``docs/hull-white-limitations.md`` section 2). :meth:`forward_rate_correlation`
exposes this property.

Cross-checked against QuantLib's ``ql.G2`` (``discountBond`` and
``discountBondOption``) in ``tests/models/test_g2pp.py``. That parity is
software verification -- two implementations of the same closed form -- not
empirical or regulatory model validation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import norm

from yieldcurve.curves.protocol import DiscountCurve
from yieldcurve.models.bachelier import bachelier_vol

_FWD_STEP = 1e-5  # finite-difference epsilon for the instantaneous forward

# Calibration parameter bounds. The lower bounds are strictly positive, so a
# fit pinned to them is a boundary fit and is rejected, mirroring hullwhite.
_AB_BOUNDS = (1e-4, 2.0)
_VOL_BOUNDS = (1e-5, 0.20)

# Jacobian condition limit and start-sensitivity limit: the same thresholds as
# the Hull-White and SABR modules so every fit path hardens symmetrically.
_MAX_JACOBIAN_CONDITION = 1e12
_MAX_START_SENSITIVITY = 0.25

# Perturbed starting points for the start-sensitivity probe, one factor scaled
# at a time (a, sigma, b, eta).
_SENSITIVITY_STARTS = (
    (2.0, 1.0, 1.0, 1.0),
    (1.0, 2.0, 1.0, 1.0),
    (1.0, 1.0, 2.0, 1.0),
    (1.0, 1.0, 1.0, 2.0),
)

_FIT_TOL_TIGHT = 1e-14
_FIT_TOL_PROBE = 1e-10


class G2ppError(ValueError):
    """The model was constructed with parameters it cannot support."""


class CalibrationError(G2ppError):
    """Calibration failed or the provided caplet/volatility quotes are inconsistent."""


def _bond_sensitivity(z: float, gap: float) -> float:
    """``B(z, t, T) = (1 - e^{-z gap}) / z`` via ``-expm1`` for small ``z gap``."""
    return -math.expm1(-z * gap) / z


def _one_factor_variance(z: float, vol: float, gap: float) -> float:
    """``(vol^2/z^2)[gap + (2/z)e^{-z gap} - (1/(2z))e^{-2z gap} - 3/(2z)]``."""
    return (vol * vol / (z * z)) * (
        gap
        + (2.0 / z) * math.exp(-z * gap)
        - (1.0 / (2.0 * z)) * math.exp(-2.0 * z * gap)
        - 3.0 / (2.0 * z)
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted G2++ parameters and the fit quality, in market units.

    ``rho`` is a fixed input (caplet volatilities do not identify it); ``a``,
    ``sigma``, ``b`` and ``eta`` are the least-squares fit to a caplet
    normal-volatility surface. Diagnostics mirror
    ``yieldcurve.models.hullwhite.CalibrationResult``.
    """

    a: float
    sigma: float
    b: float
    eta: float
    rho: float
    rmse_vol_bp: float
    n_instruments: int
    model_vols: tuple[float, ...]
    market_vols: tuple[float, ...]
    success: bool
    active_bounds: tuple[bool, bool, bool, bool]
    jacobian_rank: int
    jacobian_condition: float
    residual_scale: float
    start_sensitivity: float


@dataclass(frozen=True)
class G2pp:
    """Two-factor Gaussian model fitted to ``curve`` with parameters ``(a, sigma, b, eta, rho)``.

    ``curve`` is any ``DiscountCurve``; the deterministic shift ``phi`` is
    constructed from it so the model reproduces ``curve.df(t)`` exactly at
    time 0 for every ``t``.
    """

    curve: DiscountCurve
    a: float
    sigma: float
    b: float
    eta: float
    rho: float

    def __post_init__(self) -> None:
        for name, value in (("a", self.a), ("sigma", self.sigma), ("b", self.b), ("eta", self.eta)):
            if not math.isfinite(value) or value <= 0.0:
                raise G2ppError(f"{name} must be positive and finite, got {value}")
        if not math.isfinite(self.rho) or not -1.0 < self.rho < 1.0:
            raise G2ppError(f"rho must be in (-1, 1), got {self.rho}")

    def instantaneous_fwd(self, t: float) -> float:
        """``f^M(0, t)``, the market instantaneous forward, via a finite difference."""
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        return self.curve.fwd(t, t + _FWD_STEP)

    def phi(self, t: float) -> float:
        """The deterministic shift ``phi(t)``, equal to the mean of ``r(t)``."""
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"t must be finite and non-negative, got {t}")
        ea = -math.expm1(-self.a * t)  # 1 - e^{-a t}
        eb = -math.expm1(-self.b * t)  # 1 - e^{-b t}
        return (
            self.instantaneous_fwd(t)
            + self.sigma**2 / (2.0 * self.a**2) * ea**2
            + self.eta**2 / (2.0 * self.b**2) * eb**2
            + self.rho * self.sigma * self.eta / (self.a * self.b) * ea * eb
        )

    def short_rate(self, t: float, x: float, y: float) -> float:
        """``r(t) = x + y + phi(t)`` at state ``(x, y)``."""
        return x + y + self.phi(t)

    def _variance(self, t: float, T: float) -> float:  # noqa: N803
        """The Brigo-Mercurio eq (4.12) variance ``V(t, T)``."""
        gap = T - t
        cross = (
            2.0
            * self.rho
            * self.sigma
            * self.eta
            / (self.a * self.b)
            * (
                gap
                - _bond_sensitivity(self.a, gap)
                - _bond_sensitivity(self.b, gap)
                + (1.0 - math.exp(-(self.a + self.b) * gap)) / (self.a + self.b)
            )
        )
        return (
            _one_factor_variance(self.a, self.sigma, gap)
            + _one_factor_variance(self.b, self.eta, gap)
            + cross
        )

    def _bond_factor(self, t: float, T: float) -> float:  # noqa: N803
        """The deterministic factor ``A(t, T)`` in ``P(t,T) = A exp(-B_a x - B_b y)``."""
        ratio = self.curve.df(T) / self.curve.df(t)
        return ratio * math.exp(
            0.5 * (self._variance(t, T) - self._variance(0.0, T) + self._variance(0.0, t))
        )

    def _check_interval(self, t: float, T: float) -> None:  # noqa: N803
        if not all(math.isfinite(v) for v in (t, T)) or t < 0.0 or t > T:
            raise ValueError(f"require 0 <= t <= T with finite inputs, got t={t}, T={T}")

    def discount_bond(self, t: float, T: float, x: float, y: float) -> float:  # noqa: N803
        """``P(t, T)`` given state ``(x, y)`` at time ``t``.

        At ``(t, x, y) = (0, 0, 0)`` this reproduces ``curve.df(T)`` exactly,
        which is the property that makes ``phi`` a curve fit rather than an
        approximation.
        """
        self._check_interval(t, T)
        if not math.isfinite(x) or not math.isfinite(y):
            raise G2ppError(f"state (x, y) must be finite, got ({x}, {y})")
        if t == T:
            return 1.0
        gap = T - t
        return self._bond_factor(t, T) * math.exp(
            -_bond_sensitivity(self.a, gap) * x - _bond_sensitivity(self.b, gap) * y
        )

    def _state_covariance(self, horizon: float) -> tuple[float, float, float]:
        """``(Var x, Var y, Cov(x, y))`` at ``horizon``, from ``x(0) = y(0) = 0``."""
        vx = self.sigma**2 * (-math.expm1(-2.0 * self.a * horizon)) / (2.0 * self.a)
        vy = self.eta**2 * (-math.expm1(-2.0 * self.b * horizon)) / (2.0 * self.b)
        cxy = (
            self.rho
            * self.sigma
            * self.eta
            * (-math.expm1(-(self.a + self.b) * horizon))
            / (self.a + self.b)
        )
        return vx, vy, cxy

    def state_covariance(self, t: float) -> npt.NDArray[np.float64]:
        """The 2x2 covariance matrix of ``(x(t), y(t))``, exact Gaussian law."""
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"t must be finite and non-negative, got {t}")
        vx, vy, cxy = self._state_covariance(t)
        return np.array([[vx, cxy], [cxy, vy]], dtype=np.float64)

    def _forward_variance(self, t: float, T: float) -> float:  # noqa: N803
        """Variance of the instantaneous forward ``f(t, T)`` at fixed time ``t``."""
        load_a = math.exp(-self.a * (T - t))
        load_b = math.exp(-self.b * (T - t))
        vx, vy, cxy = self._state_covariance(t)
        return load_a**2 * vx + load_b**2 * vy + 2.0 * load_a * load_b * cxy

    def _forward_covariance(self, t: float, T1: float, T2: float) -> float:  # noqa: N803
        load_a1, load_b1 = math.exp(-self.a * (T1 - t)), math.exp(-self.b * (T1 - t))
        load_a2, load_b2 = math.exp(-self.a * (T2 - t)), math.exp(-self.b * (T2 - t))
        vx, vy, cxy = self._state_covariance(t)
        return (
            load_a1 * load_a2 * vx
            + load_b1 * load_b2 * vy
            + (load_a1 * load_b2 + load_a2 * load_b1) * cxy
        )

    def forward_rate_correlation(self, t: float, T1: float, T2: float) -> float:  # noqa: N803
        """Correlation of the instantaneous forwards ``f(t, T1)`` and ``f(t, T2)``.

        ``f(t, T) = -d ln P(t,T)/dT`` is affine in ``(x(t), y(t))`` with
        loadings ``(e^{-a(T-t)}, e^{-b(T-t)})``, so its cross-sectional
        correlation across tenors has a closed form. It is strictly below 1
        for ``T1 != T2`` (and for ``t > 0``) whenever the model is genuinely
        two-factor -- ``a != b`` or ``sigma != eta``; a one-factor model forces
        it to exactly 1. ``t = 0`` is degenerate (the state is deterministic),
        so the correlation is defined for ``t > 0``.
        """
        if not all(math.isfinite(v) for v in (t, T1, T2)) or t < 0.0 or t > T1 or t > T2:
            raise ValueError(
                f"require 0 <= t <= T1, T2 with finite inputs, got t={t}, T1={T1}, T2={T2}"
            )
        if t == 0.0:
            raise ValueError("forward-rate correlation is undefined at t = 0 (deterministic state)")
        cov = self._forward_covariance(t, T1, T2)
        denom = math.sqrt(self._forward_variance(t, T1) * self._forward_variance(t, T2))
        return cov / denom

    def _increment_covariance(self, gap: float) -> npt.NDArray[np.float64]:
        """Covariance of the exact one-step ``(x, y)`` increment over ``gap``."""
        vx = self.sigma**2 * (-math.expm1(-2.0 * self.a * gap)) / (2.0 * self.a)
        vy = self.eta**2 * (-math.expm1(-2.0 * self.b * gap)) / (2.0 * self.b)
        cxy = (
            self.rho
            * self.sigma
            * self.eta
            * (-math.expm1(-(self.a + self.b) * gap))
            / (self.a + self.b)
        )
        return np.array([[vx, cxy], [cxy, vy]], dtype=np.float64)

    def _check_grid(self, times: Sequence[float]) -> list[float]:
        grid = [float(t) for t in times]
        if not grid:
            raise ValueError("times must be non-empty")
        if grid[0] != 0.0:
            raise ValueError(f"times must start at 0.0, got {grid[0]}")
        if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):  # noqa: RUF007
            raise ValueError(f"times must be strictly ascending, got {tuple(grid)}")
        return grid

    def simulate(
        self, times: Sequence[float], n_paths: int, *, seed: int
    ) -> npt.NDArray[np.float64]:
        """Exact joint Gaussian sample of ``(x, y)`` on ``times``.

        Returns ``(n_paths, len(times), 2)`` with ``[..., 0] = x`` and
        ``[..., 1] = y``. The OU transition is sampled exactly (Gaussian
        increment, no Euler-Maruyama), so the only error is Monte Carlo error.
        """
        grid = self._check_grid(times)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")

        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, len(grid), 2), dtype=np.float64)
        paths[:, 0, :] = 0.0
        for index in range(1, len(grid)):
            gap = grid[index] - grid[index - 1]
            drift = np.column_stack(
                (
                    paths[:, index - 1, 0] * math.exp(-self.a * gap),
                    paths[:, index - 1, 1] * math.exp(-self.b * gap),
                )
            )
            cholesky = np.linalg.cholesky(self._increment_covariance(gap))
            shocks = rng.standard_normal((n_paths, 2)) @ cholesky.T
            paths[:, index, :] = drift + shocks
        return paths

    def simulate_short_rate(
        self, times: Sequence[float], n_paths: int, *, seed: int
    ) -> npt.NDArray[np.float64]:
        """Short-rate paths ``r(t) = x(t) + y(t) + phi(t)`` on ``times``.

        Returns ``(n_paths, len(times))``. The mean of each column equals
        ``phi(t)`` (up to Monte Carlo error), which is the mean reversion the
        notebook demonstrates.
        """
        grid = self._check_grid(times)
        factors = self.simulate(grid, n_paths, seed=seed)
        shift = np.array([self.phi(t) for t in grid], dtype=np.float64)
        return factors[:, :, 0] + factors[:, :, 1] + shift

    def bond_option(self, expiry: float, maturity: float, strike: float, *, call: bool) -> float:
        """Value at t=0 of a European option on the zero-coupon bond ``P(·, maturity)``.

        The bond is lognormal in a single linear combination of ``(x, y)``, so
        the option has the closed form ``P(0,S) Phi(h) - K P(0,T) Phi(h - s)``
        with ``s^2`` the ``(x, y)``-loadings' variance. Expiry equal to maturity
        is degenerate (the bond is worth exactly 1 then).
        """
        if not all(math.isfinite(v) for v in (expiry, maturity, strike)):
            raise G2ppError(
                f"expiry, maturity and strike must be finite, got {expiry}, {maturity}, {strike}"
            )
        if not 0.0 <= expiry <= maturity:
            raise G2ppError(f"require 0 <= expiry <= maturity, got {expiry}, {maturity}")
        if strike <= 0.0:
            raise G2ppError(f"strike must be positive, got {strike}")
        if expiry == maturity:
            payoff = max(1.0 - strike, 0.0) if call else max(strike - 1.0, 0.0)
            return self.curve.df(maturity) * payoff
        if expiry == 0.0:
            intrinsic = (
                self.curve.df(maturity) - strike if call else strike - self.curve.df(maturity)
            )
            return max(intrinsic, 0.0)

        ba = _bond_sensitivity(self.a, maturity - expiry)
        bb = _bond_sensitivity(self.b, maturity - expiry)
        vx, vy, cxy = self._state_covariance(expiry)
        sigma_p = math.sqrt(ba * ba * vx + bb * bb * vy + 2.0 * ba * bb * cxy)
        p_maturity = self.curve.df(maturity)
        p_expiry = self.curve.df(expiry)
        h = math.log(p_maturity / (p_expiry * strike)) / sigma_p + 0.5 * sigma_p
        if call:
            return p_maturity * float(norm.cdf(h)) - strike * p_expiry * float(
                norm.cdf(h - sigma_p)
            )
        return strike * p_expiry * float(norm.cdf(-h + sigma_p)) - p_maturity * float(norm.cdf(-h))

    def caplet(self, expiry: float, tenor: float, strike: float) -> float:
        """Price at t=0 of a caplet on the forward rate resetting at ``expiry``.

        A caplet on ``L(T, T+tau)`` with strike ``K`` is a put on the bond
        ``P(T, T+tau)`` with strike ``1/(1 + tau K)``, scaled by ``1 + tau K``.
        The accrual fraction ``tau`` is taken equal to ``tenor`` in years (a
        documented simplification; an ACT/360 3M caplet has ``tau ~ 0.2535``).
        """
        if not all(math.isfinite(v) for v in (expiry, tenor, strike)):
            raise G2ppError(
                f"expiry, tenor and strike must be finite, got {expiry}, {tenor}, {strike}"
            )
        if expiry < 0.0 or tenor <= 0.0 or strike <= 0.0:
            raise G2ppError(
                f"require expiry >= 0, tenor > 0, strike > 0, got {expiry}, {tenor}, {strike}"
            )
        tau = tenor
        maturity = expiry + tenor
        bond_strike = 1.0 / (1.0 + tau * strike)
        return (1.0 + tau * strike) * self.bond_option(expiry, maturity, bond_strike, call=False)

    def caplet_normal_vol(self, expiry: float, tenor: float, strike: float) -> float:
        """Normal (Bachelier) volatility implied by the model caplet price."""
        tau = tenor
        maturity = expiry + tenor
        price = self.caplet(expiry, tenor, strike)
        forward = (self.curve.df(expiry) / self.curve.df(maturity) - 1.0) / tau
        undiscounted = price / (tau * self.curve.df(maturity))
        return bachelier_vol(undiscounted, forward, strike, expiry, pay=True)


def _check_rho(rho: float) -> None:
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        raise CalibrationError(f"rho must be in (-1, 1), got {rho}")


def calibrate(
    curve: DiscountCurve,
    quotes: Sequence[tuple[float, float, float, float]],
    *,
    rho: float = 0.0,
    initial: tuple[float, float, float, float] = (0.05, 0.010, 0.50, 0.020),
) -> CalibrationResult:
    """Fit ``(a, sigma, b, eta)`` to ATM caplet normal volatilities, ``rho`` fixed.

    ``quotes`` is a sequence of ``(expiry, tenor, strike, normal_vol)`` tuples
    with ``normal_vol`` a decimal volatility (bp / 1e4). The caplet's accrual
    fraction equals its tenor in years.

    Two identification boundaries are handled explicitly. First, ``rho`` is a
    fixed input: caplet volatilities identify the variance structure ``V(t, T)``
    but not the factor correlation, which only affects cross-tenor dependence
    (swaptions and spread options identify it; fitting those is outside this
    module). Second, caplet prices are invariant under the factor-label swap
    ``(a, sigma) <-> (b, eta)``, so the two mean-reversion speeds are only
    identified up to ordering; the fit imposes the convention ``a <= b`` (the
    factors are interchangeable, so this loses nothing). The optimizer works in
    ``(a, sigma, delta_b, eta)`` with ``b = a + delta_b`` and ``delta_b >= 0``,
    which removes the spurious second minimum the swap symmetry would otherwise
    leave for the start-sensitivity probe to trip over.

    A boundary fit, a rank-deficient or ill-conditioned Jacobian, or a
    start-sensitive solution is rejected, mirroring ``yieldcurve.models.hullwhite``.
    """
    _check_rho(rho)
    if len(quotes) < 4:
        raise CalibrationError(
            f"at least four caplet quotes are required to identify (a, sigma, b, eta), "
            f"got {len(quotes)}"
        )
    expiries: list[float] = []
    tenors: list[float] = []
    strikes: list[float] = []
    market: list[float] = []
    for index, (expiry, tenor, strike, vol) in enumerate(quotes):
        if not all(math.isfinite(v) for v in (expiry, tenor, strike, vol)):
            raise CalibrationError(
                f"caplet quote {index} is not finite: {(expiry, tenor, strike, vol)}"
            )
        if expiry < 0.0 or tenor <= 0.0 or strike <= 0.0:
            raise CalibrationError(
                f"caplet quote {index} has invalid (expiry, tenor, strike): "
                f"{(expiry, tenor, strike)}"
            )
        if vol < 0.0:
            raise CalibrationError(f"caplet quote {index} has a negative vol: {vol}")
        expiries.append(expiry)
        tenors.append(tenor)
        strikes.append(strike)
        market.append(vol)
    if not all(math.isfinite(v) for v in initial):
        raise CalibrationError(f"initial parameters must be finite, got {initial!r}")
    if not (
        _AB_BOUNDS[0] <= initial[0] <= _AB_BOUNDS[1]
        and _VOL_BOUNDS[0] <= initial[1] <= _VOL_BOUNDS[1]
        and _AB_BOUNDS[0] <= initial[2] <= _AB_BOUNDS[1]
        and _VOL_BOUNDS[0] <= initial[3] <= _VOL_BOUNDS[1]
    ):
        raise CalibrationError(f"initial parameters fall outside the bounds: {initial!r}")
    if initial[0] >= initial[2]:
        raise CalibrationError(
            f"the a <= b convention requires initial a < initial b, got {initial!r}"
        )

    # Internal parameter vector is (a, sigma, delta_b, eta) with b = a + delta_b.
    lo = np.array([_AB_BOUNDS[0], _VOL_BOUNDS[0], _AB_BOUNDS[0], _VOL_BOUNDS[0]])
    hi = np.array([_AB_BOUNDS[1], _VOL_BOUNDS[1], _AB_BOUNDS[1], _VOL_BOUNDS[1]])
    x0_internal = np.array(
        [initial[0], initial[1], initial[2] - initial[0], initial[3]], dtype=float
    )

    def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, sigma, delta_b, eta = (float(x) for x in params)
        model = G2pp(curve=curve, a=a, sigma=sigma, b=a + delta_b, eta=eta, rho=rho)
        return np.array(
            [
                model.caplet_normal_vol(expiry, tenor, strike) - mv
                for expiry, tenor, strike, mv in zip(
                    expiries, tenors, strikes, market, strict=False
                )
            ],
            dtype=np.float64,
        )

    def run(x0: npt.NDArray[np.float64], tolerance: float) -> OptimizeResult:
        return least_squares(
            residuals,
            x0=x0,
            bounds=(lo, hi),
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
        )

    fit = run(x0_internal, _FIT_TOL_TIGHT)
    if not fit.success:
        raise CalibrationError(
            f"calibration optimizer did not converge: {fit.message!r} ({len(quotes)} instruments)"
        )
    a, sigma, b, eta = float(fit.x[0]), float(fit.x[1]), float(fit.x[0] + fit.x[2]), float(fit.x[3])

    bound_tol = 1e-9 * np.maximum(1.0, np.abs(fit.x))
    active_bounds = (
        bool(fit.x[0] <= lo[0] + bound_tol[0] or fit.x[0] >= hi[0] - bound_tol[0]),
        bool(fit.x[1] <= lo[1] + bound_tol[1] or fit.x[1] >= hi[1] - bound_tol[1]),
        bool(fit.x[2] <= lo[2] + bound_tol[2] or fit.x[2] >= hi[2] - bound_tol[2]),
        bool(fit.x[3] <= lo[3] + bound_tol[3] or fit.x[3] >= hi[3] - bound_tol[3]),
    )
    if any(active_bounds):
        raise CalibrationError(
            f"calibration hit a parameter bound: active_bounds={active_bounds} at "
            f"a={a}, sigma={sigma}, b={b}, eta={eta}; the fit is on the boundary and is rejected"
        )

    jacobian = np.asarray(fit.jac, dtype=float)
    jacobian_rank = int(np.linalg.matrix_rank(jacobian))
    jacobian_condition = float(np.linalg.cond(jacobian))
    if jacobian_rank < 4:
        raise CalibrationError(
            f"calibration Jacobian is rank-deficient: rank={jacobian_rank} from "
            f"{len(quotes)} instruments; the quotes do not identify (a, sigma, b, eta)"
        )
    if not math.isfinite(jacobian_condition) or jacobian_condition > _MAX_JACOBIAN_CONDITION:
        raise CalibrationError(
            f"calibration Jacobian is ill-conditioned: condition {jacobian_condition:.3g} "
            f"from {len(quotes)} instruments; the quotes do not identify "
            "(a, sigma, b, eta) reliably"
        )

    sensitivity = 0.0
    for sa, ss, sd, se in _SENSITIVITY_STARTS:
        start = np.clip(x0_internal * np.array([sa, ss, sd, se], dtype=float), lo, hi)
        refit = run(start, _FIT_TOL_PROBE)
        if not refit.success:
            sensitivity = float("inf")
            break
        deviation = max(
            abs(float(refit.x[0]) - a) / max(abs(a), _AB_BOUNDS[0]),
            abs(float(refit.x[1]) - sigma) / max(abs(sigma), _VOL_BOUNDS[0]),
            abs(float(refit.x[2]) - (b - a)) / max(abs(b - a), _AB_BOUNDS[0]),
            abs(float(refit.x[3]) - eta) / max(abs(eta), _VOL_BOUNDS[0]),
        )
        sensitivity = max(sensitivity, deviation)
    if not math.isfinite(sensitivity) or sensitivity > _MAX_START_SENSITIVITY:
        raise CalibrationError(
            f"calibration is overly sensitive to its starting point: "
            f"start_sensitivity={sensitivity:.3f} (limit {_MAX_START_SENSITIVITY}); "
            f"the surface is not well identified"
        )

    model = G2pp(curve=curve, a=a, sigma=sigma, b=b, eta=eta, rho=rho)
    model_vols = tuple(
        model.caplet_normal_vol(expiry, tenor, strike)
        for expiry, tenor, strike in zip(expiries, tenors, strikes, strict=False)
    )
    errors = np.array(model_vols) - np.array(market)
    return CalibrationResult(
        a=a,
        sigma=sigma,
        b=b,
        eta=eta,
        rho=rho,
        rmse_vol_bp=float(np.sqrt((errors**2).mean()) * 1e4),
        n_instruments=len(quotes),
        model_vols=model_vols,
        market_vols=tuple(float(v) for v in market),
        success=True,
        active_bounds=active_bounds,
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        residual_scale=float(np.sqrt((errors**2).mean())),
        start_sensitivity=sensitivity,
    )
