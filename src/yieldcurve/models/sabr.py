"""SABR stochastic-volatility model (Hagan, Kumar, Lesniewski & Woodward 2002).

The SABR model is a CEV forward with stochastic volatility:

    dF_t      = alpha_t F_t^beta dW_t^1,   F(0) = F0
    dalpha_t  = nu alpha_t dW_t^2,          alpha(0) = alpha
    d<W^1, W^2>_t = rho dt

with ``alpha > 0`` the volatility level, ``beta in [0, 1]`` the CEV exponent
(``beta = 0`` normal, ``beta = 1`` lognormal), ``rho in (-1, 1)`` the
forward-vol correlation (the smile's skew), and ``nu >= 0`` the
volatility-of-volatility (the smile's curvature).

Two implied-volatility forms are implemented, both the Hagan et al. (2002)
*leading-order asymptotic* result -- an approximation, not an exact closed
form -- for arbitrary ``beta``:

- :func:`sabr_normal_vol` -- the implied *normal* (Bachelier) volatility,
  matching the repository's normal-volatility convention
  (``yieldcurve.models.bachelier``): swaptions are quoted in basis points and
  the forward and strike are rates, not prices.

- :func:`sabr_lognormal_vol` -- the implied *Black* (lognormal) volatility.

Both follow the same moneyness convention as QuantLib's ``sabrVolatility``
(``z = (nu / alpha) * (F K)^((1-beta)/2) * ln(F/K)``), and both are
cross-checked against it in ``tests/models/test_sabr.py``. That parity is
software verification of the implementation -- two implementations of the same
closed form -- not an empirical validation of the model against markets.

Convention inherited from QuantLib and stated for the record: the forward and
strike must be strictly positive, because the formula evaluates ``ln(F/K)`` and
``(F K)^(beta/2)``. A market whose rates can be non-positive is modelled with a
*shifted* SABR (a positive displacement added to forward and strike), which is
not implemented here; the normal (``beta = 0``) case is the coherent base model
for the low-rate regime this repository's EUR/SEK curves document.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, least_squares

# Calibration parameter bounds. The lower bounds are strictly positive (or, for
# rho, strictly interior), so a fit pinned to a bound is a boundary fit and is
# rejected. alpha and nu are decimal volatility units (bp / 1e4).
_ALPHA_BOUNDS = (1e-6, 0.20)
_NU_BOUNDS = (1e-6, 2.0)
_RHO_BOUNDS = (-0.9999, 0.9999)

# Jacobian condition limit: a full-rank but ill-conditioned Jacobian does not
# identify (alpha, rho, nu) reliably. Same threshold as the Hull-White and
# parametric modules so every fit path hardens symmetrically.
_MAX_JACOBIAN_CONDITION = 1e12

# Start-sensitivity limit: if refitting from perturbed starting points moves
# the solution by more than this relative amount, the smile is not well
# identified and the fit is rejected.
_MAX_START_SENSITIVITY = 0.25

# Perturbed starting points for the start-sensitivity probe (alpha, rho, nu).
_SENSITIVITY_STARTS = (
    (0.5, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (1.0, 0.5, 1.0),
    (1.0, 1.0, 0.5),
    (1.0, 1.0, 2.0),
)

# Optimizer tolerances: the initial fit is converged tightly; the sensitivity
# refits only probe the solution's location and run looser.
_FIT_TOL_TIGHT = 1e-14
_FIT_TOL_PROBE = 1e-10

# Threshold on z^2 below which z / x(z) is evaluated by its Taylor series.
# QuantLib's threshold (QL_EPSILON * 10); the series
# z / x(z) = 1 - (rho/2) z - ((3 rho^2 - 2)/12) z^2 + O(z^3) is exact at the
# raw-quotient's 0/0 point z = 0.
_SMALL_Z2 = 2.220446049250313e-15

# Near-ATM guard on ln(F/K): below this relative spread the log is evaluated by
# its second-order series ln(F/K) ~ eps - eps^2/2, eps = (F-K)/K, matching
# QuantLib's close() branch and avoiding cancellation.
_NEAR_ATM_REL = 1e-8


class SabrError(ValueError):
    """The model was constructed with parameters it cannot support."""


class CalibrationError(SabrError):
    """Calibration failed or the provided strike/volatility quotes are inconsistent."""


def _validate(
    forward: float,
    strike: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    expiry: float,
) -> None:
    if not all(math.isfinite(x) for x in (forward, strike, alpha, beta, rho, nu, expiry)):
        raise SabrError(
            f"non-finite input: forward={forward}, strike={strike}, alpha={alpha}, "
            f"beta={beta}, rho={rho}, nu={nu}, expiry={expiry}"
        )
    if forward <= 0.0:
        raise SabrError(f"forward must be positive, got {forward}")
    if strike <= 0.0:
        raise SabrError(f"strike must be positive, got {strike}")
    if alpha <= 0.0:
        raise SabrError(f"alpha must be positive, got {alpha}")
    if not 0.0 <= beta <= 1.0:
        raise SabrError(f"beta must be in [0, 1], got {beta}")
    if not -1.0 < rho < 1.0:
        raise SabrError(f"rho must be in (-1, 1), got {rho}")
    if nu < 0.0:
        raise SabrError(f"nu must be non-negative, got {nu}")
    if expiry < 0.0:
        raise SabrError(f"expiry must be non-negative, got {expiry}")


def _logm(forward: float, strike: float) -> float:
    """``ln(forward / strike)`` with the near-ATM series guard."""
    if forward == strike:
        return 0.0
    if abs(forward - strike) / strike < _NEAR_ATM_REL:
        eps = (forward - strike) / strike
        return eps - 0.5 * eps * eps
    return math.log(forward / strike)


def _smile_term(z: float, rho: float) -> float:
    """``z / x(z)``, ``x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho))``.

    Stable at ``z = 0`` via the Taylor series
    ``z / x(z) = 1 - (rho/2) z - ((3 rho^2 - 2)/12) z^2 + O(z^3)``.
    """
    if z * z < _SMALL_Z2:
        return 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    q = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    return z / math.log((q + z - rho) / (1.0 - rho))


def _moneyness(forward: float, strike: float, alpha: float, beta: float, nu: float) -> float:
    """``z = (nu / alpha) * (F K)^((1-beta)/2) * ln(F/K)`` (QuantLib's convention)."""
    logm = _logm(forward, strike)
    mid = math.pow(forward * strike, (1.0 - beta) / 2.0)
    return (nu / alpha) * mid * logm


def sabr_normal_vol(
    forward: float,
    strike: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    expiry: float,
) -> float:
    """Implied normal (Bachelier) volatility, Hagan et al. (2002), general ``beta``.

    Follows QuantLib's ``unsafeSabrNormalVolatility`` exactly, so the parity
    test is a tight equality. Returns a decimal volatility (bp / 1e4).
    """
    _validate(forward, strike, alpha, beta, rho, nu, expiry)

    one_minus_beta = 1.0 - beta
    fk = forward * strike
    a = math.pow(fk, one_minus_beta)  # (F K)^(1-beta)
    sqrt_a = math.sqrt(a)
    logm = _logm(forward, strike)

    z = (nu / alpha) * sqrt_a * logm
    smile = _smile_term(z, rho)

    # E = E1 / E2 with D = ln^2(F/K), C = (1-beta)^2 ln^2(F/K).
    d_sq = logm * logm
    c_sq = one_minus_beta * one_minus_beta * logm * logm
    e = (1.0 + d_sq / 24.0 + d_sq * d_sq / 1920.0) / (1.0 + c_sq / 24.0 + c_sq * c_sq / 1920.0)

    atm = 1.0 + expiry * (
        -beta * (2.0 - beta) * alpha * alpha / (24.0 * a)
        + 0.25 * rho * beta * nu * alpha / sqrt_a
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )

    level = alpha * math.pow(fk, beta / 2.0)  # alpha (F K)^(beta/2)
    return level * e * smile * atm


def sabr_lognormal_vol(
    forward: float,
    strike: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    expiry: float,
) -> float:
    """Implied Black (lognormal) volatility, Hagan et al. (2002), general ``beta``.

    Follows QuantLib's ``unsafeSabrLogNormalVolatility`` exactly, so the parity
    test is a tight equality. Returns a decimal volatility.
    """
    _validate(forward, strike, alpha, beta, rho, nu, expiry)

    one_minus_beta = 1.0 - beta
    fk = forward * strike
    a = math.pow(fk, one_minus_beta)  # (F K)^(1-beta)
    sqrt_a = math.sqrt(a)
    logm = _logm(forward, strike)

    z = (nu / alpha) * sqrt_a * logm
    smile = _smile_term(z, rho)

    c_sq = one_minus_beta * one_minus_beta * logm * logm
    denominator = sqrt_a * (1.0 + c_sq / 24.0 + c_sq * c_sq / 1920.0)

    atm = 1.0 + expiry * (
        one_minus_beta * one_minus_beta * alpha * alpha / (24.0 * a)
        + 0.25 * rho * beta * nu * alpha / sqrt_a
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )

    return (alpha / denominator) * smile * atm


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted SABR parameters and the fit quality, in market units.

    ``beta`` is fixed (not fitted) by the caller; ``alpha``, ``rho`` and ``nu``
    are the least-squares fit to a normal-volatility smile. Diagnostics are
    reported on every successful fit: the optimizer status, which parameters
    sit on a bound, the Jacobian rank and condition number, the residual scale
    (RMS residual in decimal vol units, equal to ``rmse_vol_bp / 1e4``), and
    the start sensitivity (largest relative move of the fitted parameters under
    perturbed starts).
    """

    alpha: float
    rho: float
    nu: float
    beta: float
    rmse_vol_bp: float
    n_strikes: int
    model_vols: tuple[float, ...]
    market_vols: tuple[float, ...]
    success: bool
    active_bounds: tuple[bool, bool, bool]
    jacobian_rank: int
    jacobian_condition: float
    residual_scale: float
    start_sensitivity: float


def _model_vols(
    forward: float,
    strikes: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    expiry: float,
) -> npt.NDArray[np.float64]:
    return np.array(
        [sabr_normal_vol(forward, float(k), alpha, beta, rho, nu, expiry) for k in strikes],
        dtype=np.float64,
    )


def calibrate(
    forward: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    expiry: float,
    *,
    beta: float = 0.0,
    alpha0: float = 0.0060,
    rho0: float = -0.20,
    nu0: float = 0.50,
    alpha_bounds: tuple[float, float] = _ALPHA_BOUNDS,
    rho_bounds: tuple[float, float] = _RHO_BOUNDS,
    nu_bounds: tuple[float, float] = _NU_BOUNDS,
) -> CalibrationResult:
    """Fit ``(alpha, rho, nu)`` to a market smile at a fixed ``beta``.

    ``strikes`` and ``market_vols`` are parallel sequences; ``market_vols`` are
    decimal *normal* volatilities (bp / 1e4), the repository's convention. A
    fit on a bound, a rank-deficient or ill-conditioned Jacobian, or a
    start-sensitive solution is rejected with :class:`CalibrationError`,
    mirroring ``yieldcurve.models.hullwhite``.
    """
    strikes_arr = np.array(list(strikes), dtype=np.float64)
    market = np.array(list(market_vols), dtype=np.float64)
    if strikes_arr.shape != market.shape or strikes_arr.ndim != 1:
        raise CalibrationError(
            f"strikes and market_vols must be equal-length 1-D sequences, got "
            f"{strikes_arr.shape} and {market.shape}"
        )
    if strikes_arr.size < 3:
        raise CalibrationError(
            f"a SABR fit needs at least three strike quotes to identify "
            f"(alpha, rho, nu), got {strikes_arr.size}"
        )
    if not 0.0 <= beta <= 1.0:
        raise SabrError(f"beta must be in [0, 1], got {beta}")
    if not math.isfinite(forward) or forward <= 0.0:
        raise CalibrationError(f"forward must be positive and finite, got {forward}")
    if expiry <= 0.0:
        raise CalibrationError(f"expiry must be positive, got {expiry}")

    def residual(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        alpha, rho, nu = (float(x) for x in params)
        return _model_vols(forward, strikes_arr, alpha, beta, rho, nu, expiry) - market

    lo = np.array([alpha_bounds[0], rho_bounds[0], nu_bounds[0]])
    hi = np.array([alpha_bounds[1], rho_bounds[1], nu_bounds[1]])
    bounds = (lo, hi)

    def fit(p0: tuple[float, float, float], tol: float) -> OptimizeResult:
        return least_squares(
            residual,
            x0=np.asarray(p0, dtype=float),
            bounds=bounds,
            xtol=tol,
            ftol=tol,
            gtol=tol,
        )

    result = fit((alpha0, rho0, nu0), _FIT_TOL_TIGHT)
    alpha, rho, nu = (float(x) for x in result.x)

    active_bounds = (
        bool(alpha <= alpha_bounds[0] * (1.0 + 1e-12) or alpha >= alpha_bounds[1] * (1.0 - 1e-12)),
        bool(rho <= rho_bounds[0] + 1e-9 or rho >= rho_bounds[1] - 1e-9),
        bool(nu <= nu_bounds[0] * (1.0 + 1e-12) or nu >= nu_bounds[1] * (1.0 - 1e-12)),
    )
    if any(active_bounds):
        raise CalibrationError(
            f"calibration hit a parameter bound: active_bounds={active_bounds} at "
            f"alpha={alpha}, rho={rho}, nu={nu}; the fit is on the boundary and is rejected"
        )

    jacobian = np.asarray(result.jac, dtype=float)
    jacobian_rank = int(np.linalg.matrix_rank(jacobian))
    jacobian_condition = float(np.linalg.cond(jacobian))
    if jacobian_rank < 3:
        raise CalibrationError(
            f"calibration Jacobian is rank-deficient: rank={jacobian_rank} from "
            f"{strikes_arr.size} strikes; the quotes do not identify (alpha, rho, nu)"
        )
    if not math.isfinite(jacobian_condition) or jacobian_condition > _MAX_JACOBIAN_CONDITION:
        raise CalibrationError(
            f"calibration Jacobian is ill-conditioned: cond={jacobian_condition}; "
            f"the quotes do not identify (alpha, rho, nu) reliably"
        )

    base = (alpha, rho, nu)
    sensitivity = 0.0
    for sa, sr, sn in _SENSITIVITY_STARTS:
        probe = fit((alpha0 * sa, rho0 * sr, nu0 * sn), _FIT_TOL_PROBE)
        pa, pr, pn = (float(x) for x in probe.x)
        for b, p, lb, ub in (
            (base[0], pa, alpha_bounds[0], alpha_bounds[1]),
            (base[1], pr, rho_bounds[0], rho_bounds[1]),
            (base[2], pn, nu_bounds[0], nu_bounds[1]),
        ):
            scale = (ub - lb) * 1e-3
            sensitivity = max(sensitivity, abs(p - b) / max(abs(b), scale))
    if sensitivity > _MAX_START_SENSITIVITY:
        raise CalibrationError(
            f"calibration is start-sensitive: relative move {sensitivity:.3f} exceeds "
            f"{_MAX_START_SENSITIVITY}; the smile is not well identified"
        )

    model_vols = _model_vols(forward, strikes_arr, alpha, beta, rho, nu, expiry)
    residual_scale = float(np.sqrt(np.mean((model_vols - market) ** 2)))
    rmse_vol_bp = residual_scale * 1e4

    return CalibrationResult(
        alpha=alpha,
        rho=rho,
        nu=nu,
        beta=beta,
        rmse_vol_bp=rmse_vol_bp,
        n_strikes=int(strikes_arr.size),
        model_vols=tuple(float(v) for v in model_vols),
        market_vols=tuple(float(v) for v in market),
        success=bool(result.success),
        active_bounds=active_bounds,
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        residual_scale=residual_scale,
        start_sensitivity=sensitivity,
    )
