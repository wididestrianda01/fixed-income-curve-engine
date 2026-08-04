"""Hull-White one-factor short-rate model.

    dr(t) = [theta(t) - a r(t)] dt + sigma dW(t)

theta(t) is determined by the initial curve, not chosen:

    theta(t) = df(0,t)/dt + a f(0,t) + (sigma^2 / 2a)(1 - e^{-2at})

which is what makes the model fit today's term structure exactly rather than
approximately, and is the reason it is used here in preference to Vasicek.

**theta is never evaluated.** It contains the derivative of the instantaneous
forward, which is the second derivative of the discount curve; on a
bootstrapped curve with monotone-convex interpolation the forward is only C0,
so theta is discontinuous at knots and any numerical estimate of it is noise.
The analytic bond price A(t,T) needs only P(0,.) and f(0,.) — first
derivatives — and exact simulation needs only the same. So the implementation
routes around theta entirely. The formula is documented above because the model
is not comprehensible without it, and implemented nowhere because it is not
needed.

The model binds to the DiscountCurve Protocol, never to a concrete curve class,
so it runs equally on a bootstrapped curve, a parametric fit, or a curve shifted
by ``yieldcurve.risk.scenarios``.

For limitations of this model (negative-rate probability, single-factor correlation,
missing volatility smile, calibration scope), see ``docs/hull-white-limitations.md``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, brentq, least_squares
from scipy.stats import norm

from yieldcurve.curves.protocol import CurveSet, DiscountCurve, curve_time
from yieldcurve.instruments import Swaption, VanillaSwap
from yieldcurve.market.snapshot import Snapshot

_FWD_STEP = 1e-5  # finite-difference epsilon for instantaneous forward, not 1-day step

_SMALL_A = 1e-8

# Calibration parameter bounds. The lower bounds are strictly positive, so a
# fit pinned to them is a boundary fit and is rejected.
_A_BOUNDS = (1e-4, 2.0)
_SIGMA_BOUNDS = (1e-5, 0.20)

# Start-sensitivity limit: if refitting from perturbed starting points moves
# the solution by more than this relative amount, the surface is not well
# identified and the fit is rejected.
_MAX_START_SENSITIVITY = 0.25
_SENSITIVITY_STARTS = ((0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0))

# Optimizer tolerances: the initial fit is converged tightly, while the
# start-sensitivity refits only probe the solution's location, so they run
# with a looser tolerance.
_FIT_TOL_TIGHT = 1e-14
_FIT_TOL_PROBE = 1e-10

# Initial bracket for the Jamshidian root and the cap on dynamic widening.
_JAMSHIDIAN_BRACKET = (-0.50, 1.00)
_JAMSHIDIAN_MAX_DOUBLINGS = 100

# Matched with re.fullmatch, so the pattern needs no ^...$ anchors.
_DATASET_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class ModelError(ValueError):
    """The model was constructed with parameters it cannot support."""


class CalibrationError(ModelError):
    """Calibration failed or the provided instruments are inconsistent."""


class SwaptionStrikeError(ModelError):
    """A swaption whose configured strike differs from its swap fixed rate."""


class DatasetNameError(ValueError):
    """A dataset name that could escape the snapshot directory was rejected."""


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted parameters and the fit quality, in market units.

    Diagnostics are reported on every successful fit: the optimizer status,
    whether either parameter sits on a bound, the Jacobian rank and condition
    number, the residual scale (RMS residual in decimal vol units, equal to
    ``rmse_vol_bp / 1e4``), and the start sensitivity (largest relative move of
    the fitted parameters when the starting point is perturbed).
    """

    a: float
    sigma: float
    rmse_vol_bp: float
    n_instruments: int
    model_vols: tuple[float, ...]
    market_vols: tuple[float, ...]
    success: bool
    active_bounds: tuple[bool, bool]
    jacobian_rank: int
    jacobian_condition: float
    residual_scale: float
    start_sensitivity: float


@dataclass(frozen=True)
class HullWhite:
    """Hull-White 1F fitted to ``curve`` with constant ``a`` and ``sigma``."""

    curve: DiscountCurve
    a: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ModelError(f"sigma must be non-negative, got {self.sigma}")
        if self.a < 0.0:
            raise ModelError(
                f"mean reversion a must be non-negative, got {self.a}; "
                "a negative value makes the short rate explosive"
            )

    @property
    def r0(self) -> float:
        return self.instantaneous_fwd(0.0)

    def instantaneous_fwd(self, t: float) -> float:
        """f(0,t), derived from the Protocol's continuously compounded forward."""
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        return self.curve.fwd(t, t + _FWD_STEP)

    def B(self, t: float, T: float) -> float:  # noqa: N802, N803
        """(1 - e^{-a(T-t)}) / a, the sensitivity of ln P(t,T) to r(t)."""
        if t > T:
            raise ValueError(f"B requires T >= t, got t={t}, T={T}")
        gap = T - t
        if self.a < _SMALL_A:
            return gap
        # -expm1(-x) avoids the catastrophic cancellation in (1 - exp(-x))
        # when a*gap is tiny, keeping B == gap to machine precision.
        return -math.expm1(-self.a * gap) / self.a

    def A(self, t: float, T: float) -> float:  # noqa: N802, N803
        """The deterministic factor in P(t,T) = A(t,T) exp(-B(t,T) r(t))."""
        if t > T:
            raise ValueError(f"A requires T >= t, got t={t}, T={T}")
        b = self.B(t, T)
        ratio = self.curve.df(T) / self.curve.df(t)
        forward_term = b * self.instantaneous_fwd(t)
        if self.a < _SMALL_A:
            variance_term = 0.5 * self.sigma**2 * t * b**2
        else:
            variance_term = self.sigma**2 / (4.0 * self.a) * (-math.expm1(-2.0 * self.a * t)) * b**2
        return ratio * math.exp(forward_term - variance_term)

    def zcb(self, t: float, T: float, r: float) -> float:  # noqa: N803
        """P(t,T) given the short rate r at time t."""
        return self.A(t, T) * math.exp(-self.B(t, T) * r)

    def _alpha(self, t: float) -> float:
        if self.a < _SMALL_A:
            return self.instantaneous_fwd(t) + 0.5 * self.sigma**2 * t**2
        decay = -math.expm1(-self.a * t)
        return self.instantaneous_fwd(t) + self.sigma**2 / (2.0 * self.a**2) * decay**2

    def conditional_mean(self, t: float, s: float, r: float) -> float:
        if s < t:
            raise ValueError(f"conditional_mean requires s >= t, got t={t}, s={s}")
        decay = math.exp(-self.a * (s - t))
        return r * decay + self._alpha(s) - self._alpha(t) * decay

    def conditional_sd(self, t: float, s: float) -> float:
        if s < t:
            raise ValueError(f"conditional_sd requires s >= t, got t={t}, s={s}")
        gap = s - t
        if self.a < _SMALL_A:
            return self.sigma * math.sqrt(gap)
        return self.sigma * math.sqrt((-math.expm1(-2.0 * self.a * gap)) / (2.0 * self.a))

    def simulate(
        self, times: Sequence[float], n_paths: int, *, seed: int
    ) -> npt.NDArray[np.float64]:
        grid = [float(t) for t in times]
        if not grid:
            raise ValueError("times must be non-empty")
        if grid[0] != 0.0:
            raise ValueError(f"times must start at 0.0, got {grid[0]}")
        if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):  # noqa: RUF007
            raise ValueError(f"times must be strictly ascending, got {tuple(grid)}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")

        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, len(grid)), dtype=np.float64)
        paths[:, 0] = self.r0
        for index in range(1, len(grid)):
            t, s = grid[index - 1], grid[index]
            decay = math.exp(-self.a * (s - t))
            drift = self._alpha(s) - self._alpha(t) * decay
            sd = self.conditional_sd(t, s)
            paths[:, index] = (
                paths[:, index - 1] * decay + drift + sd * rng.standard_normal(n_paths)
            )
        return paths

    def simulate_path_discount_factors(
        self,
        t: float,
        T: float,  # noqa: N803
        n_paths: int,
        *,
        seed: int,
        steps: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Per-path discount factors exp(-∫_t^T r(s) ds) by trapezoid quadrature.

        This is a *path discount-factor approximation*, not exact zero-coupon
        bond simulation: the trapezoid rule on a monthly grid (``steps=None``,
        or an explicit ``steps`` subdivision count) carries an O(step^2)
        time-step bias relative to the exact bond price. ``steps`` exists so
        that the time-step bias can be tested separately from Monte Carlo
        error.
        """
        if t >= T:
            raise ValueError(f"simulate_path_discount_factors requires T > t, got t={t}, T={T}")
        if steps is None:
            steps = max(round((T - t) * 12), 1)
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        grid = [t + (T - t) * k / steps for k in range(steps + 1)]
        if t == 0.0:
            paths = self.simulate(grid, n_paths, seed=seed)
        else:
            paths = self.simulate([0.0, *grid], n_paths, seed=seed)[:, 1:]
        integral = np.trapezoid(paths, x=np.asarray(grid), axis=1)
        result: npt.NDArray[np.float64] = np.exp(-integral)
        return result

    def zbo(self, expiry: float, maturity: float, strike: float, *, call: bool) -> float:
        """Value at t=0 of a European option on the zero-coupon bond P(·, maturity).

        The bond is lognormal in the short rate, so the option has the closed
        Jamshidian form below. Expiry equal to maturity is degenerate — the
        bond is worth exactly 1 then — and is handled directly; strikes must be
        positive.
        """
        if not math.isfinite(expiry) or not math.isfinite(maturity) or not math.isfinite(strike):
            raise ModelError(
                f"expiry, maturity and strike must be finite, got {expiry}, {maturity}, {strike}"
            )
        if not 0.0 <= expiry <= maturity:
            raise ModelError(f"require 0 <= expiry <= maturity, got {expiry}, {maturity}")
        if strike <= 0.0:
            raise ModelError(f"strike must be positive, got {strike}")
        if expiry == maturity:
            # P(T, T) == 1 deterministically, so only the intrinsic payoff remains
            payoff = max(1.0 - strike, 0.0) if call else max(strike - 1.0, 0.0)
            return self.curve.df(maturity) * payoff
        p_maturity = self.curve.df(maturity)
        p_expiry = self.curve.df(expiry)
        if expiry == 0.0 or self.sigma == 0.0:
            intrinsic = p_maturity - strike * p_expiry if call else strike * p_expiry - p_maturity
            return max(intrinsic, 0.0)
        if self.a < _SMALL_A:
            variance = self.sigma**2 * expiry
        else:
            # -expm1 keeps the variance non-zero (and the ratio below finite)
            # for tiny expiries, instead of underflowing to a division by zero.
            variance = self.sigma**2 * (-math.expm1(-2.0 * self.a * expiry)) / (2.0 * self.a)
        sigma_p = math.sqrt(variance) * self.B(expiry, maturity)
        h = math.log(p_maturity / (p_expiry * strike)) / sigma_p + 0.5 * sigma_p
        if call:
            return p_maturity * float(norm.cdf(h)) - strike * p_expiry * float(
                norm.cdf(h - sigma_p)
            )
        return strike * p_expiry * float(norm.cdf(-h + sigma_p)) - p_maturity * float(norm.cdf(-h))

    def forward_swap_value(self, swap: VanillaSwap, t: float, r: float, asof: date) -> float:
        """Value of ``swap`` at time ``t`` in state ``r``, seen from the fixed-rate payer."""
        flows = swap.fixed_cashflows(asof)
        times = [curve_time(asof, flow.date) for flow in flows]
        notional = swap.notional
        amounts = [flow.amount / notional for flow in flows]
        amounts[-1] += 1.0
        fixed = sum(a * self.zcb(t, tenor, r) for a, tenor in zip(amounts, times, strict=True))
        return notional * (1.0 - fixed)

    def swaption(self, swaption: Swaption, asof: date) -> float:
        """Jamshidian decomposition of ``swaption`` into zero-coupon bond options.

        ``swaption.strike`` is the single canonical strike source: a swaption
        whose configured strike differs from its swap's fixed rate is rejected
        with both values and the trade context. The decomposition also requires
        non-negative cash-flow coefficients (negative coupons are unsupported).
        """
        expiry = curve_time(asof, swaption.expiry)
        flows = swaption.swap.fixed_cashflows(asof)
        times = [curve_time(asof, flow.date) for flow in flows]
        notional = swaption.swap.notional
        amounts = [flow.amount / notional for flow in flows]
        if not times or times[0] < expiry - 1e-9:
            raise ModelError(
                f"swaption expiry {swaption.expiry} falls at or after the first fixed cash "
                f"flow of swap {swaption.swap.start}->{swaption.swap.maturity}; unsupported trade"
            )
        if not math.isclose(swaption.strike, swaption.swap.fixed_rate, rel_tol=1e-9, abs_tol=1e-12):
            raise SwaptionStrikeError(
                f"swaption strike {swaption.strike} differs from its swap fixed rate "
                f"{swaption.swap.fixed_rate}; expiry={swaption.expiry}, "
                f"swap {swaption.swap.start}->{swaption.swap.maturity}, "
                f"notional={notional}, pay_fixed={swaption.pay_fixed}"
            )
        amounts[-1] += 1.0  # terminal notional
        if any(amount < 0.0 for amount in amounts):
            raise ModelError(
                f"Jamshidian decomposition requires non-negative cash-flow coefficients; "
                f"swap {swaption.swap.start}->{swaption.swap.maturity} with fixed rate "
                f"{swaption.swap.fixed_rate} gives {tuple(amounts)}; unsupported trade "
                f"(expiry={swaption.expiry}, notional={notional}, pay_fixed={swaption.pay_fixed})"
            )

        def swap_value(r: float) -> float:
            fixed = sum(
                a * self.zcb(expiry, tenor, r) for a, tenor in zip(amounts, times, strict=True)
            )
            return 1.0 - fixed

        r_star = _jamshidian_root(swap_value, swaption)
        strikes = [self.zcb(expiry, tenor, r_star) for tenor in times]

        call = not swaption.pay_fixed
        return notional * sum(
            a * self.zbo(expiry, tenor, k, call=call)
            for a, tenor, k in zip(amounts, times, strikes, strict=True)
        )

    def swaption_normal_vol(self, swaption: Swaption, asof: date) -> float:
        from yieldcurve.curves.pricing import annuity, par_rate
        from yieldcurve.models.bachelier import bachelier_vol

        curves = CurveSet.single(self.curve)
        forward = par_rate(swaption.swap, curves, asof)
        expiry = curve_time(asof, swaption.expiry)
        undiscounted = self.swaption(swaption, asof) / (
            swaption.swap.notional * annuity(swaption.swap, curves, asof)
        )
        return bachelier_vol(undiscounted, forward, swaption.strike, expiry, pay=swaption.pay_fixed)


def _jamshidian_root(swap_value: Callable[[float], float], swaption: Swaption) -> float:
    """Solve swap_value(r) = 0 with robust dynamic bracketing.

    For non-negative cash-flow coefficients the swap value is monotone
    (weakly) increasing in the short rate: strictly so except in the
    degenerate case where a cash flow falls exactly at the option expiry,
    whose zero-coupon bond has B = 0 and is therefore constant in r. So when
    both bracket endpoints share a sign the root lies beyond the endpoint
    with the smaller-magnitude value. The bracket is widened by doubling
    until a sign change appears, capped at ``_JAMSHIDIAN_MAX_DOUBLINGS``
    widenings.
    """
    lo, hi = _JAMSHIDIAN_BRACKET
    f_lo, f_hi = swap_value(lo), swap_value(hi)
    for _ in range(_JAMSHIDIAN_MAX_DOUBLINGS):
        if f_lo * f_hi <= 0.0:
            return float(brentq(swap_value, lo, hi, xtol=1e-14))
        if f_lo < 0.0:
            hi += hi - lo
            f_hi = swap_value(hi)
        else:
            lo -= hi - lo
            f_lo = swap_value(lo)
    raise ModelError(
        f"no Jamshidian root found for swaption expiry={swaption.expiry}, "
        f"swap {swaption.swap.start}->{swaption.swap.maturity}, strike={swaption.strike}, "
        f"after {_JAMSHIDIAN_MAX_DOUBLINGS} bracket doublings"
    )


def calibrate(
    curve: DiscountCurve,
    swaptions: Sequence[Swaption],
    market_vols: Sequence[float],
    asof: date,
    *,
    initial: tuple[float, float] = (0.05, 0.01),
) -> CalibrationResult:
    if len(swaptions) != len(market_vols):
        raise ValueError(
            f"swaptions and market_vols must be the same length, got "
            f"{len(swaptions)} and {len(market_vols)}"
        )
    if len(swaptions) < 2:
        raise CalibrationError(
            f"at least two independent swaptions are required to identify (a, sigma), "
            f"got {len(swaptions)}"
        )
    for index, vol in enumerate(market_vols):
        if not math.isfinite(vol):
            raise CalibrationError(f"market vol {index} is not finite: {vol!r}")
        if vol < 0.0:
            raise CalibrationError(f"market vol {index} is negative: {vol!r}")
    if not (math.isfinite(initial[0]) and math.isfinite(initial[1])):
        raise CalibrationError(f"initial parameters must be finite, got {initial!r}")
    if not (
        _A_BOUNDS[0] <= initial[0] <= _A_BOUNDS[1]
        and _SIGMA_BOUNDS[0] <= initial[1] <= _SIGMA_BOUNDS[1]
    ):
        raise CalibrationError(f"initial parameters fall outside the bounds: {initial!r}")

    lo = np.array([_A_BOUNDS[0], _SIGMA_BOUNDS[0]])
    hi = np.array([_A_BOUNDS[1], _SIGMA_BOUNDS[1]])

    def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, sigma = float(params[0]), float(params[1])
        model = HullWhite(curve=curve, a=a, sigma=sigma)
        return np.array(
            [
                model.swaption_normal_vol(s, asof) - v
                for s, v in zip(swaptions, market_vols, strict=False)
            ]
        )

    def run(x0: tuple[float, float], tolerance: float) -> OptimizeResult:
        return least_squares(
            residuals,
            x0=np.asarray(x0, dtype=float),
            bounds=(lo, hi),
            xtol=tolerance,
            ftol=tolerance,
        )

    fit = run(initial, _FIT_TOL_TIGHT)
    if not fit.success:
        raise CalibrationError(
            f"calibration optimizer did not converge: {fit.message!r} "
            f"({len(swaptions)} instruments)"
        )
    a, sigma = float(fit.x[0]), float(fit.x[1])

    bound_tol = 1e-9 * np.maximum(1.0, np.abs(fit.x))
    active_bounds = (
        bool(fit.x[0] <= _A_BOUNDS[0] + bound_tol[0] or fit.x[0] >= _A_BOUNDS[1] - bound_tol[0]),
        bool(
            fit.x[1] <= _SIGMA_BOUNDS[0] + bound_tol[1]
            or fit.x[1] >= _SIGMA_BOUNDS[1] - bound_tol[1]
        ),
    )
    if any(active_bounds):
        raise CalibrationError(
            f"calibration hit a parameter bound: active_bounds={active_bounds} "
            f"at a={a}, sigma={sigma}; the fit is on the boundary and is rejected"
        )

    jacobian = np.asarray(fit.jac, dtype=float)
    jacobian_rank = int(np.linalg.matrix_rank(jacobian))
    jacobian_condition = float(np.linalg.cond(jacobian))
    if jacobian_rank < 2:
        raise CalibrationError(
            f"calibration Jacobian is rank-deficient: rank={jacobian_rank} from "
            f"{len(swaptions)} instruments; the quotes do not identify (a, sigma)"
        )

    final_residuals = np.asarray(fit.fun, dtype=float)
    residual_scale = float(np.sqrt((final_residuals**2).mean()))

    # Start sensitivity: refit from perturbed starting points. A solution that
    # moves materially with the start is not identified.
    sensitivity = 0.0
    for factor_a, factor_sigma in _SENSITIVITY_STARTS:
        start = np.clip(
            np.asarray([initial[0] * factor_a, initial[1] * factor_sigma], dtype=float),
            lo,
            hi,
        )
        refit = run((float(start[0]), float(start[1])), _FIT_TOL_PROBE)
        if not refit.success:
            sensitivity = float("inf")
            break
        deviation = max(
            abs(float(refit.x[0]) - a) / max(abs(a), _A_BOUNDS[0]),
            abs(float(refit.x[1]) - sigma) / max(abs(sigma), _SIGMA_BOUNDS[0]),
        )
        sensitivity = max(sensitivity, deviation)
    if not math.isfinite(sensitivity) or sensitivity > _MAX_START_SENSITIVITY:
        raise CalibrationError(
            f"calibration is overly sensitive to its starting point: "
            f"start_sensitivity={sensitivity:.3f} (limit {_MAX_START_SENSITIVITY}); "
            f"the surface is not well identified"
        )

    model = HullWhite(curve=curve, a=a, sigma=sigma)
    model_vols = tuple(model.swaption_normal_vol(s, asof) for s in swaptions)
    errors = np.array(model_vols) - np.array(market_vols)
    return CalibrationResult(
        a=a,
        sigma=sigma,
        rmse_vol_bp=float(np.sqrt((errors**2).mean()) * 1e4),
        n_instruments=len(swaptions),
        model_vols=model_vols,
        market_vols=tuple(float(v) for v in market_vols),
        success=True,
        active_bounds=active_bounds,
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        residual_scale=residual_scale,
        start_sensitivity=sensitivity,
    )


def swaption_grid(
    rows: Sequence[tuple[date, date, float]], asof: date, curve: DiscountCurve
) -> tuple[tuple[Swaption, ...], tuple[float, ...]]:
    """Build ATM payer swaptions from (expiry, maturity, normal vol in bp) rows."""
    from yieldcurve.calendars import USGovernmentBondCalendar
    from yieldcurve.conventions import BusinessDayConvention, DayCount
    from yieldcurve.curves.pricing import par_rate
    from yieldcurve.instruments import Swaption, VanillaSwap

    calendar = USGovernmentBondCalendar()
    bdc = BusinessDayConvention.MODIFIED_FOLLOWING

    curves = CurveSet.single(curve)

    swaptions: list[Swaption] = []
    vols: list[float] = []

    for expiry_date, maturity_date, vol_bp in rows:
        swap = VanillaSwap(
            start=expiry_date,
            maturity=maturity_date,
            fixed_rate=0.0,
            fixed_frequency=2,
            fixed_day_count=DayCount.THIRTY_360_BOND,
            float_tenor="3M",
            float_day_count=DayCount.ACT_360,
            calendar=calendar,
            bdc=bdc,
            notional=1.0,
        )
        strike = par_rate(swap, curves, asof)

        swap = VanillaSwap(
            start=expiry_date,
            maturity=maturity_date,
            fixed_rate=strike,
            fixed_frequency=2,
            fixed_day_count=DayCount.THIRTY_360_BOND,
            float_tenor="3M",
            float_day_count=DayCount.ACT_360,
            calendar=calendar,
            bdc=bdc,
            notional=1.0,
        )
        swaption = Swaption(expiry=expiry_date, swap=swap, strike=strike, pay_fixed=True)
        swaptions.append(swaption)
        vols.append(vol_bp / 1e4)

    return tuple(swaptions), tuple(vols)


def atm_swaption_grid(
    snapshot: Snapshot,
    asof: date,
    curve: DiscountCurve,
    *,
    dataset: str = "cme_swaption_vols",
) -> tuple[tuple[Swaption, ...], tuple[float, ...]]:
    """ATM payer swaptions and their normal vols from a snapshot dataset.

    The default is the licensed CME cleared-swaption settlement grid, which this
    repository does not ship. Pass dataset="illustrative_swaption_vols" for the
    constructed grid that is committed here.

    The dataset name must match a strict identifier grammar: a name containing a
    path separator would escape the snapshot directory.
    """
    if not _DATASET_NAME_RE.fullmatch(dataset):
        raise DatasetNameError(
            f"dataset name {dataset!r} does not match the strict identifier grammar "
            r"[a-z][a-z0-9_]* and could escape the snapshot directory"
        )
    data = snapshot.load(dataset)
    rows = [
        (
            date.fromisoformat(row["expiry"]),
            date.fromisoformat(row["maturity"]),
            float(row["vol"]),
        )
        for _, row in data.iterrows()
    ]
    return swaption_grid(rows, asof, curve)
