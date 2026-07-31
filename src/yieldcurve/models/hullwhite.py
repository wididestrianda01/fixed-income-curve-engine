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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from yieldcurve.curves.protocol import CurveSet, DiscountCurve, curve_time
from yieldcurve.instruments import Swaption, VanillaSwap
from yieldcurve.market.snapshot import Snapshot

_FWD_STEP = 1e-5  # finite-difference epsilon for instantaneous forward, not 1-day step

_SMALL_A = 1e-8


class ModelError(ValueError):
    """The model was constructed with parameters it cannot support."""


class CalibrationError(ModelError):
    """Calibration failed or the provided instruments are inconsistent."""


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted parameters and the fit quality, in market units."""

    a: float
    sigma: float
    rmse_vol_bp: float
    n_instruments: int
    model_vols: tuple[float, ...]
    market_vols: tuple[float, ...]


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
        return (1.0 - math.exp(-self.a * gap)) / self.a

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
            variance_term = (
                self.sigma**2 / (4.0 * self.a) * (1.0 - math.exp(-2.0 * self.a * t)) * b**2
            )
        return ratio * math.exp(forward_term - variance_term)

    def zcb(self, t: float, T: float, r: float) -> float:  # noqa: N803
        """P(t,T) given the short rate r at time t."""
        return self.A(t, T) * math.exp(-self.B(t, T) * r)

    def _alpha(self, t: float) -> float:
        if self.a < _SMALL_A:
            return self.instantaneous_fwd(t) + 0.5 * self.sigma**2 * t**2
        decay = 1.0 - math.exp(-self.a * t)
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
        return self.sigma * math.sqrt((1.0 - math.exp(-2.0 * self.a * gap)) / (2.0 * self.a))

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

    def simulate_zcb(
        self,
        t: float,
        T: float,  # noqa: N803
        n_paths: int,
        *,
        seed: int,
    ) -> npt.NDArray[np.float64]:
        if t >= T:
            raise ValueError(f"simulate_zcb requires T > t, got t={t}, T={T}")
        steps = max(round((T - t) * 12), 1)
        grid = [t + (T - t) * k / steps for k in range(steps + 1)]
        if t == 0.0:
            paths = self.simulate(grid, n_paths, seed=seed)
        else:
            paths = self.simulate([0.0, *grid], n_paths, seed=seed)[:, 1:]
        integral = np.trapezoid(paths, x=np.asarray(grid), axis=1)
        result: npt.NDArray[np.float64] = np.exp(-integral)
        return result

    def zbo(self, expiry: float, maturity: float, strike: float, *, call: bool) -> float:
        if not 0.0 <= expiry <= maturity:
            raise ValueError(f"require 0 <= expiry <= maturity, got {expiry}, {maturity}")
        p_maturity = self.curve.df(maturity)
        p_expiry = self.curve.df(expiry)
        if expiry == 0.0 or self.sigma == 0.0:
            intrinsic = p_maturity - strike * p_expiry if call else strike * p_expiry - p_maturity
            return max(intrinsic, 0.0)
        if self.a < _SMALL_A:
            variance = self.sigma**2 * expiry
        else:
            variance = self.sigma**2 * (1.0 - math.exp(-2.0 * self.a * expiry)) / (2.0 * self.a)
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
        expiry = curve_time(asof, swaption.expiry)
        flows = swaption.swap.fixed_cashflows(asof)
        times = [curve_time(asof, flow.date) for flow in flows]
        notional = swaption.swap.notional
        amounts = [flow.amount / notional for flow in flows]
        if not times or times[0] < expiry - 1e-9:
            raise CalibrationError("Swaption expiry must not fall after the first fixed cash flow")
        amounts[-1] += 1.0  # terminal notional

        def swap_value(r: float) -> float:
            fixed = sum(
                a * self.zcb(expiry, tenor, r) for a, tenor in zip(amounts, times, strict=True)
            )
            return 1.0 - fixed

        r_star = float(brentq(swap_value, -0.50, 1.00, xtol=1e-14))
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
    if not swaptions:
        raise ValueError("At least one swaption is required")

    def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, sigma = float(params[0]), float(params[1])
        model = HullWhite(curve=curve, a=a, sigma=sigma)
        return np.array(
            [
                model.swaption_normal_vol(s, asof) - v
                for s, v in zip(swaptions, market_vols, strict=False)
            ]
        )

    fit = least_squares(
        residuals,
        x0=np.array(initial),
        bounds=(np.array([1e-4, 1e-5]), np.array([2.0, 0.20])),
        xtol=1e-14,
        ftol=1e-14,
    )
    a, sigma = float(fit.x[0]), float(fit.x[1])
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
    snapshot: Snapshot, asof: date, curve: DiscountCurve
) -> tuple[tuple[Swaption, ...], tuple[float, ...]]:
    data = snapshot.load("cme_swaption_vols")
    rows = [
        (
            date.fromisoformat(row["expiry"]),
            date.fromisoformat(row["maturity"]),
            float(row["vol"]),
        )
        for _, row in data.iterrows()
    ]
    return swaption_grid(rows, asof, curve)
