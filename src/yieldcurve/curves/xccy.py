"""Cross-currency basis and collateral-currency (CSA) discounting.

Piterbarg (2010), "Funding beyond discounting: collateral agreements and
derivatives pricing", *Risk*, February 2010, established that the discount
curve for a derivative is a property of its *collateral agreement* (CSA), not
of the currency the trade is denominated in: daily variation margin is
remunerated at the overnight index of the collateral currency, so the value of
a collateralised position is carried at that currency's OIS rate. When the
collateral currency differs from the trade currency, the two OIS curves are
linked by the *cross-currency (XCCY) basis*, and this module implements that
link.

Notation (trade currency USD, collateral currency EUR):

- ``S_0``: spot FX, units of USD per 1 EUR;
- ``P_usd(t) = exp(-r_usd(t) * t)``: USD OIS (SOFR) discount factor;
- ``P_eur(t) = exp(-r_eur(t) * t)``: EUR OIS (ESTR) discount factor;
- ``b(t)``: the XCCY basis, the market's deviation from covered interest
  parity, defined through the FX forward

      F(t) = S_0 * P_eur(t) / P_usd(t) * exp(-b(t) * t).

  ``b = 0`` is pure CIP; ``b < 0`` (the long-standing EUR/USD sign, a "USD
  funding premium") means the EUR forward trades below its CIP level.

A USD cash flow of one unit at time ``t``, collateralised in EUR, is worth
today (its EUR amount ``1/X_t`` is funded at ESTR):

      df_usd^csa_eur(t) = P_usd(t) * exp(+b(t) * t),          (leading order)

or in zero rates,

      r_usd^csa_eur(t) = r_usd(t) - b(t).

Derivation: the EUR-collateral value of ``1/X_t`` EUR is ``P_eur(t)/F(t)``
(EUR), so in USD it is ``S_0 * P_eur(t) / F(t)``; substituting the FX forward
gives ``S_0 P_eur / (S_0 P_eur/P_usd * exp(-b t)) = P_usd exp(+b t)``. The
Siegel's-paradox convexity of ``1/X_t`` is second order in ``b`` and ignored.

The core no-arbitrage result is the identity that this pins: the XCCY basis is
exactly the difference between the own-currency OIS zero rate and the
collateral-currency-adjusted zero rate,

      b(t) = r_usd(t) - r_usd^csa_eur(t),

the quanto-style adjustment that converts "discount in the trade currency"
into "discount in the collateral currency".

The CIP-consistent *fair* spread on the EUR leg of a cross-currency basis swap
(what the basis would be if the FX forward obeyed pure OIS CIP) is a separate
quantity, the rate differential already priced by the forward:

      b_fair(t) = (P_eur(t) - P_usd(t)) / (t * P_eur(t)) ~ r_usd(t) - r_eur(t),

implemented by :func:`basis_from_zeros`. It is not the market basis ``b(t)``:
``b(t)`` is the residual dislocation on top of ``b_fair``.

The core closed forms are pinned against hand-derived references (derivations in
``tests/curves/test_xccy.py``). There is no QuantLib parity test: the market
basis is a synthetic *input* (a dislocation on top of CIP), so QuantLib's
cross-currency machinery — whose fair spread is zero under pure OIS curves
because its engine already applies CIP through the FX forward — does not
independently check the quantities this module computes. This is software
verification of an implementation, not empirical or regulatory model validation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import numpy.typing as npt

from yieldcurve.calendars import NullCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.interpolation import InterpMethod, InterpolatedDiscountCurve
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet, DiscountCurve
from yieldcurve.instruments import VanillaSwap
from yieldcurve.market.snapshot import Snapshot

_MONTH_ALIGNMENT_TOLERANCE = 1e-3
_MAX_TENOR_YEARS = 1000.0
_FLOAT_TENOR = "3M"


class XccyError(ValueError):
    """The cross-currency basis was constructed with inputs it cannot support."""


def _add_months(asof: date, months: int) -> date:
    """Calendar-month arithmetic, clamping an end-of-month anchor.

    Identical to ``yieldcurve.curves.build._add_months``; kept local so this
    module carries no private cross-module import.
    """
    total = asof.year * 12 + (asof.month - 1) + months
    year, zero_based_month = divmod(total, 12)
    month = zero_based_month + 1
    day = min(asof.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def _maturity(asof: date, years: float) -> date:
    """Maturity for a year tenor, by calendar months from ``asof``.

    Identical to ``yieldcurve.curves.build._maturity`` (CORE-01: integer years
    land on the anniversary, month-grid tenors on the calendar-month date,
    non-aligned tenors fall back to whole days).
    """
    if not math.isfinite(years) or years <= 0.0:
        raise XccyError(f"Invalid tenor in years: {years!r}")
    if years > _MAX_TENOR_YEARS:
        raise XccyError(
            f"Tenor {years!r} years exceeds the supported maximum of {_MAX_TENOR_YEARS:g} years"
        )
    months = round(years * 12.0)
    if abs(months / 12.0 - years) <= _MONTH_ALIGNMENT_TOLERANCE:
        return _add_months(asof, months)
    return asof + timedelta(days=round(years * 365.0))


@dataclass(frozen=True)
class BasisCurve:
    """A cross-currency basis spread interpolated over tenor.

    ``basis_bp`` is the annualised basis in basis points (e.g. -20.0 = -20 bp)
    quoted at each ``tenors`` node in years. ``basis(t)`` returns the decimal
    spread (bp / 1e4), linearly interpolated between nodes and held flat beyond
    the first and last node (an unobservable extrapolation, stated, not hidden).
    """

    tenors: tuple[float, ...]
    basis_bp: tuple[float, ...]

    _times: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _basis_bp: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.basis_bp):
            raise XccyError(f"{len(self.tenors)} tenors but {len(self.basis_bp)} basis quotes")
        if len(self.tenors) < 2:
            raise XccyError("A basis curve needs at least two tenor points")
        if any(not math.isfinite(t) for t in self.tenors):
            raise XccyError(f"Tenors must be finite: {self.tenors}")
        if any(not math.isfinite(b) for b in self.basis_bp):
            raise XccyError(f"Basis quotes must be finite: {self.basis_bp}")
        if any(t <= 0.0 for t in self.tenors):
            raise XccyError(f"Tenors must be positive: {self.tenors}")
        if any(later <= earlier for earlier, later in itertools.pairwise(self.tenors)):
            raise XccyError(f"Tenors must be strictly increasing: {self.tenors}")
        object.__setattr__(self, "_times", np.array(self.tenors, dtype=np.float64))
        object.__setattr__(self, "_basis_bp", np.array(self.basis_bp, dtype=np.float64))

    def basis_bp_at(self, t: float) -> float:
        """The basis in basis points at curve time ``t`` (years from as-of)."""
        self._check_time(t)
        return float(np.interp(t, self._times, self._basis_bp))

    def basis(self, t: float) -> float:
        """The basis as a decimal spread at curve time ``t`` (years from as-of)."""
        return self.basis_bp_at(t) / 1e4

    @staticmethod
    def _check_time(t: float) -> None:
        if not math.isfinite(t) or t < 0.0:
            raise XccyError(f"Curve time must be finite and non-negative, got {t}")


@dataclass(frozen=True)
class CsaDiscountCurve:
    """A discount curve shifted by a cross-currency basis.

    The collateral-currency CSA discount curve for a trade-currency cash flow:
    ``df(t) = base.df(t) * exp(+basis(t) * t)``, equivalently
    ``zero(t) = base.zero(t) - basis(t)``. ``base`` is the trade currency's OIS
    curve; the shift is the XCCY basis against the collateral currency.
    """

    base: DiscountCurve
    basis: BasisCurve

    @property
    def reference_date(self) -> date:
        return self.base.reference_date

    def df(self, t: float) -> float:
        return self.base.df(t) * math.exp(self.basis.basis(t) * t)

    def zero(self, t: float) -> float:
        return self.base.zero(t) - self.basis.basis(t)

    def fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError(f"t2 {t2} must exceed t1 {t1}")
        shift = (self.basis.basis(t2) * t2 - self.basis.basis(t1) * t1) / (t2 - t1)
        return self.base.fwd(t1, t2) - shift


def basis_from_zeros(usd_zero: float, eur_zero: float, tenor: float) -> float:
    """The CIP-consistent fair EUR-leg spread of a cross-currency basis swap.

    A single-period cross-currency basis swap (USD SOFR flat vs EUR ESTR + s),
    notional 1 EUR = S USD, is fair when ``S (1 - P_usd) = S[(1 - P_eur)
    + s * tenor * P_eur]``, i.e. ``s = (P_eur - P_usd) / (tenor * P_eur)`` with
    ``P = exp(-r * tenor)``. This is the rate differential the FX forward
    prices under pure CIP, distinct from the market basis ``BasisCurve``
    carries. Returns the decimal annualised spread.
    """
    if not all(math.isfinite(x) for x in (usd_zero, eur_zero, tenor)):
        raise XccyError(
            f"non-finite input: usd_zero={usd_zero}, eur_zero={eur_zero}, tenor={tenor}"
        )
    if tenor <= 0.0:
        raise XccyError(f"tenor must be positive, got {tenor}")
    p_usd = math.exp(-usd_zero * tenor)
    p_eur = math.exp(-eur_zero * tenor)
    return (p_eur - p_usd) / (tenor * p_eur)


def basis_between(curve_a: DiscountCurve, curve_b: DiscountCurve, t: float) -> float:
    """The annualised zero-rate spread ``curve_a.zero(t) - curve_b.zero(t)``.

    With ``curve_a`` the own-currency OIS curve and ``curve_b`` the
    collateral-adjusted curve, this equals the XCCY basis ``b(t)``: the basis is
    exactly the difference between the two discountings.
    """
    return curve_a.zero(t) - curve_b.zero(t)


def basis_curve_from_snapshot(snapshot: Snapshot, asof: date) -> BasisCurve:
    """The packaged illustrative XCCY basis curve (``illustrative_xccy_basis``).

    ``asof`` must equal the snapshot date; the basis tenors are years from it.
    """
    frame = snapshot.load("illustrative_xccy_basis").sort_values("tenor_years")
    tenors = tuple(float(t) for t in frame["tenor_years"])
    basis_bp = tuple(float(b) for b in frame["basis_bp"])
    curve = BasisCurve(tenors=tenors, basis_bp=basis_bp)
    if asof != snapshot.date:
        raise XccyError(f"basis as-of {asof} differs from snapshot date {snapshot.date}")
    return curve


def eur_discount_curve(
    snapshot: Snapshot,
    asof: date,
    *,
    method: InterpMethod = InterpMethod.LOG_LINEAR_DF,
) -> InterpolatedDiscountCurve:
    """The EUR discounting curve from the packaged ECB spot curve.

    The frozen snapshot's only EUR curve is the ECB AAA-rated euro-area
    *government* spot curve (``ecb_spot_curve``), continuously compounded; there
    is no ESTR OIS curve in the snapshot. It is used here as the EUR discounting
    proxy and labelled as such downstream, never as ESTR.
    """
    frame = snapshot.load("ecb_spot_curve").sort_values("tenor_years")
    tenors = tuple(float(t) for t in frame["tenor_years"])
    dfs = tuple(math.exp(-float(r) * t) for t, r in zip(tenors, frame["zero_rate"], strict=True))
    return InterpolatedDiscountCurve(reference_date=asof, times=tenors, dfs=dfs, method=method)


def usd_swap(
    asof: date,
    tenor_years: float,
    fixed_rate: float,
    *,
    notional: float = 1_000_000.0,
) -> VanillaSwap:
    """A spot-starting USD 3M vanilla swap in the repository's USD conventions.

    Annual fixed leg, 3M floating leg, ACT/360 on both legs, unadjusted dates on
    a no-holiday calendar — the same conventions ``build.usd_ois_quotes`` uses,
    so the instrument can be priced off ``build.usd_curveset``.
    """
    return VanillaSwap(
        start=asof,
        maturity=_maturity(asof, tenor_years),
        fixed_rate=fixed_rate,
        fixed_frequency=1,
        fixed_day_count=DayCount.ACT_360,
        float_tenor=_FLOAT_TENOR,
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        notional=notional,
    )


@dataclass(frozen=True)
class NpvDifference:
    """The value of one swap under two discounting conventions, in trade currency.

    ``base_npv`` discounts on the trade currency's own OIS curve (CSA in USD);
    ``csa_npv`` discounts on the collateral-currency-adjusted curve (CSA in
    EUR). ``delta`` is ``csa_npv - base_npv``: the CSA-currency effect.
    """

    base_npv: float
    csa_npv: float
    delta: float
    asof: date


def swap_npv_difference(
    swap: VanillaSwap,
    base: CurveSet,
    basis: BasisCurve,
    asof: date,
) -> NpvDifference:
    """Value ``swap`` on its own-currency curve and on the CSA-adjusted curve.

    Only the *discount* curve changes between the two valuations; the forecast
    curve (the floating leg's projection) is ``base``'s and stays put, because
    the floating index is still the trade currency's. That separation is the
    CSA-discounting point: the funding currency changes, the index does not.
    """
    csa = CurveSet(
        discount=CsaDiscountCurve(base.discount, basis),
        forecast=base.forecast,
    )
    base_npv = float(price(swap, base, asof).dirty)
    csa_npv = float(price(swap, csa, asof).dirty)
    return NpvDifference(base_npv=base_npv, csa_npv=csa_npv, delta=csa_npv - base_npv, asof=asof)
