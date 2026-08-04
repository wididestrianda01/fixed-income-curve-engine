"""Par-rate delta ladder: risk in the coordinates the market quotes.

Key-rate duration bumps *zero* rates, which is the natural basis for describing
a curve but not one anybody can trade. This bumps each *quoted instrument* by a
basis point, rebootstraps the whole curve, and reprices — so each number answers
"how much of that instrument hedges this position", which is the report a swaps
desk actually runs against its book.

The two ladders do not agree entry by entry and are not supposed to. A bump to
the 5y par quote moves every zero out to 5y, not just the 5y zero, so par delta
spreads where KRD localises.

Nor do they agree in total. A uniform 1bp bump of every *quote* is not a uniform
1bp bump of the *zero* curve: the par-to-zero map has a Jacobian that is only
the identity for a flat curve quoted on curve basis, and the quotes here are
ACT/360 while curve time is ACT/365F. On the OIS set in the tests the ladder
totals 97-99% of DV01, and the residual is that Jacobian, not an error.

A caveat on ``method``. The ladder is additive — the entries sum to the effect
of bumping every quote at once — only if the curve depends smoothly on the
quotes. Log-linear and cubic-log-DF satisfy that to 1e-4 relative. Monotone
convex does not: its amendment tests are branches on which region a forward
falls into, so a 1bp bump can flip a region and reshape the curve between
knots, and additivity breaks by around 1.4% on the test quote set. The default
stays monotone convex; the library's canonical build is log-linear DF. A desk
hedging off this ladder should pass a smooth method — a hedge ratio that jumps
when a quote moves one basis point is not one you can trade.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date

from yieldcurve.curves.bootstrap import Quote, bootstrap
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet, DiscountCurve
from yieldcurve.instruments import OIS, FixedCouponBond, Instrument, VanillaSwap

_BASIS_POINT = 1e-4

# The quoted rate lives on the instrument for everything that carries a fixed
# leg, and only on the Quote for a Bill. Bumping one and not the other would
# bootstrap a curve against a coupon nobody quoted.
_RATE_FIELD = {FixedCouponBond: "coupon", VanillaSwap: "fixed_rate", OIS: "fixed_rate"}


def bump_quote(quote: Quote, size: float) -> Quote:
    """The same quote with its rate moved by ``size``, instrument included."""
    field = _RATE_FIELD.get(type(quote.instrument))
    instrument = (
        quote.instrument
        if field is None
        else replace(quote.instrument, **{field: getattr(quote.instrument, field) + size})  # type: ignore[type-var]
    )
    return replace(quote, instrument=instrument, rate=quote.rate + size)


def par_delta_ladder(
    instrument: Instrument,
    quotes: Sequence[Quote],
    asof: date,
    *,
    method: InterpMethod = InterpMethod.MONOTONE_CONVEX,
    bump: float = _BASIS_POINT,
    discount_curve: DiscountCurve | None = None,
) -> dict[date, float]:
    """Price change per quote for a ``bump`` rise in that quote, keyed by maturity.

    Signed like ``dv01``: positive means the position loses when the rate rises.
    ``discount_curve`` is passed through to the bootstrap, so a forecast curve
    is rebuilt dual-curve exactly as it was built.
    """
    if not quotes:
        raise ValueError("A delta ladder needs at least one quote")

    def repriced(quoted: Sequence[Quote]) -> float:
        curve = bootstrap(quoted, asof=asof, method=method, discount_curve=discount_curve)
        curves = (
            CurveSet.single(curve)
            if discount_curve is None
            else CurveSet(discount=discount_curve, forecast={"3M": curve})
        )
        return price(instrument, curves, asof).dirty

    base = repriced(quotes)
    ladder: dict[date, float] = {}
    for i, quote in enumerate(quotes):
        bumped = list(quotes)
        bumped[i] = bump_quote(quote, bump)
        ladder[quote.instrument.maturity] = base - repriced(bumped)  # type: ignore[attr-defined]
    return ladder
