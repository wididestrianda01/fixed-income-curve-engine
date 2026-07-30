"""Dual-curve pricing: the valuation difference, measured rather than asserted.

No module in ``src/`` changes for this file to pass. ``pricing._price_frn`` has
projected off ``forecast_for`` and discounted off ``discount`` since Phase 1.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from yieldcurve.calendars import USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.build import usd_curveset
from yieldcurve.curves.pricing import price
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import FRN
from yieldcurve.market.snapshot import Snapshot

_CALENDAR = USGovernmentBondCalendar()

ASOF = date(2026, 7, 24)


@pytest.fixture
def frn() -> FRN:
    """A five-year quarterly floater at 3M plus 50bp, par 100."""
    return FRN(
        issue=ASOF,
        maturity=date(2031, 7, 24),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=_CALENDAR,
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        index_tenor="3M",
        spread=0.0050,
    )


def test_single_curve_and_dual_curve_prices_differ(snapshot: Snapshot, frn: FRN) -> None:
    """The phase done condition (b).

    Under a single curve the projected forwards and the discount factors come
    from the same term structure and the floating leg telescopes to par. Under
    two curves it does not, and the residual is the dual-curve adjustment. A
    zero difference means the CurveSet is not carrying two curves.
    """
    dual = usd_curveset(snapshot, ASOF)
    single = CurveSet.single(dual.discount)

    dual_price = price(frn, dual, ASOF).dirty
    single_price = price(frn, single, ASOF).dirty

    assert abs(dual_price - single_price) > 0.01


def test_the_difference_has_the_sign_the_basis_implies(snapshot: Snapshot, frn: FRN) -> None:
    """Forecast above discount means projected coupons exceed what the discount
    curve alone would imply, so the dual-curve FRN is worth more. Getting this
    backwards is the classic dual-curve sign error, and it survives every test
    that only checks the two numbers are unequal."""
    dual = usd_curveset(snapshot, ASOF)
    single = CurveSet.single(dual.discount)

    assert price(frn, dual, ASOF).dirty > price(frn, single, ASOF).dirty


def test_zero_spread_frn_is_par_under_a_single_curve(snapshot: Snapshot) -> None:
    """The Phase 1 telescoping identity, restated on real data as the control.
    If this drifts from par the discrepancy in the test above is an artefact of
    schedule or day-count handling, not of the second curve."""
    dual = usd_curveset(snapshot, ASOF)
    flat_spread_frn = FRN(
        issue=ASOF,
        maturity=date(2031, 7, 24),
        frequency=4,
        day_count=DayCount.ACT_360,
        calendar=_CALENDAR,
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        index_tenor="3M",
        spread=0.0,
    )
    single = CurveSet.single(dual.discount)

    assert price(flat_spread_frn, single, ASOF).dirty == pytest.approx(100.0, abs=1e-8)


def test_the_pricer_was_not_modified_for_this_phase() -> None:
    """An architecture test. ``pricing.py`` must contain no reference to any
    module introduced in Phase 3; multi-curve support is a data shape, not a
    code path.

    The substring check is intentionally coarse: a grep-level gate that fails
    if anyone adds a Phase-3 import or attribute reference to pricing.py,
    including through indirect names or comments. False positives (innocent
    mentions of "build" or "snapshot") are acceptable — the test tells the
    developer to move that concern out of the pricing module.
    """
    import yieldcurve.curves.pricing as pricing_module

    source = Path(pricing_module.__file__).read_text(encoding="utf-8")

    assert "build" not in source
    assert "snapshot" not in source
