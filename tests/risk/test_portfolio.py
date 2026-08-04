"""Unit tests for portfolio aggregation, ΔEVE, and historical VaR/ES."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import yieldcurve.risk.portfolio as portfolio_module
from yieldcurve.calendars import NullCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import par_rate
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import FixedCouponBond, VanillaSwap
from yieldcurve.risk.keyrate import SEK_KEY_RATES
from yieldcurve.risk.portfolio import (
    Portfolio,
    PortfolioError,
    Position,
    bucket_exposure,
    delta_eve,
    eve_ladder,
    historical_pnl,
    present_value,
    var_es,
)
from yieldcurve.risk.scenarios import eu_scenarios, parallel, shift_curveset

ASOF = date(2026, 7, 24)
DEMO_TOML = Path(__file__).resolve().parents[2] / "data" / "demo_portfolio.toml"


@pytest.fixture
def flat_curves() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.02))


@pytest.fixture
def two_bonds() -> Portfolio:
    bond_a = FixedCouponBond(
        issue=date(2020, 5, 12),
        maturity=date(2031, 5, 12),
        coupon=0.00125,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    bond_b = FixedCouponBond(
        issue=date(2022, 5, 6),
        maturity=date(2033, 11, 11),
        coupon=0.0175,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    return Portfolio(
        positions=(
            Position(label="A", instrument=bond_a, notional=1_000_000.0),
            Position(label="B", instrument=bond_b, notional=-500_000.0),
        )
    )


def test_present_value_matches_hand_discounted_cashflows(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    """Independent oracle: the 0.125% SGB in ``two_bonds`` is discounted by hand
    from its known cash flows. 30/360 annual coupons pay 0.125 per 100 face
    every May 12, plus 100.125 at maturity; the flat 2% curve discounts with
    e^{-0.02t}. No library pricing code participates."""
    long_only = Portfolio(positions=(two_bonds.positions[0],))
    flows = (
        (date(2027, 5, 12), 0.125),
        (date(2028, 5, 12), 0.125),
        (date(2029, 5, 12), 0.125),
        (date(2030, 5, 12), 0.125),
        (date(2031, 5, 12), 100.125),
    )
    expected = (
        sum(amount * math.exp(-0.02 * (d - ASOF).days / 365.0) for d, amount in flows)
        / 100.0
        * 1_000_000.0
    )
    assert present_value(long_only, flat_curves, ASOF) == pytest.approx(expected, rel=1e-9)


def test_short_position_contributes_negative_value(flat_curves: CurveSet) -> None:
    bond = FixedCouponBond(
        issue=date(2020, 5, 12),
        maturity=date(2031, 5, 12),
        coupon=0.02,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    long = Portfolio(positions=(Position(label="L", instrument=bond, notional=1_000_000.0),))
    short = Portfolio(positions=(Position(label="S", instrument=bond, notional=-1_000_000.0),))
    assert present_value(long, flat_curves, ASOF) == pytest.approx(
        -present_value(short, flat_curves, ASOF), rel=1e-12
    )
    assert present_value(long, flat_curves, ASOF) > 0.0


def test_bond_positions_scale_by_face_value(flat_curves: CurveSet) -> None:
    """One scale contract for bonds: position value is notional / face x price.
    Two bonds with identical terms but different face amounts quote different
    prices per unit of face, so the same position notional must buy the same
    book value."""

    def _make(face: float) -> FixedCouponBond:
        return FixedCouponBond(
            issue=date(2020, 5, 12),
            maturity=date(2031, 5, 12),
            coupon=0.02,
            frequency=1,
            day_count=DayCount.THIRTY_360_BOND,
            calendar=NullCalendar(),
            bdc=BusinessDayConvention.UNADJUSTED,
            face=face,
        )

    small = _make(100.0)
    large = _make(250.0)
    notional = 1_000_000.0

    value_small = present_value(
        Portfolio(positions=(Position(label="s", instrument=small, notional=notional),)),
        flat_curves,
        ASOF,
    )
    value_large = present_value(
        Portfolio(positions=(Position(label="l", instrument=large, notional=notional),)),
        flat_curves,
        ASOF,
    )
    assert value_small == pytest.approx(value_large, rel=1e-12)


def test_swap_positions_scale_by_notional(flat_curves: CurveSet) -> None:
    """One scale contract for swaps: the swap's own notional is divided out of
    the quoted value, and the position notional scales linearly. Striking at a
    non-par rate keeps the book value non-degenerate."""
    base = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.03,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    unit = replace(base, notional=1.0)
    big = replace(base, notional=3_000_000.0)
    notional = 5_000_000.0

    value_unit = present_value(
        Portfolio(positions=(Position(label="u", instrument=unit, notional=notional),)),
        flat_curves,
        ASOF,
    )
    value_big = present_value(
        Portfolio(positions=(Position(label="b", instrument=big, notional=notional),)),
        flat_curves,
        ASOF,
    )
    assert value_unit == pytest.approx(value_big, rel=1e-9)

    doubled = present_value(
        Portfolio(positions=(Position(label="d", instrument=unit, notional=2 * notional),)),
        flat_curves,
        ASOF,
    )
    assert doubled == pytest.approx(2.0 * value_unit, rel=1e-12)


def test_delta_eve_is_negative_for_a_long_book_under_a_rate_rise(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    long_only = Portfolio(positions=(two_bonds.positions[0],))
    change = delta_eve(long_only, flat_curves, ASOF, parallel(0.0200))
    assert change < 0.0


def test_delta_eve_of_a_zero_shock_is_zero(two_bonds: Portfolio, flat_curves: CurveSet) -> None:
    assert delta_eve(two_bonds, flat_curves, ASOF, parallel(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_eve_ladder_key_order_matches_the_scenario_sequence(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    scenarios = eu_scenarios("SEK")
    ladder = eve_ladder(two_bonds, flat_curves, ASOF, scenarios)
    assert tuple(ladder) == tuple(s.name for s in scenarios)
    assert len(ladder) == 6


def test_from_toml_round_trips_the_demo_portfolio() -> None:
    book = Portfolio.from_toml(DEMO_TOML)
    assert len(book.positions) == 6
    assert all(p.label for p in book.positions)
    assert all(p.notional != 0.0 for p in book.positions)


def test_from_toml_rejects_an_unknown_instrument_kind(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        '[[position]]\nlabel = "x"\nkind = "collateralised_moon_rock"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="collateralised_moon_rock"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_position_missing_a_required_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="issue"):
        Portfolio.from_toml(bad)


def test_from_toml_requires_a_declared_currency(tmp_path: Path) -> None:
    """The supported portfolio is explicitly single-currency: the file must
    declare which currency the notionals are in (spec section 4)."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\n"
        "coupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="currency"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_non_string_currency(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "currency = 2024\n"
        '[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="currency"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_wrong_typed_fields(tmp_path: Path) -> None:
    """Exact TOML types, no coercion (error policy): a string notional or
    coupon, a float frequency, or a string date is invalid input, not
    something to coerce."""
    common = 'label = "x"\nkind = "bond"\n'
    cases = {
        "notional": common + 'notional = "one million"\n',
        "coupon": common + 'coupon = "0.01"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        "frequency": common + "coupon = 0.01\nfrequency = 1.5\n"
        "issue = 2020-05-12\nmaturity = 2031-05-12\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        "issue": common + "coupon = 0.01\nfrequency = 1\n"
        'issue = "2020-05-12"\nmaturity = 2031-05-12\n'
        'day_count = "30/360"\nnotional = 1.0\n',
    }
    for field, body in cases.items():
        bad = tmp_path / f"{field}.toml"
        bad.write_text(f'currency = "SEK"\n[[position]]\n{body}', encoding="utf-8")
        with pytest.raises(PortfolioError, match=field):
            Portfolio.from_toml(bad)


def test_from_toml_rejects_a_zero_notional(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 0.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="notional"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_an_invalid_frequency(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 5\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="frequency"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_reversed_bond_dates(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2031-05-12\nmaturity = 2020-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="after"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_an_invalid_float_tenor(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "swap"\n'
        "start = 2026-07-24\nmaturity = 2031-07-24\nfixed_rate = 0.03\n"
        "fixed_frequency = 1\nfixed_day_count = '30/360'\n"
        'float_tenor = "9M"\nfloat_day_count = "ACT/360"\npay_fixed = true\n'
        "notional = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="float_tenor"):
        Portfolio.from_toml(bad)


@pytest.mark.parametrize("value", ['"yes"', '"false"', "1", "0"])
def test_from_toml_rejects_a_non_boolean_pay_fixed(tmp_path: Path, value: str) -> None:
    """pay_fixed = "false" (or any non-boolean) is rejected, never coerced —
    the brief's canonical strict-loader case (behavioral test 1)."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "swap"\n'
        "start = 2026-07-24\nmaturity = 2031-07-24\nfixed_rate = 0.03\n"
        "fixed_frequency = 1\nfixed_day_count = '30/360'\n"
        'float_tenor = "3M"\nfloat_day_count = "ACT/360"\n'
        f"pay_fixed = {value}\n"
        "notional = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="pay_fixed"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_non_finite_notional(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = inf\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="notional"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_an_fx_mapping(tmp_path: Path) -> None:
    """The supported portfolio is single-currency: a currency conversion
    table is rejected with a named error — no FX mapping exists (spec
    section 4)."""
    bad = tmp_path / "fx.toml"
    bad.write_text(
        'currency = "SEK"\n[fx]\nUSD = 10.5\n'
        '[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="fx"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_position_with_its_own_currency(tmp_path: Path) -> None:
    bad = tmp_path / "posfx.toml"
    bad.write_text(
        'currency = "SEK"\n[[position]]\nlabel = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\ncurrency = "USD"\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="currency"):
        Portfolio.from_toml(bad)


def test_bucket_exposure_is_negative_for_a_long_fixed_rate_book(
    flat_curves: CurveSet,
) -> None:
    bond = FixedCouponBond(
        issue=date(2020, 5, 12),
        maturity=date(2031, 5, 12),
        coupon=0.02,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    long = Portfolio(positions=(Position(label="L", instrument=bond, notional=1_000_000.0),))
    exposure = bucket_exposure(long, flat_curves, ASOF, SEK_KEY_RATES)
    assert tuple(exposure) == SEK_KEY_RATES
    assert sum(exposure.values()) < 0.0


def test_bucket_exposures_sum_to_the_parallel_sensitivity(flat_curves: CurveSet) -> None:
    """Ho hats partition unity, so bumping every bucket equals a parallel bump."""
    bond = FixedCouponBond(
        issue=date(2020, 5, 12),
        maturity=date(2029, 5, 12),
        coupon=0.02,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
    )
    book = Portfolio(positions=(Position(label="L", instrument=bond, notional=1_000_000.0),))
    bump = 1e-4
    total = sum(bucket_exposure(book, flat_curves, ASOF, SEK_KEY_RATES, bump=bump).values())
    up = present_value(book, shift_curveset(flat_curves, parallel(bump)), ASOF)
    down = present_value(book, shift_curveset(flat_curves, parallel(-bump)), ASOF)
    assert total == pytest.approx((up - down) / (2.0 * bump), rel=1e-6)


def test_historical_pnl_returns_one_number_per_observation(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    rng = np.random.default_rng(20260803)
    tenors = (0.25, 1.0, 5.0, 10.0)
    changes = rng.normal(0.0, 0.0005, size=(40, len(tenors)))
    pnl = historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors)
    assert pnl.shape == (40,)
    assert np.isfinite(pnl).all()


def test_historical_pnl_of_zero_changes_is_zero(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    tenors = (0.25, 1.0, 5.0, 10.0)
    changes = np.zeros((5, len(tenors)))
    pnl = historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors)
    assert np.allclose(pnl, 0.0, atol=1e-9)


def test_historical_pnl_matches_a_hand_contracted_parallel_move(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    """A uniform 1bp rise across every input tenor is a parallel bump on the keys."""
    tenors = SEK_KEY_RATES
    changes = np.full((1, len(tenors)), 1e-4)
    exposure = bucket_exposure(two_bonds, flat_curves, ASOF, SEK_KEY_RATES)
    expected = 1e-4 * sum(exposure.values())
    pnl = historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors)
    assert pnl[0] == pytest.approx(expected, rel=1e-9)


def test_historical_pnl_rejects_a_column_count_mismatch(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    with pytest.raises(PortfolioError, match="columns"):
        historical_pnl(two_bonds, flat_curves, ASOF, np.zeros((5, 3)), (0.25, 1.0, 5.0, 10.0))


def test_historical_pnl_rejects_nonfinite_changes(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    tenors = (0.25, 1.0, 5.0, 10.0)
    changes = np.array([[1e-4, np.nan, 1e-4, 1e-4]])
    with pytest.raises(PortfolioError, match="finite"):
        historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors)


def test_historical_pnl_rejects_unsorted_tenors(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    with pytest.raises(PortfolioError, match="increasing"):
        historical_pnl(two_bonds, flat_curves, ASOF, np.zeros((3, 4)), (0.25, 5.0, 1.0, 10.0))


def test_historical_pnl_rejects_an_invalid_key_grid(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    """Aligned-history validation before arithmetic: the bucket key grid must
    be a strictly ascending finite ladder, like the tenor grid."""
    tenors = (0.25, 1.0, 5.0, 10.0)
    changes = np.zeros((3, len(tenors)))
    with pytest.raises(PortfolioError, match="increasing"):
        historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors, keys=(0.25, 1.0, 10.0, 5.0))
    with pytest.raises(PortfolioError, match="finite"):
        historical_pnl(two_bonds, flat_curves, ASOF, changes, tenors, keys=(0.25, np.nan, 5.0))


def test_bucket_exposure_rejects_an_invalid_bump(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    with pytest.raises(PortfolioError, match="bump"):
        bucket_exposure(two_bonds, flat_curves, ASOF, SEK_KEY_RATES, bump=0.0)
    with pytest.raises(PortfolioError, match="bump"):
        bucket_exposure(two_bonds, flat_curves, ASOF, SEK_KEY_RATES, bump=-1e-4)


def test_bucket_exposure_remains_available_for_a_par_swap(flat_curves: CurveSet) -> None:
    """A par swap prices at zero, where normalized duration is undefined, but
    the monetary BPV ladder never divides by the base price. Its sum must
    reconcile to the parallel monetary sensitivity computed independently."""
    base = VanillaSwap(
        start=ASOF,
        maturity=date(2031, 7, 24),
        fixed_rate=0.0,
        fixed_frequency=1,
        fixed_day_count=DayCount.THIRTY_360_BOND,
        float_tenor="3M",
        float_day_count=DayCount.ACT_360,
        calendar=NullCalendar(),
        bdc=BusinessDayConvention.UNADJUSTED,
        notional=1.0,
    )
    swap = replace(base, fixed_rate=par_rate(base, flat_curves, ASOF))
    book = Portfolio(positions=(Position(label="par", instrument=swap, notional=1_000_000.0),))

    exposure = bucket_exposure(book, flat_curves, ASOF, SEK_KEY_RATES)
    assert all(np.isfinite(v) for v in exposure.values())

    bump = 1e-4
    up = present_value(book, shift_curveset(flat_curves, parallel(bump)), ASOF)
    down = present_value(book, shift_curveset(flat_curves, parallel(-bump)), ASOF)
    assert sum(exposure.values()) == pytest.approx((up - down) / (2.0 * bump), rel=1e-6)


def test_var_es_tail_direction_matches_hand_derived_values() -> None:
    """Independent oracle for the loss-tail direction: losses of 1 (x190),
    100 (x10) and 200 (x1). With n=201 the linear-quantile convention puts the
    95% quantile exactly on the 190th order statistic, so VaR = 100 and
    ES = mean of the tail beyond it = 1200/11 — a strict es > var pin on
    asymmetric loss data, not an inequality over random draws."""
    pnl = np.array([-1.0] * 190 + [-100.0] * 10 + [-200.0])
    var, es = var_es(pnl, confidence=0.95)

    assert var == pytest.approx(100.0, rel=1e-12)
    assert es == pytest.approx(1200.0 / 11.0, rel=1e-12)
    assert es > var


def test_var_es_rejects_a_gain_at_the_confidence_quantile() -> None:
    """Mandatory Task 6 handoff: var_es reports non-negative loss magnitudes.
    When the confidence quantile of the loss distribution is a gain (negative
    loss magnitude), the convention has no loss to report — raising with the
    quantile as context is honest, a silent negative 'positive loss' is not."""
    pnl = np.full(200, 1.0)  # every observation a gain; losses all -1.0
    with pytest.raises(PortfolioError, match="gain"):
        var_es(pnl, confidence=0.95)


def test_var_es_rejects_nonfinite_pnl() -> None:
    with pytest.raises(PortfolioError, match="finite"):
        var_es(np.array([1.0, np.nan, 2.0]))


def test_var_es_raises_when_the_tail_is_too_thin() -> None:
    pnl = np.linspace(-1.0, 1.0, 50)
    with pytest.raises(PortfolioError, match="tail"):
        var_es(pnl, confidence=0.99)


def test_var_es_rejects_a_confidence_outside_the_unit_interval() -> None:
    pnl = np.linspace(-1.0, 1.0, 2000)
    with pytest.raises(PortfolioError, match="confidence"):
        var_es(pnl, confidence=1.0)


def test_from_toml_rejects_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.toml"
    empty.write_text("tier1_capital = 1_000.0\n", encoding="utf-8")
    with pytest.raises(PortfolioError, match="no \\[\\[position\\]\\]"):
        Portfolio.from_toml(empty)


def test_from_toml_rejects_an_unknown_day_count(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        "[[position]]\n"
        'label = "x"\n'
        'kind = "bond"\n'
        "issue = 2020-05-12\n"
        "maturity = 2031-05-12\n"
        "coupon = 0.01\n"
        "frequency = 1\n"
        'day_count = "nonsense"\n'
        "notional = 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="nonsense"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    """SEC-01/QUANTRISK-12: unknown document keys cannot inject behavior; only
    the documented allowlist (currency, position, tier1_capital) is accepted."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        "[magic]\n"
        "leverage = 99\n"
        "[[position]]\n"
        'label = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="unknown"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_an_unknown_position_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        "[[position]]\n"
        'label = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n'
        'rating = "AAA"\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="rating"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_key_that_is_not_valid_for_the_kind(tmp_path: Path) -> None:
    """A bond entry carrying a swap-only field is malformed input, not an
    ignored field."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        "[[position]]\n"
        'label = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n'
        "pay_fixed = true\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="pay_fixed"):
        Portfolio.from_toml(bad)


def test_from_toml_accepts_the_allowlisted_tier1_capital(tmp_path: Path) -> None:
    """tier1_capital is the disclosed invented capital denominator of the demo
    exhibit (spec section 4); it is allowlisted and validated as a positive
    finite number."""
    good = tmp_path / "good.toml"
    good.write_text(
        'currency = "SEK"\n'
        "tier1_capital = 4_000_000_000.0\n"
        "[[position]]\n"
        'label = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    book = Portfolio.from_toml(good)
    assert len(book.positions) == 1


@pytest.mark.parametrize("value", ["-1.0", "0.0", '"four billion"', "inf"])
def test_from_toml_rejects_an_invalid_tier1_capital(tmp_path: Path, value: str) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        f"tier1_capital = {value}\n"
        "[[position]]\n"
        'label = "x"\nkind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="tier1_capital"):
        Portfolio.from_toml(bad)


def test_from_toml_wraps_malformed_toml_in_the_named_error(tmp_path: Path) -> None:
    """A file with duplicate keys (the loader-level 'duplicate tenors' class of
    defect) is malformed TOML and surfaces as PortfolioError with file context,
    not as a raw parser exception."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'currency = "SEK"\n'
        "[[position]]\n"
        'label = "x"\nlabel = "y"\n'
        'kind = "bond"\n'
        "issue = 2020-05-12\nmaturity = 2031-05-12\ncoupon = 0.01\nfrequency = 1\n"
        'day_count = "30/360"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="invalid TOML"):
        Portfolio.from_toml(bad)


def test_result_labels_are_illustrative_delta_eve_not_regulatory_capital() -> None:
    """Controller decision (behavioral test 5): portfolio result labels say
    illustrative Delta EVE; the disclosed capital proxy is never called
    regulatory capital. The public docstrings are the labels' contract."""
    module_doc = inspect.getdoc(portfolio_module)
    assert module_doc is not None
    lowered = module_doc.lower()
    assert "illustrative delta eve" in lowered
    assert "regulatory capital" not in lowered

    for name in ("delta_eve", "eve_ladder"):
        doc = inspect.getdoc(getattr(portfolio_module, name))
        assert doc is not None
        assert "illustrative" in doc.lower(), name
        assert "delta eve" in doc.lower(), name
        assert "regulatory capital" not in doc.lower(), name

    loader_doc = inspect.getdoc(Portfolio.from_toml)
    assert loader_doc is not None
    assert "not regulatory capital" in loader_doc.lower()
