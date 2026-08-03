"""Unit tests for portfolio aggregation, ΔEVE, and historical VaR/ES."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from yieldcurve.calendars import NullCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import FixedCouponBond
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
from yieldcurve.risk.scenarios import bcbs_scenarios, parallel, shift_curveset

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


def test_present_value_is_additive_across_positions(
    two_bonds: Portfolio, flat_curves: CurveSet
) -> None:
    total = present_value(two_bonds, flat_curves, ASOF)
    parts = sum(
        present_value(Portfolio(positions=(p,)), flat_curves, ASOF) for p in two_bonds.positions
    )
    assert total == pytest.approx(parts, rel=1e-12)


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
    scenarios = bcbs_scenarios("SEK")
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
        '[[position]]\nlabel = "x"\nkind = "collateralised_moon_rock"\nnotional = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(PortfolioError, match="collateralised_moon_rock"):
        Portfolio.from_toml(bad)


def test_from_toml_rejects_a_position_missing_a_required_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[[position]]\nlabel = "x"\nkind = "bond"\nnotional = 1.0\n', encoding="utf-8")
    with pytest.raises(PortfolioError, match="issue"):
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


def test_expected_shortfall_is_at_least_as_large_as_var() -> None:
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.0, 1.0, size=2000)
    for confidence in (0.95, 0.99):
        var, es = var_es(pnl, confidence=confidence)
        assert es >= var


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
