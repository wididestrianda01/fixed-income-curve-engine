"""Key-rate durations, Ho (1992)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from yieldcurve.calendars import NullCalendar, USGovernmentBondCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.pricing import par_rate, price
from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.instruments import Bill, FixedCouponBond, VanillaSwap
from yieldcurve.risk.keyrate import (
    SEK_KEY_RATES,
    USD_KEY_RATES,
    bucket_pnl,
    hat,
    krd,
    piecewise_linear,
)
from yieldcurve.risk.scenarios import Scenario, shift_curveset
from yieldcurve.risk.sensitivities import effective_duration

ASOF = date(2026, 7, 24)


@pytest.fixture
def bond() -> FixedCouponBond:
    return FixedCouponBond(
        issue=ASOF,
        maturity=date(2036, 7, 24),
        coupon=0.04,
        frequency=2,
        day_count=DayCount.ACT_ACT_ICMA,
        calendar=USGovernmentBondCalendar(),
        bdc=BusinessDayConvention.FOLLOWING,
    )


@pytest.fixture
def flat() -> CurveSet:
    return CurveSet.single(FlatCurve(reference_date=ASOF, rate=0.04))


def test_hat_is_full_size_at_its_own_key() -> None:
    keys = (1.0, 5.0, 10.0)

    assert hat(keys, 1, 0.01).shift(5.0) == pytest.approx(0.01, abs=1e-15)


def test_hat_is_zero_at_the_neighbouring_keys() -> None:
    keys = (1.0, 5.0, 10.0)
    middle = hat(keys, 1, 0.01)

    assert middle.shift(1.0) == pytest.approx(0.0, abs=1e-15)
    assert middle.shift(10.0) == pytest.approx(0.0, abs=1e-15)


def test_hat_interpolates_linearly_between_keys() -> None:
    keys = (1.0, 5.0, 10.0)

    assert hat(keys, 1, 0.01).shift(3.0) == pytest.approx(0.005, abs=1e-15)


def test_first_hat_is_flat_before_the_first_key() -> None:
    keys = (1.0, 5.0, 10.0)
    first = hat(keys, 0, 0.01)

    assert first.shift(0.0) == pytest.approx(0.01, abs=1e-15)
    assert first.shift(0.5) == pytest.approx(0.01, abs=1e-15)


def test_last_hat_is_flat_beyond_the_last_key() -> None:
    keys = (1.0, 5.0, 10.0)
    last = hat(keys, 2, 0.01)

    assert last.shift(10.0) == pytest.approx(0.01, abs=1e-15)
    assert last.shift(50.0) == pytest.approx(0.01, abs=1e-15)


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 2.5, 5.0, 8.0, 10.0, 40.0])
def test_hats_sum_to_the_bump_at_every_time(t: float) -> None:
    keys = (1.0, 5.0, 10.0)

    total = sum(hat(keys, i, 0.01).shift(t) for i in range(len(keys)))

    assert total == pytest.approx(0.01, abs=1e-15)


def test_krd_sums_to_effective_duration(bond: FixedCouponBond, flat: CurveSet) -> None:
    """Ho hats partition unity, so the sum of the per-key central differences
    equals the parallel central difference up to the O(bump^2) truncation error
    of the finite differences. At the default 1bp bump that error term is
    ~1e-8 in duration units; the 1e-6 absolute tolerance is two orders above
    it, so the pin is on the identity, not on noise."""
    durations = krd(bond, flat, ASOF, USD_KEY_RATES)

    assert sum(durations.values()) == pytest.approx(effective_duration(bond, flat, ASOF), abs=1e-6)


def test_krd_units_are_price_basis_points_per_yield_basis_point(flat: CurveSet) -> None:
    """KRD is duration-like and carries explicit units: a 1bp rise in a key
    rate moves the price by ``durations[k]`` price basis points (1 price bp is
    1e-4 of price), numerically equal to years of duration — not multiplied by
    100. Pinned by repricing the 5y key on a zero."""
    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)
    keys = USD_KEY_RATES
    durations = krd(zero, flat, ASOF, keys)

    base = price(zero, flat, ASOF).dirty
    key = 5.0
    up = price(zero, shift_curveset(flat, hat(keys, keys.index(key), 1e-4)), ASOF).dirty

    assert (up - base) / base == pytest.approx(-durations[key] * 1e-4, rel=1e-3)


def test_krd_rejects_duplicate_tenors(bond: FixedCouponBond, flat: CurveSet) -> None:
    """A shift ladder over duplicate tenors would silently overwrite one bucket
    with the other; the grid must be strictly ascending."""
    with pytest.raises(ValueError, match="ascending"):
        krd(bond, flat, ASOF, (1.0, 1.0, 5.0))


@pytest.mark.parametrize("bump", [0.0, -1e-4, float("nan"), float("inf")])
def test_krd_rejects_an_invalid_bump(bond: FixedCouponBond, flat: CurveSet, bump: float) -> None:
    with pytest.raises(ValueError, match="bump"):
        krd(bond, flat, ASOF, USD_KEY_RATES, bump=bump)


def test_krd_rejects_a_materially_zero_base_pv(flat: CurveSet) -> None:
    """A par swap prices at zero: KRD normalizes by that, so it must refuse
    rather than return inf/NaN. The monetary ladder lives in portfolio
    ``bucket_exposure``."""
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
    swap = replace(base, fixed_rate=par_rate(base, flat, ASOF))

    with pytest.raises(ValueError, match="materially zero"):
        krd(swap, flat, ASOF, SEK_KEY_RATES)


def test_hat_rejects_a_nonfinite_size() -> None:
    with pytest.raises(ValueError, match="finite"):
        hat(SEK_KEY_RATES, 1, float("nan"))


def test_krd_is_concentrated_at_the_maturity_of_a_zero(flat: CurveSet) -> None:
    zero = Bill(maturity=date(2031, 7, 24), day_count=DayCount.ACT_365F)

    durations = krd(zero, flat, ASOF, USD_KEY_RATES)

    assert durations[5.0] == max(durations.values())
    assert abs(durations[30.0]) < 1e-9


def test_krd_of_a_coupon_bond_spans_the_whole_curve(bond: FixedCouponBond, flat: CurveSet) -> None:
    durations = krd(bond, flat, ASOF, USD_KEY_RATES)

    assert durations[10.0] == max(durations.values())
    assert durations[1.0] > 0.0


def test_bucket_pnl_reconciles_to_a_full_reprice(bond: FixedCouponBond, flat: CurveSet) -> None:
    shifts = {
        0.25: 0.0020,
        0.5: 0.0018,
        1.0: 0.0015,
        2.0: 0.0010,
        3.0: 0.0006,
        5.0: 0.0000,
        7.0: -0.0004,
        10.0: -0.0008,
        20.0: -0.0012,
        30.0: -0.0015,
    }

    scenario = Scenario(name="twist", shift=piecewise_linear(USD_KEY_RATES, shifts))
    base = price(bond, flat, ASOF).dirty
    actual = price(bond, shift_curveset(flat, scenario), ASOF).dirty - base

    predicted = bucket_pnl(bond, flat, ASOF, USD_KEY_RATES, shifts)

    assert predicted == pytest.approx(actual, abs=1e-4 * base)


def test_the_sek_grid_matches_the_specification() -> None:
    assert SEK_KEY_RATES == (0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0)


def test_the_usd_grid_matches_the_specification() -> None:
    assert USD_KEY_RATES == (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def test_unsorted_keys_are_rejected(bond: FixedCouponBond, flat: CurveSet) -> None:
    with pytest.raises(ValueError, match="ascending"):
        krd(bond, flat, ASOF, (5.0, 1.0, 10.0))


def test_hats_partition_unity_across_the_key_rate_grid() -> None:
    """Ho (1992) hats sum to the parallel shift at every maturity, including outside."""
    keys = SEK_KEY_RATES
    for t in (0.0, 0.1, 0.25, 0.9, 1.0, 3.5, 7.0, 10.0, 25.0):
        total = sum(hat(keys, i, 1.0).shift(t) for i in range(len(keys)))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_a_hat_is_one_at_its_own_key_and_zero_at_its_neighbours() -> None:
    keys = SEK_KEY_RATES
    index = keys.index(2.0)
    shift = hat(keys, index, 1.0).shift
    assert shift(2.0) == pytest.approx(1.0)
    assert shift(1.0) == pytest.approx(0.0)
    assert shift(5.0) == pytest.approx(0.0)


def test_the_end_hats_extrapolate_flat() -> None:
    keys = SEK_KEY_RATES
    assert hat(keys, 0, 1.0).shift(0.0) == pytest.approx(1.0)
    assert hat(keys, len(keys) - 1, 1.0).shift(40.0) == pytest.approx(1.0)


def test_hat_rejects_an_index_outside_the_grid() -> None:
    with pytest.raises(IndexError):
        hat(SEK_KEY_RATES, len(SEK_KEY_RATES), 1.0)


def test_key_rate_durations_sum_to_the_effective_duration(
    bond: FixedCouponBond,
    flat: CurveSet,
) -> None:
    """A consequence of partition of unity: the ladder must reconstruct a parallel move."""
    ladder = krd(bond, flat, ASOF, SEK_KEY_RATES)
    parallel_duration = effective_duration(bond, flat, ASOF)
    assert sum(ladder.values()) == pytest.approx(parallel_duration, rel=2e-2)
