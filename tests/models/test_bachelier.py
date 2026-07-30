"""Bachelier (normal) option pricing — the SOFR swaption market convention."""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from yieldcurve.models.bachelier import bachelier_price, bachelier_vol


def test_atm_price_matches_the_closed_form() -> None:
    price = bachelier_price(0.03, 0.03, 0.0080, 2.0, pay=True)

    assert price == pytest.approx(0.0080 * math.sqrt(2.0 / (2 * math.pi)), rel=1e-12)


def test_payer_and_receiver_satisfy_put_call_parity() -> None:
    payer = bachelier_price(0.035, 0.030, 0.0080, 3.0, pay=True)
    receiver = bachelier_price(0.035, 0.030, 0.0080, 3.0, pay=False)

    assert payer - receiver == pytest.approx(0.035 - 0.030, abs=1e-14)


def test_negative_forward_is_priced_without_complaint() -> None:
    assert bachelier_price(-0.005, 0.000, 0.0080, 2.0, pay=True) > 0.0


def test_zero_volatility_gives_the_intrinsic_value() -> None:
    assert bachelier_price(0.04, 0.03, 0.0, 5.0, pay=True) == pytest.approx(0.01, abs=1e-14)
    assert bachelier_price(0.02, 0.03, 0.0, 5.0, pay=True) == pytest.approx(0.0, abs=1e-14)


def test_price_increases_with_volatility() -> None:
    prices = [bachelier_price(0.03, 0.03, v, 2.0, pay=True) for v in (0.002, 0.005, 0.01)]

    assert all(b > a for a, b in zip(prices, prices[1:], strict=False))  # noqa: RUF007


@given(
    forward=st.floats(min_value=-0.02, max_value=0.10),
    strike=st.floats(min_value=-0.02, max_value=0.10),
    vol=st.floats(min_value=0.0005, max_value=0.03),
    expiry=st.floats(min_value=0.1, max_value=20.0),
)
def test_implied_vol_inverts_the_price(
    forward: float, strike: float, vol: float, expiry: float
) -> None:
    price = bachelier_price(forward, strike, vol, expiry, pay=True)

    assume(abs(price - max(forward - strike, 0.0)) > 1e-14)

    assert bachelier_vol(price, forward, strike, expiry, pay=True) == pytest.approx(vol, rel=1e-5)


def test_a_price_below_intrinsic_is_rejected() -> None:
    with pytest.raises(ValueError, match="intrinsic"):
        bachelier_vol(0.001, 0.04, 0.03, 2.0, pay=True)
