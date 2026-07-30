"""The six BCBS d368 standardised shocks."""

from __future__ import annotations

import math
import tomllib
from datetime import date
from pathlib import Path

import pytest

from curveengine.curves.protocol import FlatCurve
from curveengine.risk.scenarios import (
    ScenarioConfigError,
    bcbs_scenarios,
    shift_curve,
)

ASOF = date(2026, 7, 24)
CONFIG = Path(__file__).resolve().parents[2] / "scenarios.toml"

EXPECTED_NAMES = {
    "parallel_up",
    "parallel_down",
    "short_up",
    "short_down",
    "steepener",
    "flattener",
}


@pytest.mark.parametrize("currency", ["USD", "SEK"])
def test_six_scenarios_per_currency(currency: str) -> None:
    scenarios = bcbs_scenarios(currency)

    assert {s.name for s in scenarios} == EXPECTED_NAMES


@pytest.mark.parametrize("currency", ["USD", "SEK"])
def test_every_scenario_carries_a_citation(currency: str) -> None:
    for scenario in bcbs_scenarios(currency):
        assert scenario.citation.strip(), scenario.name
        assert "d368" in scenario.citation or "GL/2018/02" in scenario.citation


def test_parallel_up_is_flat_across_tenors() -> None:
    scenario = {s.name: s for s in bcbs_scenarios("USD")}["parallel_up"]

    sizes = [scenario.shift(t) for t in (0.25, 1.0, 5.0, 10.0, 30.0)]

    assert all(s == pytest.approx(sizes[0], abs=1e-15) for s in sizes)
    assert sizes[0] > 0.0


def test_short_shock_decays_with_the_documented_exponential() -> None:
    scenario = {s.name: s for s in bcbs_scenarios("USD")}["short_up"]

    at_zero = scenario.shift(0.0)
    at_four = scenario.shift(4.0)

    assert at_four == pytest.approx(at_zero * math.exp(-1.0), rel=1e-12)
    assert abs(scenario.shift(30.0)) < abs(at_zero) * 0.01


def test_steepener_lowers_the_front_and_raises_the_back() -> None:
    scenario = {s.name: s for s in bcbs_scenarios("USD")}["steepener"]

    assert scenario.shift(0.25) < 0.0
    assert scenario.shift(30.0) > 0.0


def test_flattener_raises_the_front_and_lowers_the_back() -> None:
    scenario = {s.name: s for s in bcbs_scenarios("USD")}["flattener"]

    assert scenario.shift(0.25) > 0.0
    assert scenario.shift(30.0) < 0.0


def test_steepener_and_flattener_are_not_mirror_images() -> None:
    by_name = {s.name: s for s in bcbs_scenarios("USD")}

    assert by_name["steepener"].shift(0.25) != pytest.approx(
        -by_name["flattener"].shift(0.25), rel=1e-6
    )


def test_usd_and_sek_share_the_same_eba_bucket() -> None:
    usd = {s.name: s for s in bcbs_scenarios("USD")}["parallel_up"].shift(1.0)
    sek = {s.name: s for s in bcbs_scenarios("SEK")}["parallel_up"].shift(1.0)

    assert usd == sek
    assert usd == pytest.approx(0.02)
    assert sek == pytest.approx(0.02)


def test_shocks_are_applied_to_a_curve_without_producing_negative_discount_factors() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.0025)

    for scenario in bcbs_scenarios("USD"):
        shifted = shift_curve(base, scenario)
        assert all(shifted.df(t) > 0.0 for t in (0.25, 1.0, 10.0, 30.0)), scenario.name


def test_config_is_valid_toml_with_every_required_field() -> None:
    with CONFIG.open("rb") as handle:
        config = tomllib.load(handle)

    for currency, block in config["currency"].items():
        for field in ("parallel_bp", "short_bp", "long_bp", "citation"):
            assert field in block, f"{currency} missing {field}"

    shape = config["shape"]
    assert "short_decay_years" in shape, "missing short_decay_years"
    assert "steepener" in shape, "missing steepener"
    assert "flattener" in shape, "missing flattener"
    for sub in ("steepener", "flattener"):
        for field in ("short_weight", "long_weight", "citation"):
            assert field in shape[sub], f"{sub} missing {field}"


def test_a_config_missing_a_currency_raises_rather_than_defaulting(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "scenarios.toml"
    empty.write_text("[currency]\n[shape]\nshort_decay_years = 4.0\n", encoding="utf-8")

    with pytest.raises(ScenarioConfigError, match="JPY"):
        bcbs_scenarios("JPY", path=empty)
