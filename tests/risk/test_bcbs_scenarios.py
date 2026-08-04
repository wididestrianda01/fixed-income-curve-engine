"""EU 2024/856 supervisory shock scenarios: typed configuration, pinned parameters.

The six supervisory shock scenarios of Commission Delegated Regulation (EU)
2024/856 are implemented for educational analysis. Every parameter below is
pinned to the independent current-source table transcribed in this module's
header; the expected values are computed in the tests from that transcription,
never from the library's own code.

Independent current-source table
================================
Commission Delegated Regulation (EU) 2024/856 of 1 December 2023 supplementing
Directive 2013/36/EU with regard to regulatory technical standards specifying
the supervisory shock scenarios, the common modelling and parametric
assumptions and what constitutes a large decline.

- EUR-Lex CELEX 32024R0856:
  https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R0856
- ELI: http://data.europa.eu/eli/reg_del/2024/856/oj
- OJ L, 2024/856, 24.4.2024, p. 1 (ISSN 1977-0677).

Article 1(1) — the six supervisory shock scenarios (EVE):
    (a) parallel shock up: same positive shock for all maturities;
    (b) parallel shock down: same negative shock for all maturities;
    (c) steepener: negative shocks for shorter maturities, positive for longer;
    (d) flattener: positive shocks for shorter maturities, negative for longer;
    (e) short rates shock up: larger positive shocks for shorter maturities,
        converging to the baseline for longer maturities;
    (f) short rates shock down: larger negative shocks for shorter maturities,
        converging to the baseline for longer maturities.
Article 1(2) — the two NII scenarios (parallel up / parallel down) are not
implemented here: this module has no net-interest-income model.

Article 2 — parameterisations, with t_k the midpoint of time bucket k:
    (1) parallel:  ΔR_parallel,c(t_k) = ± R_parallel,c
    (2) short:     ΔR_short,c(t_k)    = ± R_short,c · e^(-t_k / 4)
    (3) long:      ΔR_long,c(t_k)     = ± R_long,c · (1 - e^(-t_k / 4))
    (4) rotation:  ΔR_steepener,c(t_k) = -0.65·|ΔR_short,c(t_k)| + 0.9·|ΔR_long,c(t_k)|
                   ΔR_flattener,c(t_k) = +0.8·|ΔR_short,c(t_k)| - 0.6·|ΔR_long,c(t_k)|

Annex Part A — currency-specific interest rate shocks (basis points). Each
currency has its own row; there are no currency buckets:
    USD: parallel 200, short 300, long 150
    SEK: parallel 200, short 300, long 150

Article 3(7) — maturity-dependent post-shock rate floor:
    "A maturity-dependent post-shock interest rate floor shall be applied for
    each currency starting with -150 basis points for immediate maturity. That
    floor shall increase by 3 basis points per year, eventually reaching 0 %
    for maturities of 50 years and more. If observed interest rates are lower
    than the post-shock interest rate floor, institutions shall apply the lower
    observed interest rate."
    → floor(t) = min(0, -0.015 + 0.0003·t), with the observed-rate override.

Article 6 — entry into force: twentieth day after publication in the Official
Journal (published 24.4.2024, so 14 May 2024). The scenarios operationalise
Article 98(5) of Directive 2013/36/EU; Directive (EU) 2024/1619 (CRD VI),
Article 2, requires Member States to apply those provisions from 11 January
2026. Part B of the Annex (calibration for currencies without a Part A row) is
not implemented: USD and SEK both have Part A rows.
"""

from __future__ import annotations

import importlib.resources
import inspect
import math
from datetime import date
from pathlib import Path

import pytest

import yieldcurve.risk.scenarios as scenarios
from yieldcurve.curves.protocol import FlatCurve
from yieldcurve.risk.scenarios import (
    Scenario,
    ScenarioConfigError,
    eu_scenarios,
    load_scenarios,
    parallel,
    post_shock_floor,
    shift_curve,
)

ASOF = date(2026, 7, 24)
_BP = 1e-4

# Transcribed from the independent source table above (Annex Part A): each
# currency's (parallel, short, long) shock in basis points.
ANNEX_PART_A: dict[str, tuple[int, int, int]] = {
    "USD": (200, 300, 150),
    "SEK": (200, 300, 150),
}
# Article 2(2)-(3): the short-rate decay horizon.
DECAY_YEARS: float = 4.0
# Article 2(4): (short_weight, long_weight) per rotation shock.
SHOCK_SHAPES: dict[str, tuple[float, float]] = {
    "steepener": (-0.65, 0.9),
    "flattener": (0.8, -0.6),
}
# Article 3(7): floor starts at -150bp and rises 3bp per year to 0% at 50y.
FLOOR_START: float = -0.015
FLOOR_SLOPE: float = 0.0003

EXPECTED_NAMES = {
    "parallel_up",
    "parallel_down",
    "short_up",
    "short_down",
    "steepener",
    "flattener",
}

# A valid configuration used to build the malformed variants in the loader
# tests. The values mirror the packaged scenarios.toml.
VALID_CONFIG = """\
[shape]
short_decay_years = 4.0
citation = "Commission Delegated Regulation (EU) 2024/856, Article 2"

[shape.steepener]
short_weight = -0.65
long_weight = 0.9
citation = "Commission Delegated Regulation (EU) 2024/856, Article 2(4)"

[shape.flattener]
short_weight = 0.8
long_weight = -0.6
citation = "Commission Delegated Regulation (EU) 2024/856, Article 2(4)"

[currency.USD]
parallel_bp = 200
short_bp = 300
long_bp = 150
citation = "Commission Delegated Regulation (EU) 2024/856, Annex Part A"

[currency.SEK]
parallel_bp = 200
short_bp = 300
long_bp = 150
citation = "Commission Delegated Regulation (EU) 2024/856, Annex Part A"
"""


def _by_name(scenarios_tuple: tuple[Scenario, ...]) -> dict[str, Scenario]:
    return {s.name: s for s in scenarios_tuple}


def _expected_shift(name: str, t: float, p: int, s: int, long_bp: int) -> float:
    """Independent transcription of Article 2: the expected shift at tenor ``t``
    computed from the regulation's formulas, not from the library's code."""
    short_factor = math.exp(-t / DECAY_YEARS)
    long_factor = 1.0 - short_factor
    if name == "parallel_up":
        return p * _BP
    if name == "parallel_down":
        return -p * _BP
    if name == "short_up":
        return s * _BP * short_factor
    if name == "short_down":
        return -s * _BP * short_factor
    if name == "steepener":
        steep_short, steep_long = SHOCK_SHAPES["steepener"]
        return steep_short * s * _BP * short_factor + steep_long * long_bp * _BP * long_factor
    if name == "flattener":
        flat_short, flat_long = SHOCK_SHAPES["flattener"]
        return flat_short * s * _BP * short_factor + flat_long * long_bp * _BP * long_factor
    raise AssertionError(f"unknown scenario name {name!r}")


def _variant(replacements: dict[str, str]) -> str:
    body = VALID_CONFIG
    for old, new in replacements.items():
        assert old in body, f"replacement target {old!r} not found in template"
        body = body.replace(old, new, 1)
    return body


def _expect_rejected(tmp_path: Path, body: str, match: str) -> None:
    bad = tmp_path / "scenarios.toml"
    bad.write_text(body, encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match=match):
        eu_scenarios("USD", path=bad)


# ---------------------------------------------------------------------------
# Scenario set and parameter pins (behavioral tests 2 and 3)
# ---------------------------------------------------------------------------


def test_the_six_supervisory_scenarios_of_article_1_1() -> None:
    for currency in ANNEX_PART_A:
        assert {s.name for s in eu_scenarios(currency)} == EXPECTED_NAMES


@pytest.mark.parametrize("currency", ["USD", "SEK"])
def test_every_scenario_carries_a_2024_856_citation(currency: str) -> None:
    for scenario in eu_scenarios(currency):
        assert scenario.citation.strip(), scenario.name
        assert "2024/856" in scenario.citation


@pytest.mark.parametrize("currency", ["USD", "SEK"])
@pytest.mark.parametrize(
    "name",
    ["parallel_up", "parallel_down", "short_up", "short_down", "steepener", "flattener"],
)
@pytest.mark.parametrize("t", [0.0, 0.25, 1.0, 4.0, 10.0, 30.0])
def test_shifts_match_the_transcribed_article_2_formulas(
    currency: str, name: str, t: float
) -> None:
    """Every USD/SEK parallel/short/long parameter pinned to the transcribed
    Annex Part A table and Article 2 parameterisations (behavioral test 2)."""
    p, s, long_bp = ANNEX_PART_A[currency]
    scenario = _by_name(eu_scenarios(currency))[name]
    assert scenario.shift(t) == pytest.approx(_expected_shift(name, t, p, s, long_bp), rel=1e-12)


def test_usd_and_sek_have_their_own_annex_rows_with_identical_values() -> None:
    """Annex Part A gives each currency its own row; USD and SEK happen to be
    identical (200/300/150 bp). The pin is per-row, not a shared bucket."""
    assert ANNEX_PART_A["USD"] == (200, 300, 150)
    assert ANNEX_PART_A["SEK"] == (200, 300, 150)
    usd = _by_name(eu_scenarios("USD"))["parallel_up"].shift(1.0)
    sek = _by_name(eu_scenarios("SEK"))["parallel_up"].shift(1.0)
    assert usd == pytest.approx(0.02, rel=1e-12)
    assert sek == pytest.approx(0.02, rel=1e-12)


def test_short_shock_decays_with_the_article_2_exponential() -> None:
    scenario = _by_name(eu_scenarios("USD"))["short_up"]
    assert scenario.shift(0.0) == pytest.approx(300 * _BP, rel=1e-12)
    assert scenario.shift(DECAY_YEARS) == pytest.approx(
        scenario.shift(0.0) * math.exp(-1.0), rel=1e-12
    )
    assert abs(scenario.shift(30.0)) < abs(scenario.shift(0.0)) * 0.01


def test_steepener_and_flattener_are_the_article_2_rotations() -> None:
    by = _by_name(eu_scenarios("USD"))
    p, s, long_bp = ANNEX_PART_A["USD"]
    # At t=0 the long factor vanishes, so the rotation is pure short shock.
    assert by["steepener"].shift(0.0) == pytest.approx(-0.65 * s * _BP, rel=1e-12)
    assert by["flattener"].shift(0.0) == pytest.approx(0.8 * s * _BP, rel=1e-12)
    # At t=50 the short factor is ~3.7e-6; the transcribed formula pins the
    # exact value including that residual (long dominates at ~90bp).
    assert by["steepener"].shift(50.0) == pytest.approx(
        _expected_shift("steepener", 50.0, p, s, long_bp), rel=1e-12
    )
    assert by["flattener"].shift(50.0) == pytest.approx(
        _expected_shift("flattener", 50.0, p, s, long_bp), rel=1e-12
    )
    # Not mirror images (the two rotation formulas have different coefficients).
    assert by["steepener"].shift(0.25) != pytest.approx(-by["flattener"].shift(0.25), rel=1e-6)


# ---------------------------------------------------------------------------
# Post-shock maturity-dependent floor (behavioral test 3)
# ---------------------------------------------------------------------------


def test_post_shock_floor_matches_article_3_7() -> None:
    """floor(t) = min(0, -150bp + 3bp/year * t): -150bp at t=0, 0% from t=50y."""
    assert post_shock_floor(0.0) == pytest.approx(FLOOR_START, abs=1e-15)
    assert post_shock_floor(10.0) == pytest.approx(FLOOR_START + 10 * FLOOR_SLOPE, abs=1e-15)
    assert post_shock_floor(25.0) == pytest.approx(FLOOR_START + 25 * FLOOR_SLOPE, abs=1e-15)
    assert post_shock_floor(50.0) == pytest.approx(0.0, abs=1e-15)
    assert post_shock_floor(60.0) == pytest.approx(0.0, abs=1e-15)


def test_down_shock_is_floored_at_the_prescribed_level() -> None:
    """A 0.1% flat curve shocked down 200bp lands on the maturity-dependent
    floor: -150bp at t=0 and -120bp at t=10, not on the raw -190bp."""
    base = FlatCurve(reference_date=ASOF, rate=0.001)
    shifted = shift_curve(base, _by_name(eu_scenarios("USD"))["parallel_down"])
    assert shifted.zero(0.0) == pytest.approx(-0.015, abs=1e-15)
    assert shifted.zero(10.0) == pytest.approx(-0.012, abs=1e-15)
    # The floor is maturity dependent: it binds harder at shorter tenors.
    assert shifted.zero(0.0) < shifted.zero(10.0)


def test_observed_rate_below_the_floor_is_applied_instead() -> None:
    """Article 3(7): if observed rates are below the floor, the lower observed
    rate applies — the floor never forces a shocked rate above the market."""
    base = FlatCurve(reference_date=ASOF, rate=-0.02)
    shifted = shift_curve(base, _by_name(eu_scenarios("USD"))["parallel_down"])
    assert shifted.zero(10.0) == pytest.approx(-0.02, abs=1e-15)


def test_up_shocks_are_not_affected_by_the_floor() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.001)
    shifted = shift_curve(base, _by_name(eu_scenarios("USD"))["parallel_up"])
    assert shifted.zero(10.0) == pytest.approx(0.021, abs=1e-15)


def test_floored_discount_factors_derive_from_the_floored_zero() -> None:
    """With a floor present, df must be exp(-zero*t) for the same floored zero,
    so pricing and zero-rate views cannot disagree."""
    base = FlatCurve(reference_date=ASOF, rate=0.001)
    shifted = shift_curve(base, _by_name(eu_scenarios("USD"))["parallel_down"])
    assert shifted.df(10.0) == pytest.approx(math.exp(-shifted.zero(10.0) * 10.0), rel=1e-15)
    assert shifted.df(0.0) == pytest.approx(math.exp(-shifted.zero(0.0) * 0.0), rel=1e-15)


def test_every_eu_scenario_carries_the_post_shock_floor() -> None:
    for scenario in eu_scenarios("USD"):
        assert scenario.floor is not None, scenario.name


def test_the_ad_hoc_parallel_primitive_has_no_floor() -> None:
    """parallel() is the unfloored shift primitive used by bump-and-reprice
    helpers; only the EU scenarios apply the Article 3(7) floor."""
    base = FlatCurve(reference_date=ASOF, rate=0.001)
    shifted = shift_curve(base, parallel(-0.02))
    assert shifted.zero(10.0) == pytest.approx(-0.019, abs=1e-15)


# ---------------------------------------------------------------------------
# Strict typed loader (behavioral test 1; SEC-02, QUANTRISK-13)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "match"),
    [
        # Incompatible types are rejected, never coerced.
        (_variant({"parallel_bp = 200": 'parallel_bp = "200"'}), "parallel_bp"),
        (_variant({"short_bp = 300": 'short_bp = "300"'}), "short_bp"),
        (_variant({"long_bp = 150": "long_bp = true"}), "long_bp"),
        # Fractional integer fields are rejected.
        (_variant({"parallel_bp = 200": "parallel_bp = 200.5"}), "parallel_bp"),
        # NaN and infinity are rejected.
        (_variant({"short_bp = 300": "short_bp = inf"}), "finite"),
        (_variant({"short_decay_years = 4.0": "short_decay_years = nan"}), "finite"),
        # Non-positive shock sizes are rejected.
        (_variant({"parallel_bp = 200": "parallel_bp = -200"}), "positive"),
        (_variant({"long_bp = 150": "long_bp = 0"}), "positive"),
        (_variant({"short_decay_years = 4.0": "short_decay_years = -1.0"}), "positive"),
        # Missing keys are rejected.
        (_variant({"short_bp = 300\n": ""}), "short_bp"),
        (_variant({"long_weight = 0.9\n": ""}), "long_weight"),
        (
            _variant(
                {'citation = "Commission Delegated Regulation (EU) 2024/856, Annex Part A"': ""}
            ),
            "citation",
        ),
        # Unknown keys are rejected, at every level.
        (
            _variant({"parallel_bp = 200": "parallel_bp = 200\nfoo = 1"}),
            "unknown",
        ),
        (_variant({"[shape]": "foo = 1\n\n[shape]"}), "unknown"),
        # The shape table and its decay are mandatory and typed.
        (
            '[currency.USD]\nparallel_bp = 200\nshort_bp = 300\nlong_bp = 150\ncitation = "x"\n',
            "shape",
        ),
        (_variant({"short_decay_years = 4.0": 'short_decay_years = "4"'}), "short_decay_years"),
        # Rotation weights are typed numbers.
        (_variant({"short_weight = -0.65": 'short_weight = "x"'}), "short_weight"),
        # Duplicate keys are malformed TOML, wrapped in the named error.
        (
            _variant(
                {
                    "[shape.steepener]\nshort_weight = -0.65": (
                        "[shape.steepener]\nshort_weight = -0.65\nshort_weight = -0.5"
                    )
                }
            ),
            "invalid TOML",
        ),
        # An empty currency code is not a usable row.
        (
            VALID_CONFIG
            + '\n[currency.""]\nparallel_bp = 1\nshort_bp = 1\nlong_bp = 1\ncitation = "x"\n',
            "currency",
        ),
    ],
)
def test_malformed_scenario_configs_raise_named_errors(
    tmp_path: Path, body: str, match: str
) -> None:
    _expect_rejected(tmp_path, body, match)


def test_a_valid_configuration_still_loads(tmp_path: Path) -> None:
    good = tmp_path / "scenarios.toml"
    good.write_text(VALID_CONFIG, encoding="utf-8")
    scenarios_tuple = eu_scenarios("USD", path=good)
    assert {s.name for s in scenarios_tuple} == EXPECTED_NAMES
    assert _by_name(scenarios_tuple)["parallel_up"].shift(1.0) == pytest.approx(0.02, rel=1e-12)


def test_a_config_missing_a_currency_raises_rather_than_defaulting(tmp_path: Path) -> None:
    bad = tmp_path / "scenarios.toml"
    bad.write_text(VALID_CONFIG, encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match="JPY"):
        eu_scenarios("JPY", path=bad)


def test_an_empty_currency_code_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "scenarios.toml"
    bad.write_text(
        VALID_CONFIG
        + '\n[currency.""]\nparallel_bp = 1\nshort_bp = 1\nlong_bp = 1\ncitation = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ScenarioConfigError, match="currency"):
        eu_scenarios("USD", path=bad)


def test_invalid_paths_raise_named_errors(tmp_path: Path) -> None:
    with pytest.raises(ScenarioConfigError, match="directory"):
        eu_scenarios("USD", path=tmp_path)
    with pytest.raises(ScenarioConfigError, match="no such file"):
        eu_scenarios("USD", path=tmp_path / "absent.toml")


def test_shocks_are_applied_to_a_curve_without_producing_negative_discount_factors() -> None:
    base = FlatCurve(reference_date=ASOF, rate=0.0025)

    for scenario in eu_scenarios("USD"):
        shifted = shift_curve(base, scenario)
        assert all(shifted.df(t) > 0.0 for t in (0.25, 1.0, 10.0, 30.0)), scenario.name


# ---------------------------------------------------------------------------
# Packaging and claims (behavioral tests 4 and 5; DOC-10, HYGIENE-01)
# ---------------------------------------------------------------------------


def test_config_loads_from_package_resources_without_a_checkout() -> None:
    """scenarios.toml ships inside the yieldcurve.risk package and loads through
    importlib.resources, so an installed wheel works with no checkout and the
    loader never walks a source tree looking for the file. The built-wheel gate
    is Task 25; this pins the importlib.resources code path in the installed
    package (behavioral test 4)."""
    resource = importlib.resources.files("yieldcurve.risk").joinpath("scenarios.toml")
    assert resource.is_file()
    config = load_scenarios()
    currencies = config["currency"]
    assert isinstance(currencies, dict)
    assert "USD" in currencies and "SEK" in currencies
    assert {s.name for s in eu_scenarios("USD")} == EXPECTED_NAMES


def test_module_and_config_describe_eu_2024_856_not_bcbs_d368() -> None:
    """Rename requirement: the module, the config and their docstrings describe
    EU 2024/856 scenarios for educational analysis; no BCBS-EBA / d368 /
    'standardised shocks' / 'current Basel' language remains (behavioral test 5
    scenario side)."""
    module_doc = inspect.getdoc(scenarios)
    assert module_doc is not None
    lowered = module_doc.lower()
    assert "2024/856" in lowered
    assert "supervisory outlier test" in lowered
    assert "regulatory reporting" in lowered
    for banned in ("bcbs-eba", "d368", "standardised shocks", "current basel"):
        assert banned not in lowered

    resource = importlib.resources.files("yieldcurve.risk").joinpath("scenarios.toml")
    text = resource.read_text(encoding="utf-8")
    assert "2024/856" in text
    for banned in ("d368", "bcbs", "gl/2018", "standardised"):
        assert banned not in text.lower()
