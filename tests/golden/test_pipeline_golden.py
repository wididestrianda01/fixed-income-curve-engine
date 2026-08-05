"""End-to-end regression: the whole pipeline against pinned values.

The pinned pipeline runs builders on their canonical log-linear discount-factor
default (no ``method=`` overlay), so the fixture tracks the calibration
contract the package actually ships. A value change must trace to a corrected
contract — bootstrap/interpolation, pricing, scenario, or risk — never to
cosmetic drift.

Regenerate deliberately, never reflexively:

    python scripts/write_golden.py

A diff here means a number moved. Find out why before you accept it — that is the entire
value of this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "pipeline_v1.json"
EXACT = 1e-10
STOCHASTIC_REL = 1e-6

_STOCHASTIC_PREFIXES = ("hullwhite.",)


@pytest.fixture(scope="module")
def expected() -> dict[str, float]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def actual() -> dict[str, float]:
    from scripts.write_golden import compute as _compute

    return _compute()  # type: ignore[no-any-return]


def test_the_golden_file_and_the_pipeline_agree_on_which_keys_exist(
    expected: dict[str, float], actual: dict[str, float]
) -> None:
    assert set(expected) == set(actual)


def test_every_pinned_value_still_holds(
    expected: dict[str, float], actual: dict[str, float]
) -> None:
    mismatches = []
    for key, want in expected.items():
        got = actual[key]
        if key.startswith(_STOCHASTIC_PREFIXES):
            ok = got == pytest.approx(want, rel=STOCHASTIC_REL)
        else:
            ok = got == pytest.approx(want, abs=EXACT)
        if not ok:
            mismatches.append(f"{key}: expected {want!r}, got {got!r}")
    assert not mismatches, "\n".join(mismatches)
