"""The limitations document exists and covers what section 4.5 requires."""

from __future__ import annotations

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "hull-white-limitations.md"


def test_the_limitations_document_exists() -> None:
    assert DOC.exists()


@pytest.mark.parametrize(
    "topic",
    ["negative rates", "correlation", "smile", "mean reversion", "calibration"],
)
def test_every_required_limitation_is_covered(topic: str) -> None:
    text = DOC.read_text(encoding="utf-8").lower()

    assert topic in text


def test_the_scenario_generation_boundary_is_stated_explicitly() -> None:
    text = DOC.read_text(encoding="utf-8").lower()

    assert "not" in text and "scenario" in text


def test_the_negative_rate_probability_is_a_number_not_an_adjective() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "%" in text
