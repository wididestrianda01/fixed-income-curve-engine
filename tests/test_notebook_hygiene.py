"""Notebooks stay deterministic, output-bearing, and free of business logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _code_source(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def test_all_six_notebooks_exist() -> None:
    assert len(NOTEBOOKS) == 6


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_are_deterministic(path: Path) -> None:
    """A notebook that reads the wall clock produces a different output every
    run, which makes the committed outputs worthless as a regression signal and
    fills every diff with noise."""
    source = _code_source(path)

    assert "datetime.now" not in source
    assert "date.today" not in source
    assert "np.random.seed" not in source  # legacy global RNG; use default_rng(seed)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_seed_every_generator(path: Path) -> None:
    source = _code_source(path)

    if "default_rng" in source:
        assert "default_rng(" in source and "default_rng()" not in source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_import_the_package(path: Path) -> None:
    assert "yieldcurve" in _code_source(path)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_define_no_business_logic(path: Path) -> None:
    """Cells may reshape data for a plot. They may not define pricing or
    curve-building functions: that code would be untested and uncovered, and
    the notebook would silently diverge from the package it demonstrates."""
    source = _code_source(path)

    assert "def price" not in source
    assert "def bootstrap" not in source
    assert "class " not in source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_have_committed_outputs(path: Path) -> None:
    """GitHub is where these get read, and it does not execute anything. A
    stripped notebook is a wall of code with no results."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).strip()]

    assert any(cell.get("outputs") for cell in code_cells)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_open_with_a_markdown_explanation(path: Path) -> None:
    """A notebook whose first cell is an import block tells a reader nothing
    about what they are looking at."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]

    assert cells[0]["cell_type"] == "markdown"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_make_no_network_calls(path: Path) -> None:
    source = _code_source(path)

    for forbidden in ("requests.", "urlopen", "pandas_datareader", "yfinance"):
        assert forbidden not in source
