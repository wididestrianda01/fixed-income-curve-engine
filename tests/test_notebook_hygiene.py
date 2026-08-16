"""Notebooks stay deterministic, output-bearing, source-synchronized, and offline.

Sources are the reviewable source of truth (design section 9); the committed
``.ipynb`` files must round-trip to ``notebooks/src/*.py`` exactly, execute in
order with clean outputs, and never touch the network.

Regeneration runs ``jupytext --to notebook --execute`` from the corrected
sources and requires a kernelspec that runs the project venv's interpreter:
jupytext resolves ``python3`` through JUPYTER_PATH, which includes
``.venv/share/jupyter/kernels`` when the venv is active (Tasks 20-23 used that
kernelspec; it is what produced the committed execution timestamps). Execution
must run under the network block in ``notebooks/_netblock``: ``notebooks/
conftest.py`` applies it automatically for ``pytest --nbmake``, and
regenerating by hand requires ``PYTHONPATH=notebooks/_netblock`` so the kernel
process imports the blocking ``sitecustomize`` at startup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jupytext
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def _code_source(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def _notebook_source_for(path: Path) -> Path:
    return REPO_ROOT / "notebooks" / "src" / f"{path.stem}.py"


def _strip_jupytext_header(text: str) -> str:
    """Drop the ``# ---`` fenced text-representation header jupytext prepends
    when writing percent-format .py (it records versions, not notebook code)."""
    lines = text.splitlines()
    try:
        start = lines.index("# ---")
        end = lines.index("# ---", start + 1)
    except ValueError:
        return text
    body = lines[end + 1 :]
    while body and not body[0].strip():
        body.pop(0)
    return "\n".join(body)


def _normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def test_every_source_has_a_rendered_notebook() -> None:
    """Each notebook source (``notebooks/src/*.py``) has a paired ``.ipynb``
    render, and no rendered notebook is orphaned from a source."""
    sources = {p.stem for p in (REPO_ROOT / "notebooks" / "src").glob("*.py")}
    rendered = {p.stem for p in NOTEBOOKS}
    assert rendered == sources


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
def test_notebook_source_and_notebook_are_in_sync(path: Path) -> None:
    """The committed .ipynb round-trips to its source: converting it back to
    percent-format .py (jupytext) reproduces notebooks/src/<name>.py exactly.
    Sources are the reviewable source of truth; a committed notebook that
    drifted from its source would show stale code or prose on GitHub and the
    next regeneration would silently overwrite someone's review."""
    source = _notebook_source_for(path)
    assert source.exists(), f"no source {source.name} for {path.name}; sources must not be deleted"
    converted = jupytext.writes(jupytext.read(path), fmt="py:percent")
    assert _normalize(_strip_jupytext_header(converted)) == _normalize(
        source.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_have_no_error_outputs(path: Path) -> None:
    """Committed outputs are clean: an error output means the notebook was
    committed broken, executed out of order, or is hiding a failure behind the
    surrounding prose."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    bad = [
        cell["id"]
        for cell in notebook["cells"]
        if any(o.get("output_type") == "error" for o in cell.get("outputs", []))
    ]
    assert not bad, f"error outputs in cells {bad}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_execute_in_order_with_a_sane_kernelspec(path: Path) -> None:
    """Cells were executed 1..N in notebook order against the declared python3
    kernelspec. A notebook executed out of order would carry stale
    intermediate values into later cells."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    kernelspec = notebook["metadata"]["kernelspec"]
    assert kernelspec["name"] == "python3"
    assert kernelspec["language"] == "python"
    code_cells = [
        c for c in notebook["cells"] if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]
    counts = [c["execution_count"] for c in code_cells]
    assert counts == list(range(1, len(counts) + 1)), counts


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_make_no_network_calls(path: Path) -> None:
    source = _code_source(path)

    for forbidden in ("requests.", "urlopen", "pandas_datareader", "yfinance"):
        assert forbidden not in source


def test_notebook_conftest_puts_the_netblock_on_kernel_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nbmake run must execute kernels offline: notebooks/conftest.py puts
    notebooks/_netblock (its sitecustomize) on PYTHONPATH before any kernel is
    spawned. Importing the conftest applies its real side effect; monkeypatch
    snapshots PYTHONPATH first so that side effect stays scoped to this test."""
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    conftest = REPO_ROOT / "notebooks" / "conftest.py"
    spec = importlib.util.spec_from_file_location("notebooks_conftest", conftest)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    netblock: Path = module._NETBLOCK  # type: ignore[attr-defined]
    assert str(netblock) in os.environ["PYTHONPATH"]


def test_notebook_execution_network_block_refuses_connections() -> None:
    """Prove the sitecustomize mechanism itself: a fresh interpreter with
    notebooks/_netblock on PYTHONPATH refuses any outbound connection with the
    block's RuntimeError (TQ-09). This is exactly what the nbmake kernel (and
    the regeneration path) inherit at startup; it cannot be exercised by
    patching this pytest process, whose ssl/asyncio are already initialized."""
    probe = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://127.0.0.1:9', timeout=2)\n"
        "except RuntimeError as exc:\n"
        "    assert 'network call blocked' in str(exc), str(exc)\n"
        "else:\n"
        "    raise SystemExit('network call was not blocked')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "notebooks" / "_netblock")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
