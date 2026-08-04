"""Installed-wheel contracts: isolated import/quickstart, metadata version,
and the app dependency group's independence from developer tooling.

These tests build throwaway virtual environments with ``uv``. They are the
Task 13 packaging gate (design spec section 7): the wheel alone must support
importing every public module, listing packaged datasets, building
representative curves, loading scenarios, and pricing a bond -- with no
checkout on the import path.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRS = (REPO_ROOT / "dist-review", REPO_ROOT / "dist")


def find_built_wheel() -> Path:
    """The newest wheel under dist-review/ or dist/; fails with guidance."""
    for directory in ARTIFACT_DIRS:
        candidates = sorted(directory.glob("yieldcurve-*.whl"))
        if candidates:
            return candidates[-1]
    raise AssertionError(
        "no built wheel found; run `uv build --out-dir dist-review` first "
        f"(looked in {[str(d) for d in ARTIFACT_DIRS]})"
    )


# The isolated-wheel quickstart (design spec section 7, verbatim contract):
# imports every public module, lists packaged datasets, builds representative
# SEK/USD curves, loads EU scenarios, and prices a bond. It runs from a
# neutral directory in a wheel-only virtual environment, so any missing
# packaged resource or checkout fallback fails the run.
QUICKSTART = textwrap.dedent(
    """\
    import importlib.metadata
    import pkgutil
    import tomllib
    from datetime import date
    from importlib import resources

    import yieldcurve

    # HYGIENE-07: installed metadata version equals the package version.
    assert importlib.metadata.version("yieldcurve") == yieldcurve.__version__

    # Every public module imports from the wheel.
    for module in pkgutil.walk_packages(
        yieldcurve.__path__, yieldcurve.__name__ + "."
    ):
        __import__(module.name)

    # The packaged snapshot lists its datasets.
    from yieldcurve.market.snapshot import Snapshot

    manifest = tomllib.loads(
        resources.files("yieldcurve.data")
        .joinpath("snapshot_manifest.toml")
        .read_text(encoding="utf-8")
    )
    asof = date.fromisoformat(manifest["snapshot_date"])
    snapshot = Snapshot(date=asof)
    available = snapshot.available()
    assert available, "packaged snapshot must list datasets"
    assert "fred_treasury_cmt" in available
    assert "usd_ois_swaps" in available

    # Representative SEK and USD curves from packaged data.
    from yieldcurve.curves.build import sek_government_curve, usd_curveset

    sek = sek_government_curve(snapshot, asof)
    assert sek.zero(10.0) > 0.0
    usd = usd_curveset(snapshot, asof)
    assert usd.discount.df(1.0) > 0.0

    # EU scenarios load from the packaged scenarios.toml.
    from yieldcurve.risk.scenarios import eu_scenarios, load_scenarios

    config = load_scenarios()
    assert config["currency"]
    scenarios = eu_scenarios("SEK")
    assert len(scenarios) == 6

    # A bond prices off the wheel-built curves.
    from yieldcurve.curves.pricing import price
    from yieldcurve.instruments import Bill

    bill = Bill(maturity=date(asof.year + 1, asof.month, asof.day))
    result = price(bill, usd, asof=asof)
    assert result.clean > 0.0

    print("isolated-wheel quickstart OK")
    """
)


def _run_uv(*args: str) -> None:
    subprocess.run(
        ["uv", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture(scope="module")
def wheel_venv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway virtual environment with only the built wheel installed."""
    venv_dir = tmp_path_factory.mktemp("wheel-venv")
    _run_uv("venv", "--python", "3.12", str(venv_dir))
    _run_uv("pip", "install", "--python", str(venv_dir), str(find_built_wheel()))
    return venv_dir


@pytest.fixture(scope="module")
def app_venv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway virtual environment with the project's app extra only:
    core dependencies plus the app group, no developer tooling."""
    venv_dir = tmp_path_factory.mktemp("app-venv")
    _run_uv("venv", "--python", "3.12", str(venv_dir))
    _run_uv("pip", "install", "--python", str(venv_dir), "-e", ".[app]")
    return venv_dir


@pytest.mark.slow
def test_isolated_wheel_imports_and_quickstarts(wheel_venv: Path, tmp_path: Path) -> None:
    """The wheel alone -- no checkout on the path -- satisfies the section 7
    isolated-environment contract (DOC-10, HYGIENE-01)."""
    env = {**os.environ, "PYTHONPATH": ""}
    result = subprocess.run(
        [str(wheel_venv / "bin" / "python"), "-c", QUICKSTART],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "isolated-wheel quickstart OK" in result.stdout


@pytest.mark.slow
def test_app_dependency_group_installs_independently_of_dev_tooling(
    app_venv: Path, tmp_path: Path
) -> None:
    """The app group resolves and installs without developer tooling: the app's
    third-party imports work and pytest/ruff/mypy are absent (TQ-12)."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    program = (
        "import importlib.util, plotly, streamlit, yieldcurve;"
        "tools = ('pytest', 'ruff', 'mypy');"
        "present = [t for t in tools if importlib.util.find_spec(t) is not None];"
        "assert not present, present;"
        "print('app group OK')"
    )
    result = subprocess.run(
        [str(app_venv / "bin" / "python"), "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "app group OK" in result.stdout

    # The app itself imports with only core + app dependencies installed.
    result = subprocess.run(
        [str(app_venv / "bin" / "python"), "-c", "import app; print('app import OK')"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "app import OK" in result.stdout
