from __future__ import annotations

import tomllib
from datetime import date
from importlib import resources
from pathlib import Path

import pytest

from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.market.snapshot import Snapshot

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


def find_built_sdist() -> Path:
    """The newest sdist under dist-review/ or dist/; fails with guidance."""
    for directory in ARTIFACT_DIRS:
        candidates = sorted(directory.glob("yieldcurve-*.tar.gz"))
        if candidates:
            return candidates[-1]
    raise AssertionError(
        "no built sdist found; run `uv build --out-dir dist-review` first "
        f"(looked in {[str(d) for d in ARTIFACT_DIRS]})"
    )


def _packaged_snapshot_date() -> date:
    """The packaged manifest's ``snapshot_date`` — the snapshot's single as-of."""
    resource = resources.files("yieldcurve.data").joinpath("snapshot_manifest.toml")
    with resource.open("rb") as handle:
        return date.fromisoformat(tomllib.load(handle)["snapshot_date"])


SNAPSHOT_DATE = _packaged_snapshot_date()


@pytest.fixture(scope="session")
def snapshot() -> Snapshot:
    return Snapshot(date=SNAPSHOT_DATE)


@pytest.fixture
def flat_curves() -> CurveSet:
    """A 3% flat continuously compounded single-curve set at SNAPSHOT_DATE."""
    return CurveSet.single(FlatCurve(reference_date=SNAPSHOT_DATE, rate=0.03))
