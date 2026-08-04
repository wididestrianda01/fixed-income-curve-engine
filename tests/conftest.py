from __future__ import annotations

import tomllib
from datetime import date
from importlib import resources

import pytest

from yieldcurve.curves.protocol import CurveSet, FlatCurve
from yieldcurve.market.snapshot import Snapshot


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
