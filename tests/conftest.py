from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from curveengine.market.snapshot import DEFAULT_SNAPSHOT_ROOT, Snapshot

SNAPSHOT_DATE = date(2026, 7, 24)


@pytest.fixture(scope="session")
def snapshot_root() -> Path:
    return DEFAULT_SNAPSHOT_ROOT


@pytest.fixture(scope="session")
def snapshot() -> Snapshot:
    return Snapshot(date=SNAPSHOT_DATE, root=DEFAULT_SNAPSHOT_ROOT)
