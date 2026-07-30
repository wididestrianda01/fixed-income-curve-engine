"""Dated CSV snapshots of market data.

Every market data adapter writes through this module, and every consumer reads
through it. That is what makes a clean clone of the repository runnable with no
API keys and no network: the data it needs is committed under
``data/snapshots/<date>/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_SNAPSHOT_ROOT = Path(__file__).resolve().parents[3] / "data" / "snapshots"

_FLOAT_FORMAT = "%.10g"


class MissingDatasetError(FileNotFoundError):
    """A requested dataset is absent from the snapshot directory."""


@dataclass(frozen=True)
class Snapshot:
    """A single dated directory of market data CSVs."""

    date: date
    root: Path = DEFAULT_SNAPSHOT_ROOT

    @property
    def directory(self) -> Path:
        return self.root / self.date.isoformat()

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.csv"

    def load(self, name: str) -> pd.DataFrame:
        target = self.path(name)
        if not target.exists():
            raise MissingDatasetError(
                f"Dataset {name!r} not found at {target}. "
                f"Available in this snapshot: {self.available()}. "
                "Rebuild a snapshot with: python -m yieldcurve.market.refresh"
            )
        return pd.read_csv(target)

    def save(self, name: str, frame: pd.DataFrame) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(name)
        frame.to_csv(target, index=False, float_format=_FLOAT_FORMAT)
        return target

    def available(self) -> tuple[str, ...]:
        if not self.directory.exists():
            return ()
        return tuple(sorted(p.stem for p in self.directory.glob("*.csv")))

    @classmethod
    def latest(cls, root: Path = DEFAULT_SNAPSHOT_ROOT) -> Snapshot:
        candidates = sorted(p.name for p in root.glob("*-*-*") if p.is_dir())
        if not candidates:
            raise MissingDatasetError(
                f"There are no snapshots under {root}. "
                "Build one with: python -m yieldcurve.market.refresh"
            )
        return cls(date=date.fromisoformat(candidates[-1]), root=root)
