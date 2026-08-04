"""The frozen market-data snapshot, packaged as read-only resources.

The one audited snapshot (2026-07-24) ships under ``yieldcurve.data`` and is
loaded through ``importlib.resources``, so a source checkout and an installed
wheel behave identically: no network, and no fallback to a checkout path. The
default :class:`Snapshot` is read-only; writing a snapshot requires an explicit
external filesystem root.

Every dataset name must match the strict identifier grammar ``[a-z][a-z0-9_]*``,
so a name can never escape the snapshot directory (no separators, traversal,
absolute paths, or odd characters).
"""

from __future__ import annotations

import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

_PACKAGE = "yieldcurve.data"
_MANIFEST_NAME = "snapshot_manifest.toml"
_DATASET_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_FLOAT_FORMAT = "%.10g"


class MissingDatasetError(FileNotFoundError):
    """A requested dataset is absent, or the packaged snapshot is unusable.

    The class predates the manifest and also covers packaged-manifest problems
    (missing or invalid manifest, a snapshot-date mismatch, or a manifest
    requested on an external root), which share the same remedy: the packaged
    snapshot is incomplete or unusable. It deliberately keeps one name and
    subclasses :class:`FileNotFoundError`: ``app.py`` catches this exact class
    in its error-rendering path (``RENDER_ERRORS``), so renaming it would
    silently drop the app's handling of snapshot failures.
    """


class DatasetNameError(ValueError):
    """A dataset name does not match the strict identifier grammar."""


class ReadOnlySnapshotError(ValueError):
    """``save`` was called on the read-only packaged snapshot."""


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _DATASET_NAME_RE.fullmatch(name):
        raise DatasetNameError(
            f"dataset name {name!r} does not match the strict identifier grammar "
            r"[a-z][a-z0-9_]* and cannot name a snapshot file"
        )


def _fail_packaged(source: str, message: str) -> NoReturn:
    raise MissingDatasetError(f"packaged snapshot {source}: {message}")


@lru_cache(maxsize=1)
def _load_packaged_manifest() -> dict[str, Any]:
    """Parse the packaged ``snapshot_manifest.toml`` resource, once per process."""
    resource = resources.files(_PACKAGE).joinpath(_MANIFEST_NAME)
    try:
        handle = resource.open("rb")
    except (FileNotFoundError, KeyError, OSError):
        _fail_packaged("manifest", "is missing; reinstall the package")
    with handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            _fail_packaged("manifest", f"is invalid TOML: {exc}")


def _packaged_manifest() -> dict[str, Any]:
    """The packaged manifest, parsed once and returned fresh on every call."""
    return deepcopy(_load_packaged_manifest())


def _manifest_datasets(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        _fail_packaged("manifest", "has no [datasets] table")
    return {name: block for name, block in datasets.items() if isinstance(block, dict)}


def _manifest_snapshot_date(manifest: dict[str, Any]) -> date:
    raw = manifest.get("snapshot_date")
    try:
        return date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        _fail_packaged("manifest", "snapshot_date is not an ISO date")


@dataclass(frozen=True)
class Snapshot:
    """A dated market-data snapshot: the packaged resources, or an external root.

    ``root=None`` selects the packaged resources under ``yieldcurve.data``,
    discovered through ``importlib.resources`` and therefore identical in a
    source checkout and an installed wheel. The packaged snapshot is read-only:
    ``save`` raises :class:`ReadOnlySnapshotError`. Pass an explicit external
    root to write ``<root>/<date>/<name>.csv``.
    """

    date: date
    root: Path | None = None

    def __post_init__(self) -> None:
        if self.root is None:
            packaged = _manifest_snapshot_date(_packaged_manifest())
            if self.date != packaged:
                raise MissingDatasetError(
                    f"the packaged snapshot is dated {packaged.isoformat()}, not "
                    f"{self.date.isoformat()}; construct Snapshot(date={packaged.isoformat()!r}) "
                    "or pass an explicit external root for another date"
                )

    def _external_target(self, name: str) -> Path:
        if self.root is None:  # guarded by load/save, which check root first
            raise ValueError(
                "internal error: an external snapshot target requires an explicit root"
            )
        return self.root / self.date.isoformat() / f"{name}.csv"

    @property
    def manifest(self) -> dict[str, Any]:
        """The machine-readable snapshot manifest (packaged resources only)."""
        if self.root is not None:
            raise MissingDatasetError(
                "external snapshot roots are caller-managed filesystem directories "
                "and carry no manifest; only the packaged snapshot has one"
            )
        return _packaged_manifest()

    def load(self, name: str) -> pd.DataFrame:
        """Read one dataset as a DataFrame (``#`` comment lines skipped)."""
        _validate_name(name)
        if self.root is None:
            resource = resources.files(_PACKAGE).joinpath(f"{name}.csv")
            try:
                handle = resource.open("r")
            except (FileNotFoundError, KeyError, OSError):
                raise MissingDatasetError(
                    f"Dataset {name!r} is not in the packaged snapshot dated "
                    f"{self.date.isoformat()}. Available: {self.available()}."
                ) from None
            with handle:
                return pd.read_csv(handle, comment="#")
        target = self._external_target(name)
        if not target.exists():
            raise MissingDatasetError(
                f"Dataset {name!r} not found at {target}. "
                f"Available in this snapshot: {self.available()}."
            )
        return pd.read_csv(target, comment="#")

    def save(self, name: str, frame: pd.DataFrame) -> Path:
        """Write ``<root>/<date>/<name>.csv`` on an explicit external root.

        Raises:
            ReadOnlySnapshotError: on the packaged (default) snapshot.
            DatasetNameError: if ``name`` does not match the strict grammar.
        """
        _validate_name(name)
        if self.root is None:
            raise ReadOnlySnapshotError(
                "the packaged snapshot is read-only; save() requires an explicit "
                "external root: Snapshot(date=..., root=Path(...))"
            )
        target = self._external_target(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, float_format=_FLOAT_FORMAT)
        return target

    def available(self) -> tuple[str, ...]:
        """Dataset names in this snapshot, sorted, without the ``.csv`` suffix."""
        if self.root is None:
            return tuple(sorted(_manifest_datasets(_packaged_manifest())))
        directory = self.root / self.date.isoformat()
        if not directory.exists():
            return ()
        return tuple(sorted(p.stem for p in directory.glob("*.csv")))
