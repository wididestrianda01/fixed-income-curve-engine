"""The Snapshot contract: packaged read-only default, strict names, explicit roots.

Behavioral tests for Task 10:
- the default snapshot is discovered through ``importlib.resources`` (source and wheel);
- dataset identifiers reject absolute paths, separators, traversal, empty names
  and invalid characters;
- default resources are read-only; saving requires an explicit external root.
"""

from __future__ import annotations

from datetime import date
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest

from yieldcurve.market.snapshot import (
    DatasetNameError,
    MissingDatasetError,
    ReadOnlySnapshotError,
    Snapshot,
)

PACKAGED_DATE = date(2026, 7, 24)


def test_default_snapshot_is_discoverable_through_importlib_resources() -> None:
    """The packaged snapshot is readable with no root, via yieldcurve.data."""
    package = resources.files("yieldcurve.data")
    snapshot = Snapshot(date=PACKAGED_DATE)

    assert snapshot.available()
    for name in snapshot.available():
        assert package.joinpath(f"{name}.csv").is_file()
        frame = snapshot.load(name)
        assert not frame.empty


def test_packaged_snapshot_date_must_match_the_manifest() -> None:
    """The packaged snapshot is dated 2026-07-24; a different as-of is rejected."""
    with pytest.raises(MissingDatasetError, match="2026-07-24"):
        Snapshot(date=date(2026, 7, 25))


def test_default_snapshot_is_read_only() -> None:
    """Saving through the packaged snapshot is refused: no silent checkout write."""
    snapshot = Snapshot(date=PACKAGED_DATE)

    with pytest.raises(ReadOnlySnapshotError, match="external root"):
        snapshot.save("new_dataset", pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a b",
        "a-b",
        "a.b",
        "A",
        "1abc",
        "a.csv",
        "/etc/passwd",
        "../escape",
        "..\\escape",
        "a:b",
        "a,b",
    ],
)
def test_dataset_names_reject_paths_and_invalid_characters(name: str) -> None:
    """Names are strict identifiers: no separators, traversal, or odd characters."""
    snapshot = Snapshot(date=PACKAGED_DATE)

    with pytest.raises(DatasetNameError) as excinfo:
        snapshot.load(name)
    assert repr(name) in str(excinfo.value)


def test_save_validates_dataset_names_before_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    snapshot = Snapshot(date=PACKAGED_DATE, root=tmp_path)

    with pytest.raises(DatasetNameError):
        snapshot.save("../escape", pd.DataFrame({"x": [1]}))

    assert not (tmp_path / PACKAGED_DATE.isoformat()).exists()


def test_save_then_load_round_trips_with_an_external_root(tmp_path: Path) -> None:
    snapshot = Snapshot(date=PACKAGED_DATE, root=tmp_path)
    frame = pd.DataFrame({"tenor": ["3M", "6M"], "rate": [0.0231, 0.0245]})

    written = snapshot.save("riksbank_bills", frame)

    assert written == tmp_path / "2026-07-24" / "riksbank_bills.csv"
    pd.testing.assert_frame_equal(snapshot.load("riksbank_bills"), frame)


def test_load_missing_dataset_raises_a_named_error(tmp_path: Path) -> None:
    snapshot = Snapshot(date=PACKAGED_DATE, root=tmp_path)

    with pytest.raises(MissingDatasetError, match="not found"):
        snapshot.load("riksbank_bills")


def test_available_lists_dataset_names_without_extensions(tmp_path: Path) -> None:
    snapshot = Snapshot(date=PACKAGED_DATE, root=tmp_path)
    snapshot.save("b_second", pd.DataFrame({"x": [1]}))
    snapshot.save("a_first", pd.DataFrame({"x": [1]}))

    assert snapshot.available() == ("a_first", "b_second")


def test_load_skips_comment_preamble(tmp_path: Path) -> None:
    root = tmp_path / PACKAGED_DATE.isoformat()
    root.mkdir()
    (root / "commented.csv").write_text(
        "# provenance line one\n# provenance line two\na,b\n1,2\n", encoding="utf-8"
    )

    frame = Snapshot(date=PACKAGED_DATE, root=tmp_path).load("commented")

    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (1, 2)
