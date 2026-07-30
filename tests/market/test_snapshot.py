from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from curveengine.market.snapshot import MissingDatasetError, Snapshot


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    snapshot = Snapshot(date=date(2026, 7, 24), root=tmp_path)
    frame = pd.DataFrame({"tenor": ["3M", "6M"], "rate": [0.0231, 0.0245]})

    written = snapshot.save("riksbank_bills", frame)

    assert written == tmp_path / "2026-07-24" / "riksbank_bills.csv"
    pd.testing.assert_frame_equal(snapshot.load("riksbank_bills"), frame)


def test_load_missing_dataset_names_the_refresh_command(tmp_path: Path) -> None:
    snapshot = Snapshot(date=date(2026, 7, 24), root=tmp_path)

    with pytest.raises(MissingDatasetError, match=r"curveengine\.market\.refresh"):
        snapshot.load("riksbank_bills")


def test_available_lists_dataset_names_without_extensions(tmp_path: Path) -> None:
    snapshot = Snapshot(date=date(2026, 7, 24), root=tmp_path)
    snapshot.save("b_second", pd.DataFrame({"x": [1]}))
    snapshot.save("a_first", pd.DataFrame({"x": [1]}))

    assert snapshot.available() == ("a_first", "b_second")


def test_latest_picks_the_most_recent_dated_directory(tmp_path: Path) -> None:
    for day in (date(2026, 7, 17), date(2026, 7, 24), date(2026, 6, 30)):
        Snapshot(date=day, root=tmp_path).save("x", pd.DataFrame({"x": [1]}))

    assert Snapshot.latest(root=tmp_path).date == date(2026, 7, 24)


def test_latest_on_an_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingDatasetError, match="no snapshots"):
        Snapshot.latest(root=tmp_path)
