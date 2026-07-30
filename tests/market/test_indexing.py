from datetime import date

import pytest

from curveengine.market.snapshot import Snapshot


class TestSnapshotIndexing:
    """Test Snapshot indexing and filtering."""

    @pytest.fixture
    def snap_with_data(self) -> Snapshot:
        snap = Snapshot.new(date(2026, 7, 24))
        snap = snap.with_instrument(
            {"isin": "ISIN1", "maturity": date(2026, 9, 24), "bid": 99.5, "ask": 100.5}
        )
        snap = snap.with_instrument(
            {"isin": "ISIN2", "maturity": date(2027, 7, 24), "bid": 98.0, "ask": 102.0}
        )
        snap = snap.with_instrument(
            {"isin": "ISIN3", "maturity": date(2028, 7, 24), "bid": 97.0, "ask": 103.0}
        )
        return snap

    def test_snapshot_get_by_isin(self, snap_with_data: Snapshot) -> None:
        instr = snap_with_data.by_isin("ISIN1")
        assert instr["isin"] == "ISIN1"
        assert instr["maturity"] == date(2026, 9, 24)

    def test_snapshot_get_by_isin_missing_raises_keyerror(self, snap_with_data: Snapshot) -> None:
        with pytest.raises(KeyError):
            snap_with_data.by_isin("MISSING")

    def test_snapshot_filter_by_maturity_range(self, snap_with_data: Snapshot) -> None:
        filtered = snap_with_data.filter_by_maturity(date(2026, 8, 1), date(2027, 8, 1))
        assert len(filtered) == 2
        isins = {i["isin"] for i in filtered}
        assert isins == {"ISIN1", "ISIN2"}

    def test_snapshot_filter_by_maturity_inclusive(self, snap_with_data: Snapshot) -> None:
        filtered = snap_with_data.filter_by_maturity(date(2026, 9, 24), date(2027, 7, 24))
        assert len(filtered) == 2

    def test_snapshot_filter_by_maturity_empty(self, snap_with_data: Snapshot) -> None:
        filtered = snap_with_data.filter_by_maturity(date(2030, 1, 1), date(2030, 12, 31))
        assert len(filtered) == 0

    def test_snapshot_has_isin(self, snap_with_data: Snapshot) -> None:
        assert snap_with_data.has_isin("ISIN1") is True
        assert snap_with_data.has_isin("MISSING") is False
