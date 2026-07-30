from datetime import date

from curveengine.market.snapshot import Snapshot


class TestSnapshot:
    """Test Snapshot initialization and properties."""

    def test_snapshot_new_creates_zero_instruments(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        assert snap.reference_date == date(2026, 7, 24)
        assert len(snap) == 0

    def test_snapshot_reference_date_is_immutable(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        assert snap.reference_date == date(2026, 7, 24)

    def test_snapshot_len_empty(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        assert len(snap) == 0

    def test_snapshot_len_after_add(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        snap = snap.with_instrument(
            {"isin": "TEST1", "maturity": date(2026, 8, 24), "bid": 99.0, "ask": 101.0}
        )
        assert len(snap) == 1

    def test_snapshot_with_instrument_returns_new_snapshot(self) -> None:
        snap1 = Snapshot.new(date(2026, 7, 24))
        snap2 = snap1.with_instrument(
            {"isin": "TEST1", "maturity": date(2026, 8, 24), "bid": 99.0, "ask": 101.0}
        )
        assert snap1 is not snap2
        assert len(snap1) == 0
        assert len(snap2) == 1

    def test_snapshot_instruments_returns_list_of_dicts(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        instr = {"isin": "TEST1", "maturity": date(2026, 8, 24), "bid": 99.0, "ask": 101.0}
        snap = snap.with_instrument(instr)
        instruments = snap.instruments()
        assert len(instruments) == 1
        assert instruments[0]["isin"] == "TEST1"

    def test_snapshot_instruments_returns_copy(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        snap = snap.with_instrument(
            {"isin": "TEST1", "maturity": date(2026, 8, 24), "bid": 99.0, "ask": 101.0}
        )
        instruments = snap.instruments()
        original_len = len(instruments)
        instruments.append(
            {"isin": "TEST2", "maturity": date(2026, 9, 24), "bid": 98.0, "ask": 102.0}
        )
        assert len(snap.instruments()) == original_len
