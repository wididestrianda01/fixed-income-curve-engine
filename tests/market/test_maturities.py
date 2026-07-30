from datetime import date

import pytest

from curveengine.market.snapshot import Snapshot


class TestSnapshotMaturities:
    """Test Snapshot time-to-maturity calculations."""

    @pytest.fixture
    def snap(self) -> Snapshot:
        snap = Snapshot.new(date(2026, 7, 24))
        snap = snap.with_instrument(
            {"isin": "BOND1", "maturity": date(2027, 1, 24), "bid": 99.5, "ask": 100.5}
        )
        snap = snap.with_instrument(
            {"isin": "BOND2", "maturity": date(2028, 7, 24), "bid": 98.0, "ask": 102.0}
        )
        return snap

    def test_snapshot_time_to_maturity_years(self, snap: Snapshot) -> None:
        bond1 = snap.by_isin("BOND1")
        t = snap.time_to_maturity_years(bond1)
        assert abs(t - (184 / 365)) < 1e-9

    def test_snapshot_time_to_maturity_zero_at_maturity(self) -> None:
        snap = Snapshot.new(date(2026, 7, 24))
        snap = snap.with_instrument(
            {"isin": "BOND_MATURE", "maturity": date(2026, 7, 24), "bid": 100.0, "ask": 100.0}
        )
        bond = snap.by_isin("BOND_MATURE")
        t = snap.time_to_maturity_years(bond)
        assert t == 0.0

    def test_snapshot_time_to_maturity_fractional_year(self, snap: Snapshot) -> None:
        bond2 = snap.by_isin("BOND2")
        t = snap.time_to_maturity_years(bond2)
        # 2026-07-24 to 2028-07-24 is 731 days (leap year 2028)
        assert abs(t - (731 / 365)) < 1e-9
