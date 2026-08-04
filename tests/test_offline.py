"""Offline guarantee: no code path reaches the network."""

from __future__ import annotations

import ast
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

import yieldcurve.market as _market
from tests.conftest import SNAPSHOT_DATE as ASOF
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.market.snapshot import Snapshot


@pytest.fixture(autouse=True)
def _block_network() -> Generator[None, None, None]:
    """Fail any socket connection attempted during the test."""
    with patch("socket.socket", side_effect=RuntimeError("network call blocked")):
        yield


def test_no_market_module_imports_requests_or_socket_at_module_scope() -> None:
    """The offline guarantee is structural: no module in ``yieldcurve.market``
    may import ``requests`` or ``socket`` at module scope, so the package has
    no network path to reach before any test fixture blocks it."""
    market_dir = Path(_market.__file__).resolve().parent
    forbidden = {"requests", "socket"}
    offenders: list[str] = []
    for path in sorted(market_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = {alias.name.split(".")[0] for alias in node.names}
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
            if names & forbidden:
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders


def test_load_snapshot_without_network() -> None:
    snapshot = Snapshot(date=ASOF)
    assert snapshot.available()  # type: ignore[arg-type]


def test_default_snapshot_loads_every_dataset_without_network() -> None:
    """Every packaged dataset is readable through the default snapshot offline."""
    snapshot = Snapshot(date=ASOF)

    for name in snapshot.available():
        assert not snapshot.load(name).empty


def test_default_snapshot_never_opens_a_socket() -> None:
    """Loading the packaged snapshot performs no network calls at socket level."""
    with patch("socket.socket", side_effect=RuntimeError("socket blocked")):
        snapshot = Snapshot(date=ASOF)
        for name in snapshot.available():
            snapshot.load(name)

        from yieldcurve.curves.build import usd_curveset

        curves = usd_curveset(snapshot, ASOF)
        assert curves.discount.zero(5.0) > 0.0
        assert curves.forecast_for("3M").zero(5.0) > 0.0


def test_sek_curve_without_network() -> None:
    from yieldcurve.curves.build import sek_government_curve

    curve = sek_government_curve(Snapshot(date=ASOF), ASOF)
    assert curve.zero(10.0) > 0.0


def test_sek_curveset_without_network() -> None:
    from yieldcurve.curves.build import sek_government_curve

    curves = CurveSet.single(sek_government_curve(Snapshot(date=ASOF), ASOF))
    assert curves.discount.zero(5.0) > 0.0


def test_usd_curves_without_network() -> None:
    from yieldcurve.curves.build import usd_curveset

    curves = usd_curveset(Snapshot(date=ASOF), ASOF)
    assert curves.discount.zero(5.0) > 0.0


def test_cmt_history_without_network() -> None:
    history = Snapshot(date=ASOF).load("fred_treasury_cmt_history")
    assert len(history) > 0


def test_portfolio_without_network() -> None:
    from pathlib import Path

    from yieldcurve.risk.portfolio import Portfolio

    path = Path(__file__).resolve().parents[1] / "data" / "demo_portfolio.toml"
    book = Portfolio.from_toml(path)
    assert len(book.positions) == 6


def test_gov_bonds_without_network() -> None:
    bonds = _gov_bonds_raw()
    assert len(bonds) > 0


def test_pnl_sample_without_network() -> None:
    from yieldcurve.risk.pca import daily_changes

    history = Snapshot(date=ASOF).load("fred_treasury_cmt_history")
    changes, _tenors = daily_changes(history)
    assert changes.shape[0] > 0


def test_atm_swaption_grid_illustrative_without_network() -> None:
    from yieldcurve.curves.build import usd_ois_curve
    from yieldcurve.models.hullwhite import atm_swaption_grid

    curve = usd_ois_curve(Snapshot(date=ASOF), ASOF)
    swaptions, vols = atm_swaption_grid(
        Snapshot(date=ASOF), ASOF, curve, dataset="illustrative_swaption_vols"
    )
    assert len(swaptions) == len(vols)
    assert all(v > 0.0 for v in vols)


def _gov_bonds_raw() -> tuple:  # type: ignore[type-arg]
    from datetime import timedelta

    import pandas as pd

    from yieldcurve.calendars import SwedenCalendar
    from yieldcurve.conventions import BusinessDayConvention, DayCount
    from yieldcurve.instruments import FixedCouponBond

    frame = Snapshot(date=ASOF).load("riksgalden_gov_bonds")
    horizon = ASOF + timedelta(days=int(10.0 * 365.25))
    bonds = []
    for row in frame.itertuples():
        maturity = pd.Timestamp(row.maturity_date).date()  # type: ignore[arg-type]
        if maturity > horizon:
            continue
        bonds.append(
            FixedCouponBond(
                issue=pd.Timestamp(row.issue_date).date(),  # type: ignore[arg-type]
                maturity=maturity,
                coupon=float(row.coupon),  # type: ignore[arg-type]
                frequency=1,
                day_count=DayCount.THIRTY_360_BOND,
                calendar=SwedenCalendar(),
                bdc=BusinessDayConvention.FOLLOWING,
            )
        )
    return tuple(sorted(bonds, key=lambda b: b.maturity))
