"""Generate the golden-file pipeline regression values.

Run when a deliberate change moves the numbers:

    python scripts/write_golden.py

A diff in tests/golden/pipeline_v1.json means a number moved. Find out why before
you accept it — that is the entire value of this file.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from yieldcurve.calendars import SwedenCalendar
from yieldcurve.conventions import BusinessDayConvention, DayCount
from yieldcurve.curves.build import (
    government_swap_basis,
    sek_government_curve,
    usd_ois_curve,
)
from yieldcurve.curves.interpolation import InterpMethod
from yieldcurve.curves.parametric import Svensson
from yieldcurve.curves.pricing import price, ytm
from yieldcurve.curves.protocol import CurveSet
from yieldcurve.instruments import FixedCouponBond
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.models.hullwhite import atm_swaption_grid, calibrate
from yieldcurve.risk.pca import daily_changes, fit_pca
from yieldcurve.risk.portfolio import Portfolio, delta_eve, present_value
from yieldcurve.risk.scenarios import eu_scenarios
from yieldcurve.risk.sensitivities import dv01, effective_duration

ASOF = date(2026, 7, 24)
METHOD = InterpMethod.MONOTONE_CONVEX
SNAPSHOT = Snapshot(date=ASOF)
GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "pipeline_v1.json"


def compute() -> dict[str, float]:
    result: dict[str, float] = {}

    # SEK curve
    sek = sek_government_curve(SNAPSHOT, ASOF, method=METHOD)
    result["sek.zero.2y"] = sek.zero(2.0)
    result["sek.zero.10y"] = sek.zero(10.0)

    times = np.linspace(0.05, 10.0, 400)
    zeros = np.array([sek.zero(t) for t in times])
    fit = Svensson.fit(times.tolist(), zeros.tolist(), reference_date=ASOF)
    result["sek.svensson.rmse_bp"] = fit.rmse * 10_000.0

    # Bond pricing
    bond_2031 = FixedCouponBond(
        issue=date(2020, 3, 27),
        maturity=date(2031, 5, 12),
        coupon=0.00125,
        frequency=1,
        day_count=DayCount.THIRTY_360_BOND,
        calendar=SwedenCalendar(),
        bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
    )
    curves = CurveSet.single(sek)
    price_result = price(bond_2031, curves, ASOF)
    result["bond.2031.dirty"] = price_result.dirty
    result["bond.2031.clean"] = price_result.clean
    result["bond.2031.ytm"] = ytm(bond_2031, price_result.dirty, ASOF)
    result["bond.2031.dv01"] = dv01(bond_2031, curves, ASOF)
    result["bond.2031.effective_duration"] = effective_duration(bond_2031, curves, ASOF)

    # Portfolio
    portfolio_path = Path(__file__).resolve().parents[1] / "data" / "demo_portfolio.toml"
    book = Portfolio.from_toml(portfolio_path)
    result["portfolio.pv"] = present_value(book, curves, ASOF)

    for scenario in eu_scenarios("SEK"):
        key = f"portfolio.eve.{scenario.name}"
        result[key] = delta_eve(book, curves, ASOF, scenario)

    # USD basis
    basis = government_swap_basis(SNAPSHOT, ASOF, (10.0,), method=METHOD)
    result["usd.basis.10y"] = basis[10.0]

    # PCA
    history = SNAPSHOT.load("fred_treasury_cmt_history")
    changes, tenors = daily_changes(history)
    pca_result = fit_pca(changes, tenors, n_components=3)
    for i, ratio in enumerate(pca_result.explained_variance_ratio):
        result[f"pca.explained.{i}"] = ratio

    # Hull-White
    usd_curve = usd_ois_curve(SNAPSHOT, ASOF)
    swaptions, vols = atm_swaption_grid(
        SNAPSHOT, ASOF, usd_curve, dataset="illustrative_swaption_vols"
    )
    hw_result = calibrate(usd_curve, swaptions, vols, ASOF)
    result["hullwhite.a"] = hw_result.a
    result["hullwhite.sigma"] = hw_result.sigma
    result["hullwhite.rmse_vol_bp"] = hw_result.rmse_vol_bp

    return dict(sorted(result.items()))


def main() -> None:
    result = compute()
    GOLDEN.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} keys to {GOLDEN}")


if __name__ == "__main__":
    main()
