"""Tab 2 — what a price is: the discounted sum of the flows, and nothing else."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import gov_bonds, sek_curveset
from app.state import AppState
from yieldcurve.curves.pricing import price, ytm
from yieldcurve.curves.protocol import curve_time

_BOND_KEY = "bond_index"


def bond_label(bond: object) -> str:
    # ponytail: getattr avoids importing FixedCouponBond for type hint,
    # but ruff B009 wants direct access — use it.
    coupon = bond.coupon  # type: ignore[attr-defined]
    maturity = bond.maturity  # type: ignore[attr-defined]
    return f"SGB {coupon * 100:.3g}% {maturity:%b-%Y}"


def render(state: AppState) -> None:
    st.subheader("Pricing")
    st.markdown(
        "A government bond book discounts on its own government curve, so `CurveSet.single` "
        "is the correct wrapping here — one curve doing both jobs. That is the pre-2008 "
        "arrangement, and for a sovereign holding its own paper it is still the right one. "
        "The Beyond the curve tab shows where it stops being right."
    )

    bonds = gov_bonds()
    index = st.selectbox(
        "Bond",
        options=range(len(bonds)),
        format_func=lambda i: bond_label(bonds[i]),
        key=_BOND_KEY,
    )
    bond = bonds[index]

    curves = sek_curveset(state.asof, state.method)
    result = price(bond, curves, state.asof)

    left, middle, right = st.columns(3)
    left.metric("Clean price", f"{result.clean:.6f}")
    middle.metric("Accrued", f"{result.accrued:.6f}")
    right.metric("Dirty price", f"{result.dirty:.6f}")
    st.caption("Per 100 face. Clean plus accrued is dirty, exactly, by construction.")

    st.metric("Yield to maturity", f"{ytm(bond, result.dirty, state.asof) * 100:.4f}%")
    st.caption(
        "Yield to maturity is a quoting device on the street convention, not a term "
        "structure. Its exponents are `w + k` — a first fractional period followed by whole "
        "coupon periods — not year fractions off the curve. It is the number two desks agree "
        "on to name a price, not the number that produced the price."
    )

    rows = []
    for flow in bond.cashflows(state.asof):
        t = curve_time(state.asof, flow.date)
        df = curves.discount.df(t)
        rows.append(
            {
                "Date": flow.date,
                "Amount": flow.amount,
                "Curve time (y)": t,
                "Discount factor": df,
                "PV": flow.amount * df,
            }
        )
    table = pd.DataFrame(rows)
    total = table["PV"].sum()
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        f"Flows sum to {total:.6f} against a dirty price of {result.dirty:.6f} — the visible "
        "proof that the pricer is doing nothing but discounting."
    )
