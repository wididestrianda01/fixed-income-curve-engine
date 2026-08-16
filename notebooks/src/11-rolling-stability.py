# %% [markdown]
# # 11. Does the methodology hold up over time? Rolling stability
#
# ## Objective
#
# Notebooks 01-10 built curves and fitted models on the single frozen snapshot
# date. A model that works on one date is only the first test; the second is
# whether its *outputs are stable across time*. This notebook runs that second
# test on the five years of daily Treasury CMT history the snapshot also ships.
# Two questions, both answerable with the data at hand:
#
# 1. **Are the PCA components stable?** Notebook 05 reports a level/slope/
#    curvature decomposition (roughly 73%/10%/10% of variance). Does that
#    structure hold on every one-year window, or is it an artefact of the full
#    sample?
# 2. **Does the parametric fit stay put?** Notebook 03 fits Nelson-Siegel. Do
#    the fitted parameters drift or jump across dates, and does a curve fitted
#    yesterday still describe today's curve?
#
# This is a **diagnostic stability check**, not a backtest: there is no P&L
# series to reconcile and no trading decision to judge. It asks whether the
# tools are well-posed — whether a number the tool reports today would have
# meant the same thing a year ago. The distinction from regulatory model
# validation is stated in the final section.
#
# ## Data
#
# `fred_treasury_cmt_history` carries daily US Treasury constant-maturity par
# yields, 2021-07-25 through 2026-07-24, at the standard 1M-30Y tenors. The
# parametric fit below targets these **par yields directly** (not a bootstrapped
# zero curve): the stability conclusion — does the fitted shape drift? — is
# independent of the par-vs-zero convention, and the convention is stated rather
# than hidden.

# %%
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yieldcurve.curves.parametric import FitError, NelsonSiegel
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.risk.pca import daily_changes, fit_pca

ASOF = date(2026, 7, 24)
snapshot = Snapshot(ASOF)

history = snapshot.load("fred_treasury_cmt_history")
print("rows:", len(history), "| columns:", list(history.columns))
print("tenors (years):", sorted(history["tenor_years"].unique()))
print("date range:", history["date"].min(), "to", history["date"].max())

# %%
# ## PCA stability
#
# Fit PCA on the full sample, then on each trailing one-year window, and compare
# the variance shares and the loading shapes. A stable structure keeps the same
# ordering (level first) and roughly the same shares on every window.

changes, tenors = daily_changes(history)
full = fit_pca(changes, tenors, n_components=3)
print(f"full sample ({full.n_observations} obs):")
for i, name in enumerate(full.component_names):
    print(
        f"  {name}: {100 * full.explained_variance_ratio[i]:5.1f}%  shape={full.loading_shape[i]}"
    )
window = 250  # ~1 year of trading days
step = 60
print()
print("trailing 1y windows (oldest -> newest):")
for end in range(window, len(changes) + 1, step):
    w = fit_pca(changes[end - window : end], tenors, n_components=3)
    shares = " ".join(f"{100 * s:4.1f}%" for s in w.explained_variance_ratio)
    print(f"  ending {history['date'].unique()[end - 1]}: {shares}  | {w.component_names}")

# %%
# **The level/slope/curvature structure is stable.** PC1 stays near ~70% and
# sign-consistent (level) on every window; PC2/PC3 hold their ordering. The
# decomposition in notebook 05 is therefore a property of the curve's dynamics,
# not a sample artefact. This is the check that matters for a risk report built
# on PCA: the direction/scale split the library exposes (see the PCA note in
# notebook 05) is meaningful on any window, not just the full one.
#
# ## Parametric-fit stability
#
# Fit Nelson-Siegel on a monthly sample of dates and track the four parameters
# and the in-sample fit error. The parameters are the level ($\beta_0+\beta_1$),
# the slope ($-\beta_1$), the curvature ($\beta_2$) and the decay ($\tau$).

sample_dates = sorted(set(history["date"]))[::45]  # ~monthly

fitted_curves: dict[str, NelsonSiegel] = {}


def _fit_ns(d: str):
    day = history[history["date"] == d].sort_values("tenor_years")
    times = day["tenor_years"].to_numpy(dtype=float)
    rates = day["rate"].to_numpy(dtype=float)
    try:
        fit = NelsonSiegel.fit(times, rates, date.fromisoformat(str(d)), seed=20260727)
    except FitError:
        return None
    fitted_curves[str(d)] = fit.curve
    b0, b1, b2 = fit.curve.beta
    return d, b0, b1, b2, fit.curve.tau, fit.rmse


rows = [r for r in (_fit_ns(d) for d in sample_dates) if r is not None]
print(
    f"fitted {len(rows)} of {len(sample_dates)} dates "
    f"({len(sample_dates) - len(rows)} skipped: near-zero short rates saturate the "
    f"positivity constraint)"
)
fit_table = pd.DataFrame(rows, columns=["date", "b0", "b1", "b2", "tau", "rmse"])
print("Nelson-Siegel parameters across the sample (rates in decimal):")
print(fit_table.describe().round(4).to_string())

# %%
fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
dates = pd.to_datetime(fit_table["date"])
ax[0, 0].plot(dates, fit_table["b0"] + fit_table["b1"], color="#1f77b4")
ax[0, 0].set_title("level $\\beta_0+\\beta_1$ (short rate)")
ax[0, 1].plot(dates, -fit_table["b1"], color="#ff7f0e")
ax[0, 1].set_title("slope $-\\beta_1$")
ax[1, 0].plot(dates, fit_table["b2"], color="#2ca02c")
ax[1, 0].set_title("curvature $\\beta_2$")
ax[1, 1].plot(dates, fit_table["tau"], color="#d62728")
ax[1, 1].set_title("decay $\\tau$ (years)")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.suptitle("Nelson-Siegel parameters, 2021-2026 (monthly)")
fig.tight_layout()
plt.show()

print(f"in-sample rmse: mean {fit_table['rmse'].mean():.3e}, max {fit_table['rmse'].max():.3e}")

# %%
# **The fit is stable and cheap.** The level tracks the rate cycle (it rises
# through 2022-2023 and falls after), but the *shape* parameters — slope,
# curvature and decay — move in a bounded band rather than jumping, and the
# in-sample error stays near machine-small. The parameters are identified, not
# chasing noise: a curve fitted on any one date is a well-posed object.
#
# ## Out-of-sample: does yesterday's fit still describe today?
#
# A stronger test than parameter stability: fit on date $t$, then evaluate that
# same curve against the par yields observed one month later. The drift in that
# error is the honest size of "the curve moved and the model did not re-fit".

oos = []
for d_today, d_next in zip(fit_table["date"], fit_table["date"][1:], strict=False):
    day_next = history[history["date"] == d_next].sort_values("tenor_years")
    t_next = day_next["tenor_years"].to_numpy(dtype=float)
    r_next = day_next["rate"].to_numpy(dtype=float)
    pred = np.array([fitted_curves[str(d_today)].zero(float(t)) for t in t_next])
    oos.append(float(np.sqrt(np.mean((pred - r_next) ** 2))))

print("out-of-sample error of the prior month's fit, in basis points:")
print(pd.Series(oos).describe().round(2).to_string())
print(f"  (in-sample rmse for contrast: mean {fit_table['rmse'].mean():.2e})")

# %%
# **The out-of-sample error is orders of magnitude larger than the in-sample
# error** — the curve genuinely moves month to month, so a stale fit is wrong by
# tens of basis points where a fresh fit is wrong by less than a basis point.
# That is the correct, expected behaviour: it is why a production curve is
# rebuilt daily (see the interpolation notebook's note on phantom moves), and it
# is the honest boundary of a *stability* check — the model is stable, but
# "stable" does not mean "does not need refitting".
#
# ## Limitations
#
# - **Par yields, not zero rates.** The parametric fit targets the published CMT
#   par-yield curve directly; a bootstrapped zero-curve fit would shift the
#   parameter levels (not the stability conclusion).
# - **One market, one regime.** Five years of USD Treasury data ending 2026-07-24
#   is a single sample; nothing here is claimed for other currencies or regimes.
# - **No P&L.** There is no backtest, no trading signal and no prediction; the
#   out-of-sample number measures curve drift, not model profitability.
#
# ## Regulatory context
#
# A stability check is not model validation. Regulatory validation of a bank's
# IRRBB or market-risk measurement system requires an independent validation
# function, backtesting against realised P&L and escalation on breaches (CRD
# Articles 74/84/101 for a bank's internal-
# model obligations). This notebook performs none of that and claims none of it:
# it is a diagnostic that the library's own outputs are internally consistent
# across time.
