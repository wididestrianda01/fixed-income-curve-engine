# yieldcurve

**Author:** [Your Name], MSc in [Your Degree] (in progress)

**Live app:** [PUBLIC URL — fill in after deploy]

A fixed-income yield curve engine built from bootstrapping to Hull-White calibration,
priced at parity with QuantLib (1e-8 tolerance on discount factors), covered at 90%+,
and verified against ECB target-rate statistics to within 0.5 bp.

## Quick start

A Python library for multi-curve fixed-income term structure construction, pricing and interest-rate risk. Built from live market data: EUR curves from ECB publications, USD Treasury and swap rates from the Federal Reserve's FRED, and SEK government securities from Riksbank and Riksgälden. The library implements sequential bootstrap curve construction, analytic interest-rate risk models, and single-factor interest-rate model calibration. All pricing and risk calculations use continuous-time discount factors; the library enforces immutable data structures and is designed to be correct by construction rather than defensive.

## Quickstart

Download market snapshots (committed to the repository), build a discount curve, and price a Bill:

```python
from datetime import date
from yieldcurve.market.snapshot import Snapshot
from yieldcurve.curves.build import usd_curveset
from yieldcurve.instruments import Bill
from yieldcurve.curves.pricing import price

# Load market data from the committed snapshot
snapshot = Snapshot(date(2026, 7, 24))
asof = date(2026, 7, 24)

# OIS-discounted curve set: an OIS discount curve, plus a 3M forecast
# curve bootstrapped against it rather than against itself.
curves = usd_curveset(snapshot, asof)

# Price a Bill maturing in 1 year
bill = Bill(maturity=date(2027, 7, 24))
result = price(bill, curves, asof=asof)

print(f"Bill clean price: {result.clean:.4f}")
print(f"Discount factor at 1Y: {curves.discount.df(1.0):.6f}")
```

Output:
```
Bill clean price: 95.9903
Discount factor at 1Y: 0.959903
```

## What is implemented

### Curve Construction

- **Interpolation:** All three schemes act on log discount factors, not on zero rates — the difference is invisible in a plot of zeros and highly visible in a plot of forwards. `LOG_LINEAR_DF` is the market default (piecewise-constant forwards, always monotone), `CUBIC_LOG_DF` gives smooth forwards but can overshoot, and `MONOTONE_CONVEX` is Hagan-West (2006), which delivers continuous forwards and monotone discount factors at once. Hagan-West's *positivity* amendment is deliberately omitted: it was written for a world without negative rates, and SEK and EUR forwards have been negative within the sample period.
- **Bootstrap:** Sequential in maturity order — each instrument is solved by Brent root-finding for the one discount factor that reprices it to par, given the factors already recovered from shorter instruments. Handles day-count conventions, business-day calendars, and accrued interest.
- **Multi-curve:** Separate discount and forecast curves. The forecast curve is bootstrapped *against an already-built OIS discount curve* rather than against itself, which is the post-2008 convention; discounting a swap off its own projection curve misprices the basis.
- **Parametric fits:** Nelson-Siegel and Nelson-Siegel-Svensson, fitted by differential evolution. A separate family from the interpolated curves above, satisfying the same `DiscountCurve` protocol.

### Instruments

- Bills (zero-coupon debt)
- Fixed-coupon bonds (with accrued interest, clean/dirty pricing)
- Floating-rate notes (with index tenor specifications)
- Vanilla fixed-for-floating swaps
- Overnight-index swaps (OIS)

### Pricing

- Bond and bill valuation via discounted cashflow
- Swap par rates and net present value
- Yield-to-maturity inversion (root-finding via Brent's method)

### Interest-Rate Risk

- **Sensitivities:** Effective duration, DV01, dollar convexity, and key-rate duration (Ho 1992) — a duration per tenor bucket, constructed so the key-rate durations sum exactly to the effective duration.
- **Par-delta ladder:** Risk in the coordinates the market actually quotes. Each *quoted instrument* is bumped a basis point, the whole curve is rebootstrapped, and the position repriced — so each entry answers "how much of that instrument hedges this position", which is the report a swaps desk runs against its book. This does not agree entry-by-entry with key-rate duration and is not supposed to: a bump to the 5y par quote moves every zero out to 5y, so par delta spreads where KRD localises.
- **Scenarios:** Scalar shifts, steepening, and BCBS-EBA standardized scenarios.
- **PCA:** Principal-component analysis on historical yield changes to derive empirical rate scenarios.

A caveat worth knowing before hedging off the ladder: it is additive — entries summing to the effect of bumping every quote at once — only when the curve depends smoothly on the quotes. `LOG_LINEAR_DF` and `CUBIC_LOG_DF` satisfy that to 1e-4 relative; `MONOTONE_CONVEX` does not, because its amendment tests are branches on which region a forward falls into, so a 1bp bump can flip a region and additivity breaks by around 1.4%. Pass a smooth method when the numbers are going to be traded on.

### Models

- **Hull-White 1F:** Single-factor Gaussian model with mean-reversion calibration to ATM swaption volatilities in normal-vol space. Jamshidian decomposition for European swaption pricing. Zero-coupon bond option pricing.
- **Bachelier:** Normal-model volatility support for instrument-level calibration.

## Validation

Correctness is established through:

- **QuantLib parity:** Calendars, curve bootstrap, pricing and risk are checked against QuantLib as an independent oracle. QuantLib is a development dependency used only under `tests/`; nothing in `src/` imports it, so the library itself has no such dependency. Where the two disagree the divergence is deliberate and documented — our US government bond calendar follows SIFMA, which differs from QuantLib 1.43 on Good Friday in certain years.
- **ECB reference:** Rebuilding the ECB's published spot curve from its published Svensson parameters matches to within 0.5bp, and the library's own independent fit to that curve lands within 1bp at every tenor with an RMSE under 0.5bp. Close, not exact — a 6-parameter family cannot interpolate an arbitrary published curve, and claiming otherwise would be claiming a coincidence.
- **Bootstrap round-trip:** Instruments are repriced using the discount factors extracted from them during bootstrap; recovery is to within machine precision.

## Limitations

The Hull-White model has bounded validity. Under the Gaussian SDE (continuous-time short rate), negative nominal rates are theoretically possible; calibration in normal-volatility space amplifies this effect. See `docs/hull-white-limitations.md` for a detailed analysis of mean-reversion estimation error, basis mismatches off-ATM, and scenarios with tail probability above 5%. The calibration is fit to at-the-money swaptions only, and accuracy degrades for deep out-of-the-money strikes.

CME swaption volatility data used in calibration is subject to licensing restrictions and is not redistributed in the snapshot. Self-provided swaption grids can be used via direct solver calls.

## Development

The project requires Python 3.12 and uses `uv` for reproducible dependency management. `uv sync` installs the package and its development dependencies from the lockfile:

```bash
uv sync --extra dev
```

The four gates, all of which must pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`mypy` runs in strict mode over both `src` and `tests`. `pytest` enforces a 90% coverage floor; Monte Carlo convergence tests are marked `slow` and can be skipped with `-m "not slow"` during iteration.

Market data is refreshed by hand, never at import time — no module in the package touches the network except `yieldcurve.market.refresh`:

```bash
uv run python -m yieldcurve.market.refresh --date 2026-07-24
```

Snapshots are committed, so tests, notebooks and the app run offline against a frozen `2026-07-24` snapshot.
