# Fixed-income and rates, from the ground up

This document explains what this repository does, and why, in the order the
work actually happens on a rates desk. It is written so that someone with no
prior fixed-income background can follow the chain from a raw quote to a
two-factor model, and so that a recruiter can see the distinction this project
draws between *what is verified* and *what is claimed*.

Everything stated as a measured quantity below is enforced by a test
(809 passed, 1 skipped, 95.26% statement coverage against a 90% floor).
Everything stated as a limitation is stated on purpose.

---

## 1. What this is

A from-scratch Python library that turns raw market quotes into an
interest-rate curve, prices fixed-income instruments off that curve, measures
interest-rate risk, and fits the term-structure and volatility models a rates
desk uses. The models are implemented from their closed-form formulas and then
cross-checked against QuantLib, an independent industry-standard library. The
point of that second implementation is not that QuantLib is "right" and this
code is "checked against it"; it is that two independent implementations of the
same formula agreeing at machine precision rules out implementation bugs on the
tested instruments. That is the difference between *software verification* and
*model validation*, and it is the idea the whole project is built around.

The repository is deliberately educational. It is not a trading system, an
accounting-valuation system, a regulatory-reporting system, or a production
risk system. Nothing here computes capital, and no output is a regulatory
measure.

---

## 2. The atomic object: a discount factor

Fixed income is built on one idea. A discount factor d(t) is the price today of
one unit of currency delivered at time t:

    d(t) = exp(-y(t) * t)

where y(t) is the continuously-compounded zero (spot) rate for maturity t. The
instantaneous forward rate is the logarithmic derivative of the discount
factor:

    f(t) = -d/dt ln d(t)

Why this matters: a fixed-income instrument is a bundle of known future cash
flows. A five-year bond is not "worth five-year money"; it is ten coupon
payments plus a principal repayment, each discounted at its own maturity's
rate. The discipline of the field is therefore two steps. First, extract d(t)
for every t from the prices of liquid instruments. Second, discount arbitrary
cash flows with it.

One convention governs everything downstream: curve time is ACT/365F years from
the curve's reference date, and zero and forward rates are continuously
compounded. A single, stated time convention is what keeps bootstrap, pricing,
and risk internally consistent. (`yieldcurve.curves.protocol` implements this
contract; `curve_time(ref, d)` is the only conversion any curve method
accepts.)

---

## 3. Curve construction

You have quotes for a handful of tenors (bills, bonds, swaps). You need d(t)
for every t. This is two distinct problems, and keeping them distinct is the
first thing a desk learns.

### 3.1 Bootstrap: the sequential solve

Sort instruments by maturity. The shortest instrument pins the first discount
factor. Each next instrument's price equation contains exactly one new unknown,
d(t_k), because every earlier cash flow is already discounted at a known
factor. Solve for it by root-finding on the instrument's own pricing function,
or, for instruments whose cash flows land on bootstrap dates, by solving the
triangular linear system

    P = CF · d  ⇒  d = CF⁻¹ · P

where CF is lower-triangular. (`yieldcurve.curves.bootstrap`.)

### 3.2 Interpolation: what d(t) does between knots

Interpolation is a real design decision, not a detail, and the repository treats
it honestly.

- **Log-linear discount factors** (the canonical method). ln d(t) is piecewise
  linear, so forwards are piecewise constant and, critically, the interpolant on
  an interval depends only on its two bounding knots. Adding a later pillar
  never changes an earlier solve, so the bootstrap reprices every input quote
  exactly (within the documented 1e-6 bp tolerance). This is the standard choice
  when exact quote repricing matters more than smoothness.

- **Cubic log-DF** and **Hagan-West monotone-convex** (the overlays). Their
  interpolants depend on the whole knot set, so a later pillar reshapes the
  interpolant behind it and the sequential solves drift. The repository does not
  hide this: `repricing_report` measures each overlay's final per-quote residual
  and displays it rather than asserting it vanishes.

- **Extrapolation** is flat in the zero rate beyond the last knot at both ends.
  Extrapolated values are unobservable inputs, and the code records
  `covered_horizon`, the largest curve time actually backed by a quote.

The desk tradeoff, made explicit: risk and hedging want smooth, monotone
forwards so hedges are stable; accounting and valuation want exact repricing of
the input quotes. You cannot have both everywhere, and claiming you can is the
mark of someone who has not done the work. The repository shows the residual
instead.

### 3.3 Parametric fits

Nelson-Siegel and Svensson take a different route: fit a low-parameter smooth
function to the whole curve by nonlinear least squares rather than pinning
knots. The useful property here is that central banks publish their own such
parameters. The ECB publishes its Svensson parameters (beta0..beta3, tau1,
tau2), which gives the project an *independent reconstruction target*: rebuild
the ECB's published spot curve from their parameters and check agreement within
0.5 bp at every published tenor. The project's own Svensson fit lands within
1.0 bp with RMSE below 0.5 bp. (`yieldcurve.curves.parametric`.)

---

## 4. Pricing instruments off the curve

### Bonds: dirty, clean, accrued, yield

Dirty price is the sum of discounted cash flows. Accrued interest is the coupon
times the fraction of the coupon period elapsed. Clean price is dirty minus
accrued. Yield to maturity is the single flat rate that reproduces the dirty
price, found by root-finding (Brent, street convention). The subtlety the code
gets right: valuation is curve-based, each cash flow discounted at its own zero
rate, and yield is computed afterward as a flat-rate summary. On a sloped curve
the two genuinely disagree, and that disagreement is a real number, not a bug.

### Floating-rate notes and the fixing convention

A coupon whose reset date has passed uses the observed fixing (keyed by index
tenor and reset date). An unfixed coupon projects the forward from the curve.
An overnight leg compounds observed overnight fixings. Getting this wrong is the
classic floating-leg defect: the price is correct until a fixing date crosses
and the leg silently switches from projected to fixed with no observed rate to
use. The repository carries a dedicated fix branch for exactly this class of
bug.

### The multi-curve split (the post-2008 change)

Before 2008 a single curve did everything: you discounted and forecast off the
same curve. After 2008 the market recognized that the rate at which an
institution borrows (LIBOR, and now Term-SOFR) embeds credit and liquidity risk,
while the rate at which it collateralizes (OIS, SOFR) is near risk-free. A
modern stack therefore separates the **discount curve** (OIS) from the
**forecast curve** (3M Term-SOFR = OIS + a basis spread). `usd_curveset` builds
exactly this: an OIS discount curve and a 3M forecast curve with a documented
Term-SOFR basis. This is the most "real-industry" idea in the project, because
it is invisible to someone who only read a pre-2010 textbook.

Cross-currency extends the same idea: the collateral (CSA) currency's curve
discounts the cash flows, and the cross-currency basis is the premium for
borrowing in one currency against the other. (`yieldcurve.curves.xccy`.)

---

## 5. Interest-rate risk

- **DV01** is the loss a long position takes when rates rise 1 bp. The
  convention is loss-positive: a positive number is a loss. The sign convention
  is a live source of wrong interpretation in production, and it is pinned by
  tests.
- **Duration** has several honest variants. Macaulay is the present-value
  weighted mean time to receipt, defined under a flat yield. Fisher-Weil is the
  spot-curve-weighted mean time. Modified duration is the first-order
  sensitivity. Effective duration reprices under a shifted curve and is defined
  for anything the pricer prices, including swaps and FRNs where a
  yield-space duration is undefined. On a flat curve these agree; on a sloped
  curve they differ for a reason.
- **Convexity** is the second-order term. It has no standardized scaling
  convention across vendors, so the cross-check against QuantLib confirms the
  convention before comparing numbers.
- **Key-rate duration** (Ho 1992) shocks one tenor with triangular shifts that
  form a partition of unity, so key-rate durations sum to parallel duration up
  to the O(bump²) truncation error of the central finite difference. The tests
  pin that residual rather than claiming exactness.
- **Principal-component analysis** separates *direction* (the loading shape,
  level/slope/curvature) from *scale* (the component's empirical volatility).
  Conflating the two is the classic PCA risk-measure error. The two measures,
  `pca_durations` and `pca_exposure`, exist so the distinction stays explicit.
- **VaR / expected shortfall** is a linearized delta proxy: portfolio value
  change is approximated as sensitivity times the historical rate-change
  distribution. Loss-positive, and explicitly a volatility proxy for
  educational analysis, not SEK VaR and not a regulatory measure.
- **The six EU 2024/856 shocks** (parallel up/down, short up/down, steepener,
  flattener) are the supervisory scenarios a bank uses for interest-rate risk in
  the banking book (IRRBB). The module implements the shapes and the USD/SEK
  parameters (200/300/150 bp) plus the Article 3(7) post-shock rate floor, and
  states plainly that the resulting ΔEVE comparison is an educational exhibit,
  not an institution's IRRBB submission. (`yieldcurve.risk.scenarios`; the
  parameter source cites the regulation article per parameter.)

---

## 6. Term-structure models: pricing, not forecasting

### Hull-White (one factor)

The extended-Vasicek model:

    dr(t) = [theta(t) - a r(t)] dt + sigma dW(t)

with theta(t) chosen to fit the initial term structure exactly. Two
implementation details show care. First, the code never evaluates theta(t): it
contains the second derivative of a bootstrapped curve, which is noise, and the
affine bond price needs only the initial discount factor and forward, not theta.
Second, the bond price

    P(t,T) = A(t,T) exp(-B(t,T) r(t)),   B(t,T) = (1 - exp(-a(T-t)))/a

is evaluated with `expm1` to avoid catastrophic cancellation. Simulation
samples the short rate exactly from Gaussian transitions (no Euler
discretization bias), and the path discount factor uses a trapezoid rule whose
O(step²) time-step bias is measured deterministically and separated from Monte
Carlo error. The honest caveat is stated, not hidden: a Gaussian short rate has
a positive probability of negative rates at any finite horizon, and the formula
for that probability is given with example numbers.

### G2++ (two factors)

One factor forces perfect instantaneous correlation across the curve, a property
real markets do not have. The two-factor Gaussian model produces the
decorrelation a Hull-White cannot, and is cross-checked against QuantLib's G2 at
machine precision across parameter and (t, T) grids.

### SABR

Hagan (2002) produces the volatility smile, the skew and curvature that a
flat-vol model cannot. The repository implements normal and lognormal implied
volatility with skew and curvature, calibrates to an illustrative smile, and
cross-checks against QuantLib at relative 1e-10.

The industry framing: these are pricing models, fit to today's curve and a vol
strip, not forecasting models. A calibrated mean-reversion parameter is a
vol-surface shape parameter and generally disagrees with a time-series
estimate. The documentation states this rather than letting a reader infer it.

---

## 7. Inflation and cross-currency

- **Inflation.** Zero-coupon breakeven curves, inflation-linked bond (linker)
  pricing with indexation lag, and zero-coupon inflation swaps, all off the
  Fisher identity: the real zero rate is the nominal zero rate minus the
  breakeven. (`yieldcurve.inflation`.)
- **Cross-currency basis.** Cross-currency basis curves and collateral (CSA)
  currency discounting. The EUR/USD basis is quoted as the spread added to the
  EUR leg against USD SOFR flat, with a negative sign for the USD funding
  premium. (`yieldcurve.curves.xccy`.)

Both are illustrated with fabricated, deterministic data whose shape is
documented and whose generators are committed, so the distinction between "fit
to a traded price" and "constructed to look plausible" is never blurred.

---

## 8. Data provenance

One frozen snapshot (dated 2026-07-24), fourteen datasets, each classified
exactly one of:

- **public** — observed values from a third-party publisher, with source and
  licence status recorded;
- **constructed** — computed in this repository from recorded inputs, never
  presented as observed live quotes;
- **illustrative** — fabricated with a documented shape, not market data and
  not a fit to any traded price.

Every dataset records publisher, primary URL, retrieval and observation dates,
raw field meaning and units, the transformation applied, licence and
redistribution status, and known limitations, in `DATA_SOURCES.md`, with a
machine-readable twin (`snapshot_manifest.toml`) pinned to it by tests that
compare the recorded sha256 against the packaged bytes.

This is the part most portfolio projects skip and the part a real quant or risk
team cares about most: shipping data you cannot legally redistribute, or mixing
fabricated values into a curve and calling it market data, is a firing offense
in a bank. The project does the opposite. It marks the swaption vol grid, the
inflation breakevens, and the cross-currency basis as illustrative, and it
records the FRED retrieval-terms nuance (the underlying Treasury rates are U.S.
government work and therefore public domain, but FRED's retrieval service terms
restrict archiving and scraping) rather than overriding it.

Why two currencies: the SEK market is incomplete in free sources (the Riksbank
publishes benchmark yields at only four tenors, and STIBOR moved to a
commercial facility). The project uses SEK for curve construction and
interpolation, EUR for parametric fitting (the ECB publishes its own Svensson
parameters), and USD for dual-curve pricing, risk, and the models. The currency
changes on purpose, not because the project cannot stay consistent.

---

## 9. Verification: the distinction that matters

Two things are routinely confused, and the project keeps them apart:

- **Software verification** asks whether this code behaves as its own contract
  states. This is measured with tolerances, coverage, and a golden regression
  file, and it is what the repository has.
- **Model validation** asks whether a model is empirically right about markets.
  That requires historical backtesting, out-of-sample work, and independent
  review. It is not claimed.

The from-scratch-then-cross-check method is the load-bearing idea. Implementing
Hull-White twice (once from Brigo and Mercurio, once by calling QuantLib) and
observing agreement at machine precision proves the implementation is correct on
the tested instruments. It does not prove the model is right about markets, and
the documentation never describes it as such.

Measured quantities, all pinned by tests:

| Check | Measured quantity |
|---|---|
| Package tests | 809 passed, 1 skipped; 95.26% statement coverage vs a 90% floor |
| QuantLib bond parity | clean/dirty within 1e-8 per 100 face; accrued 1e-10; yield 1e-8; modified duration rel 1e-4; convexity 1e-6 via a 50 bp move |
| Hull-White swaption | matched against QuantLib's Jamshidian engine |
| ECB Svensson | published parameters rebuild the published curve within 0.5 bp; own fit within 1.0 bp, RMSE < 0.5 bp |
| SABR implied vol | normal and lognormal vs QuantLib, rel 1e-10 |
| G2++ bond price and option | vs QuantLib G2, rel 1e-10 |
| Log-linear repricing | every quote within 1e-6 bp; overlay residuals measured per quote |
| EU 2024/856 scenarios | six shocks, USD/SEK 200/300/150 bp, Article 3(7) floor |
| Distribution | wheel carries all 14 datasets and the limitations doc; sdist denylist scan reports zero local-state hits |

QuantLib is a development-only extra: nothing under `src/` imports it, and the
parity tests treat it as a cross-check, not as proof of model validity.

---

## 10. How this maps to how a bank actually works

The repository is a compact but honest model of the quant/risk stack, in the
order the work happens:

1. **Market data.** A desk cannot price anything without clean, licensed,
   dated data. The snapshot plus provenance is the data-governance layer.
2. **Curve construction.** The bootstrap plus interpolation choice is the
   curve-builder layer. The exact-repricing-vs-smoothness tradeoff is a daily
   desk decision.
3. **Instrument pricing.** Bonds, FRNs, swaps, and OIS off a multi-curve
   framework is the pricer layer, including the fixing convention that breaks
   silently if mishandled.
4. **Risk.** DV01, duration, convexity, key rates, PCA, and the supervisory
   scenarios are the risk/ALM layer, where sign conventions and direction-vs-
   scale distinctions are the difference between a correct number and a
   confidently wrong one.
5. **Models.** Hull-White, G2++, SABR, inflation, and cross-currency are the
   pricing-model layer, used to price options and calibrate vol, not to
   forecast rates.
6. **Verification.** The independent cross-check and the golden regression are
   the model-risk-governance layer, which exists because a model that is not
   independently reimplemented is a model nobody has reason to trust.

What this repository deliberately does *not* do is equally informative: no
FRTB capital, no XVA or counterparty exposure, no live data feed, no
behavioural deposit or prepayment model, no NII, and no regulatory submission.
A real desk adds these on top of the same six layers. The project's value is
that it draws the line between what is implemented and verified and what is
not, in writing, rather than implying coverage it does not have.

---

## 11. Boundaries, stated plainly

- Hull-White is a Gaussian model: negative nominal rates are possible by
  construction, calibration uses at-the-money swaptions only, and accuracy
  degrades off the money. (`docs/hull-white-limitations.md`.)
- The data is one frozen snapshot. The SEK curve interpolates a 1Y point no
  free source publishes. The USD inputs are a CMT-implied approximation plus
  constructed spreads.
- The risk outputs are diagnostics on a stylized book, not regulatory measures.
  The software-verification checks are not empirical or regulatory model
  validation.
