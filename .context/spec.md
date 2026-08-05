# P3 — Fixed Income & Yield Curve Engine + Hull-White Extension: Specification

Status: controlling specification of the implemented repository (post-remediation). This document describes what the shipped code actually does and what it explicitly does not claim. It replaces the pre-implementation draft, whose phase/milestone promises and master-index references no longer describe the repository.

## 1. Problem statement and product definition

The repository is an educational fixed-income toolkit: it (a) constructs a zero-coupon discount curve from quoted bills, par bonds and swaps, (b) prices fixed- and floating-rate bonds and swaps off that curve, (c) computes standard and curve-partitioned interest-rate-risk measures, and (d) fits a one-factor Hull-White short-rate model and simulates short-rate paths whose trapezoid-averaged integrals approximate path discount factors.

The product is a verified educational portfolio, not a production or regulatory platform. Nothing in the code, documentation, notebooks or app claims production readiness, regulatory approval, live-data operation, or suitability for an institution's own risk or capital reporting. Unsupported institutional claims were removed rather than implemented through speculative bank infrastructure.

## 2. Scope

In scope (all implemented and tested):

- **Curve construction** — sequential bootstrap of a zero-coupon discount curve from bill, par-bond and swap quotes. The canonical method is log-linear discount-factor interpolation; the canonical builder enforces exact re-pricing of every input quote (a canonical build that does not reprice within tolerance fails loudly). Cubic log-DF and monotone-convex curves remain available as comparative overlays whose final quote residuals are measured via `repricing_report`, never asserted to vanish.
- **Pricing** — dirty/clean price, accrued interest, yield-to-maturity for fixed-coupon bonds; bills; floating-rate notes with observed fixings; vanilla swaps and OIS. All valuations go through one dispatch in `yieldcurve.curves.pricing`; curve reference dates and the valuation date (`asof`) are handled explicitly (discount factors are rebased to `asof`).
- **Risk measures** — DV01 (positive loss per 1 bp rise, per instrument and portfolio), Macaulay/modified duration, convexity, Fisher-Weil duration, key-rate durations on the SEK grid, PCA direction and scale measures, and a linearized delta VaR/ES proxy. VaR/ES use the loss-positive convention (positive numbers are losses).
- **Scenario module** — the six supervisory shock scenarios of Commission Delegated Regulation (EU) 2024/856 (parallel up/down, short-rate up/down, steepener, flattener) with the USD and SEK parameters of its Annex Part A (200/300/150 bp) and the Article 3(7) maturity-dependent post-shock rate floor, for educational analysis.
- **Hull-White one-factor model** — exact Gaussian-transition simulation of the short rate; monthly path-discount-factor approximation (trapezoid quadrature, O(step²) time-step bias, tested separately from Monte Carlo error); European swaption and zero-coupon-bond-option pricing with the closed-form affine bond formula; calibration to a co-terminal ATM normal-volatility strip with diagnostic reporting.
- **Data** — one audited, versioned, read-only market-data snapshot (2026-07-24) packaged as package resources, plus explicit caller-supplied data. No network access at runtime.

Out of scope (explicitly not implemented and not claimed):
- Institution-wide supervisory outlier testing, IRRBB compliance, capital, NII or non-maturity-deposit behavioural modelling, currency aggregation, or regulatory reporting.
- Production deployment, live data feeds, execution, or trading.
- XVA, credit-spread modelling, volatility-surface calibration (the model fits the initial curve plus a co-terminal ATM strip only), multi-factor short-rate models, swaption volatility surfaces, callable/putable bonds, OAS, mortgages.
- Empirical or regulatory model validation. Cross-library checks against QuantLib are selected implementation cross-checking (see §5), not independent empirical validation.

## 3. Key contracts and conventions

- **Curve time** is always ACT/365F years from the curve reference date. **Accrual time** uses the instrument's own day count.
- **CMT and benchmark par-yield mapping** — published Treasury CMT rates (>1y) and Riksbank benchmark yields are treated as par-yield inputs: the quoted yield becomes the coupon of a par instrument that prices to 100, so the par-yield mapping is exact by construction; short CMT tenors (≤1y) are bills.
- **Fixing conventions** — an FRN coupon that has already fixed uses the observed fixing (keyed by index tenor and reset date); unfixed coupons project forwards. Overnight legs use observed overnight fixings.
- **Key-rate durations** — Ho (1992) triangular shifts forming a partition of unity; `sum(krd)` approximates the parallel-shift duration up to the O(bump²) central-difference truncation error, not exactly.
- **PCA** — direction and scale are separated: `pca_durations` is the duration along a unit-norm loading (direction only), `pca_exposure` scales it by the component's empirical standard deviation. Sign conventions are deterministic.
- **VaR/ES** — loss-positive: a positive number is a loss. The historical risk numbers are a linearized delta proxy for educational analysis; nothing here is a regulatory measure and no capital is computed.
- **Hull-White path simulation** — named for what it returns: monthly path discount-factor approximation, not exact zero-coupon bond simulation. The short rate is sampled exactly; the trapezoid integral is the only source of time-step bias.

## 4. Verification approach

- Canonical bootstrap exactness is enforced mechanically on every build (tolerance 1e-10); overlay residuals are reported and displayed, never asserted away.
- Independent test vectors cover pricing (bills, bonds, FRNs, swaps, OIS), bootstrap, risk measures, scenarios, and Hull-White; simulation tests separate Monte Carlo error from time-step bias by comparing against closed-form expectations.
- Selected cross-library comparisons against QuantLib (bond prices/yields, swaption NPVs, ECB Svensson parameter reconstruction) are reported with exact quantities and tolerances in the README and notebooks. These are software verification checks — selected implementation cross-checking — not empirical or regulatory model validation.
- The README validation section states exact quantities and tolerances: selected bond clean/dirty/accrued/yield comparisons, a selected Hull-White swaption NPV comparison, ECB Svensson parameter reconstruction and fitted-curve errors, and the current measured package statement coverage.

## 5. Relation to QuantLib

QuantLib-Python is used as a second implementation of the same closed-form formulas for selected instruments and model outputs. Agreement between the two implementations confirms absence of implementation bugs on the tested instruments; it does not validate the models against markets. This parity framing applies to every document in the repository: "selected implementation cross-checking, not independent empirical validation."
