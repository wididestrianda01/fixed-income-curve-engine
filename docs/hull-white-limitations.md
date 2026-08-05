# Hull-White Model Limitations

The Hull-White one-factor model in `yieldcurve.models.hullwhite` is used in
this project for two things: pricing European swaptions against the packaged
ATM normal-volatility grid, and simulating short-rate paths whose
trapezoid-averaged integrals approximate path discount factors — the
monthly path-discount approximation of §6. This document records what the
model **cannot** do, so that it is used within its known boundaries.

Each limitation below names a specific consequence and, where possible, gives a
quantified measure of its magnitude.

---

## 1. Rates can go arbitrarily negative

The short rate is Gaussian under the risk-neutral measure. Its distribution at
any horizon is normal with a closed-form mean and standard deviation (the
conditional Hull-White mean and the standard deviation
`σ·sqrt((1 − exp(−2a·(t−s))) / 2a)`). Here `s` is the conditioning time (the
date the distribution is conditioned on; `t` is the horizon — `s = 0` recovers
the F₀-conditional standard deviation of theory.md §11). Negative rates are
not prevented; they have a computable probability.

For the illustrative parameters `a = 0.05`, `σ = 0.010` on a flat 3% curve:

| Horizon | P(r < 0) |
|---------|----------|
| 1 year  | 0.1%     |
| 5 years | 5.9%     |
| 10 years| 9.4%     |

At 10 years, roughly one path in ten produces a negative short rate. This is a
direct consequence of the Gaussian distribution and is not an implementation
error: it is the model.

## 2. One factor means perfect instantaneous correlation across the curve

Every forward rate is driven by the same `dW`. The model-implied correlation
between any two forward rates is exactly 1. Empirical zero-rate histories do
not behave that way: the principal-component analysis in `risk/pca.py` shows
level, slope and curvature components carrying separate shares of historical
variance.

**The model is therefore not used to generate curve scenarios.** The scenario
module (`risk/scenarios.py`) implements the six EU 2024/856 supervisory shock
shapes with their USD and SEK parameters; PCA supplies direction and scale
measures (`pca_durations`, `pca_exposure`); the one-factor model is used for
short-rate simulation and swaption pricing only.

## 3. Two parameters cannot fit a volatility surface

`(a, σ)` are two numbers. A swaption surface has (expiry count) × (tenor count)
independent quotes. A co-terminal ATM strip is the standard compromise: it is
the set Hull-White can genuinely match.

The calibration RMSE in notebook 06 — about 6.5 bp against the packaged
7-expiry by 4-tenor illustrative grid — is a measure of fit quality, not a
measure of modelling error. What was not fitted — the smile across strikes,
the full expiry-tenor grid — is not captured by this model.

## 4. Constant `σ` produces no volatility smile

A flat volatility parameter implies a flat normal-volatility skew: the model
prices an off-ATM swaption with the same `σ` as the ATM one. The packaged
snapshot contains ATM normal volatilities only, so the model is not calibrated
to any strike dependence. No bound is claimed on the difference between model
and market prices at off-ATM strikes — that difference is not measured here.
An off-ATM price from this model is an illustration under the flat-`σ`
assumption, not a calibrated market price.

## 5. Mean reversion is fitted, not observed

`a` from a swaption calibration is a vol-surface shape parameter. It controls
how the volatility decays with option expiry on the co-terminal strip. It
generally disagrees with a time-series estimate of mean reversion from
historical short-rate data.

Do not read `a` as an economic quantity. It is a quoting parameter, like `σ`,
that happens to share the same letter as the mean-reversion speed in the SDE.
Do not compare it to a time-series estimate or cite it as evidence of
mean-reverting behaviour in the short rate.

## 6. Path discount factors are an approximation, not exact bond prices

`simulate_path_discount_factors` returns `exp(−∫ r(s) ds)` computed by the
trapezoid rule on a monthly grid (the default) — the monthly path-discount
approximation demonstrated in notebook 06 — which carries an O(step²)
time-step bias relative to the exact zero-coupon bond price
`P(t,T) = A(t,T)·exp(−B(t,T)·r(t))`. The short rate itself is sampled exactly
(Gaussian transition, no discretisation bias); the trapezoid integral is the
only source of time-step bias.

The bias is small but measurable: on the packaged test fixture (`a = 0.05`,
`σ = 0.01`, flat 3% curve, 5-year horizon) it is about 2.9e-7, roughly three
orders of magnitude below the 3-SE Monte Carlo window of the packaged
simulation tests. The tests measure the bias deterministically (closed-form
expectation of `exp(−trapezoid)`) and verify that it shrinks as the step size
is reduced, so Monte Carlo error and time-step bias are not conflated.
