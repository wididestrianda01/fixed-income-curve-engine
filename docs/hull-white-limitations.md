# Hull-White Model Limitations

The Hull-White one-factor model is a powerful tool for arbitrage-free discount-factor
simulation and swaption pricing. This document records what it **cannot** do, so that
the model is used within its known boundaries.

Each limitation below names a specific consequence and, where possible, gives a
quantified measure of its magnitude.

---

## 1. Rates can go arbitrarily negative

The short rate is Gaussian under the risk-neutral measure. Its distribution at any
horizon is normal with a closed-form mean and standard deviation. Negative rates are
not prevented; they have a computable probability.

For the illustrative parameters `a = 0.05`, `σ = 0.010` on a flat 3% curve:

| Horizon | P(r < 0) |
|---------|----------|
| 1 year  | 0.1%     |
| 5 years | 5.9%     |
| 10 years| 9.4%     |

At 10 years, roughly one path in ten produces a negative short rate. This is a
direct consequence of the Gaussian distribution and is not an implementation error:
it is the model.

## 2. One factor means perfect instantaneous correlation across the curve

Every forward rate is driven by the same `dW`. The model-implied correlation between
any two forward rates is exactly 1.

Phase 4's PCA measured the real thing: three components, and the second and third
carry a recorded share of the variance. **Therefore this model is used for
arbitrage-free discount-factor simulation and for swaption pricing, and explicitly
not to generate curve scenarios.** The empirical PCA components in `risk/pca.py` are
what scenario generation should use.

## 3. Two parameters cannot fit a volatility surface

`(a, σ)` are two numbers. A swaption surface has (expiry count) × (tenor count)
independent quotes. A co-terminal ATM strip is the standard compromise: it is the
set Hull-White can genuinely match.

The calibration RMSE on the co-terminal ATM strip is a measure of fit quality, not a
measure of modelling error. What was not fitted — the smile across strikes, the full
expiry-tenor grid — is not captured by this model.

## 4. Constant `σ` produces no volatility smile

A flat volatility parameter implies a flat normal-volatility skew: the model price
of an off-ATM swaption uses the same `σ` as the ATM one. If the CME data shows a
smile across strikes, the difference between the model price and the market price at
off-ATM strikes is bounded by that smile magnitude. An off-ATM price from this model
is reliable to within the width of the smile.

## 5. Mean reversion is fitted, not observed

`a` from a swaption calibration is a vol-surface shape parameter. It controls how
the volatility decays with option expiry on the co-terminal strip. It generally
disagrees with a time-series estimate of mean reversion from historical short-rate
data.

Do not read `a` as an economic quantity. It is a quoting parameter, like `σ`, that
happens to share the same letter as the mean-reversion speed in the SDE.
Do not compare it to a time-series estimate or cite it as evidence of mean-reverting
behaviour in the short rate.
