# P3 — Theory Note

Scope: the theory behind the implemented modules. Each section states the definition/result, the derivation where it is non-trivial, the module that implements it, and the assumptions it carries. Where a formula is implemented in code, the code is the contract and this note must agree with it; where this note disagrees with a module, the note is wrong.

## 1. Discount factors, spot rates, forward rates

**Definition.** For a zero-coupon bond paying 1 at maturity $t$, the discount factor $d(t)$ and continuously-compounded zero-coupon (spot) rate $y(t)$ are related by
$$d(t) = e^{-y(t)\,t} \iff y(t) = \frac{-\ln d(t)}{t}.$$
The instantaneous forward rate is $f(t) = -\partial \ln d(t)/\partial t$; under continuous compounding $d(t) = \exp\left(-\int_0^t f(u)\,du\right)$.

**Conventions.** Curve time is ACT/365F years from the curve's reference date; `curve_time(ref, d)` converts a calendar date. Zero and forward methods return continuously compounded rates. This is the only time notion a curve method accepts.

*Source:* Nawalkha, Soto & Beliaeva, Ch. 3 (eq. 3.1, 3.23). **Implemented in:** `yieldcurve.curves.protocol` (`DiscountCurve`, `curve_time`).

**Assumption to flag:** single-curve discounting — one discount curve serves valuation; no OIS/collateral-discounting distinction and no bid-ask spread inside the curve (spread effects are treated as bootstrapping noise, see §2.1).

## 2. Curve construction

### 2.1 Sequential bootstrap and interpolation

**Method.** Given $K$ quoted instruments with increasing maturities $t_1 < \dots < t_K$, the price of the shortest instrument pins down the first discount factor; each subsequent instrument's price equation contains one new unknown (the discount factor at its own maturity), since all earlier cash flows are discounted at already-known factors. Each unknown is solved by root-finding on the instrument's own pricing function (`yieldcurve.curves.pricing`), which keeps bootstrap and valuation consistent by construction. Equivalently, for instruments whose cash flows all fall on the $K$ bootstrap dates, the discount factors solve the triangular linear system
$$\mathbf{P} = \mathbf{CF}\,\mathbf{d} \;\Rightarrow\; \mathbf{d} = \mathbf{CF}^{-1}\mathbf{P},$$
where $\mathbf{CF}$ is lower-triangular.

**Canonical interpolation — log-linear on discount factors.** Between knots the discount factor is log-linear in time: $\ln d(t)$ is piecewise linear in $t$, so $d(t) = d(t_i)^{(t_{i+1}-t)/(t_{i+1}-t_i)} d(t_{i+1})^{(t-t_i)/(t_{i+1}-t_i)}$ on $[t_i, t_{i+1}]$, and zero rates are piecewise constant in $t$ (the forward curve is piecewise constant). Log-linear DF interpolation is the **canonical calibration method** and the only method claimed to reprice every input quote exactly: its interpolant on an interval depends only on the two knots bounding that interval, so adding a later pillar never changes an earlier solve — the sequential solve is exact. `yieldcurve.curves.bootstrap` implements the solve; `yieldcurve.curves.build` enforces the exactness contract mechanically (tolerance 1e-10): a canonical build that does not reprice every quote within tolerance is a bug.

**Comparative overlays.** Cubic log-DF (CubicSpline on log discount factors) and Hagan-West monotone-convex interpolation remain available as overlays built on the canonical nodes. Their interpolants depend on the whole knot set, so a later pillar reshapes the interpolant behind it and the sequential solves drift; their final quote residuals are **measured** via `repricing_report`, never asserted to vanish. The Hagan-West positivity amendment is deliberately not implemented (SEK and EUR forwards have been negative within the sample period; clamping would silently distort the curve); the monotonicity amendments are implemented in full.

**Extrapolation.** Every scheme extrapolates flat in the zero rate beyond the last knot (both ends). Extrapolated values are unobservable inputs — a stated modelling choice, not observed market data; `covered_horizon` records the largest curve time backed by quoted inputs. Under IFRS 13 an unobservable input is a Level 3 input, but no automatic hierarchy classification follows: classification depends on the significance of the input to the measurement (IFRS 13.72-74).

**Assumptions/limitations to flag:**
- Bootstrapping performs no error minimization; a stale quote bends the curve in its own neighbourhood and nothing smooths it away (the parametric fits in `yieldcurve.curves.parametric` are the counterpart; notebook 03 shows the same data under both).
- Exact sequential repricing is a property of the canonical method only; overlays' residuals are measured and displayed (notebook 02).

*Source:* Nawalkha, Soto & Beliaeva, Ch. 3, "Bootstrapping Method" (eq. 3.16-3.23, worked Example 3.3); Hagan & West (2006) for the monotone-convex scheme as cited in the module. **Implemented in:** `yieldcurve.curves.bootstrap`, `yieldcurve.curves.interpolation`, `yieldcurve.curves.build`.

### 2.2 Nelson–Siegel parametric curve

**Definition.** The parsimonious Nelson–Siegel instantaneous forward-rate form is
$$f(t) = \alpha_1 + \alpha_2 e^{-t/\beta} + \alpha_3 \frac{t}{\beta} e^{-t/\beta},$$
with implied zero rate (via $y(t) = \frac1t\int_0^t f(u)\,du$, per §1)
$$y(t) = \alpha_1 + (\alpha_2+\alpha_3)\frac{\beta}{t}\left(1-e^{-t/\beta}\right) - \alpha_3 e^{-t/\beta},$$
and discount function
$$d(t) = \exp\!\left[-\alpha_1 t - \beta(\alpha_2+\alpha_3)\left(1-e^{-t/\beta}\right) + \alpha_3\, t\, e^{-t/\beta}\right].$$
Parameter interpretation: $\alpha_1+\alpha_2 = y(0)$ (instantaneous short rate); $\alpha_1 = y(\infty)$ (asymptotic rate); $-\alpha_2$ is the level-to-asymptote spread (slope); $\alpha_3$ governs curvature (hump if $\alpha_3>0$, trough if $\alpha_3<0$); $\beta>0$ controls the speed of convergence to the asymptote.

**Calibration.** Parameters are fit by nonlinear least squares on price residuals, $\min_{\alpha_1,\alpha_2,\alpha_3,\beta} \sum_{i=1}^K (P_i^{\text{market}} - \hat P_i)^2$, subject to positivity constraints on the asymptotic and instantaneous rates and $\beta$. The fit result reports the optimizer's verdict, boundary saturation, Jacobian rank/condition, and residual metrics.

*Source:* Nawalkha, Soto & Beliaeva, Ch. 3, "Nelson and Siegel Model" (eq. 3.35-3.41, worked Example 3.5). **Implemented in:** `yieldcurve.curves.parametric`.

**Assumptions to flag:** nonlinear least squares is sensitive to starting values; the functional form imposes a single-hump/monotonic family of curve shapes; no i.i.d. assumption on the residuals is invoked. The fit is an alternative curve representation, not a superior one — its residuals are reported, not asserted away.

## 3. Bond pricing (dirty/clean price, accrued interest, YTM)

**Dirty price.** For a bond with $n$ remaining coupon payments, next payment $w$ of a period away,
$$P^{\text{dirty}} = \sum_{t=1}^{n} \frac{c}{(1+y)^{t-1+w}} + \frac{M}{(1+y)^{n-1+w}},$$
where $c$ is the periodic coupon, $y$ the periodic yield, $M$ the maturity value; at a coupon date ($w=1$) this reduces to the standard annuity-plus-principal formula.

**Accrued interest and clean price.**
$$AI = c \times \frac{\text{days since last coupon}}{\text{days in coupon period}}, \qquad P^{\text{clean}} = P^{\text{dirty}} - AI.$$

**Yield to maturity (YTM).** The $y$ solving the dirty-price equation given the observed market dirty price, found by root-finding (Brent's method, street convention).

**Curve-based valuation.** The pricing module discounts each cash flow at its own zero rate off the curve, then computes YTM afterwards as the flat-rate equivalent. Discount factors are rebased to the valuation date: a curve whose reference date differs from `asof` is handled by dividing by $d(\text{asof})$, so non-flat curves valued on dates other than their reference date stay correct.

**Fixing conventions.** For floating legs, a coupon whose reset date has passed uses the **observed fixing** (looked up in the `Fixings` map keyed by (index tenor, reset date)); an unfixed coupon projects the forward from the curve. Overnight legs compound observed overnight fixings. An active term coupon that has already fixed must use its observed rate rather than project a forward over a stub.

*Source:* Fabozzi, "Bond Pricing for Option-Free Bonds..." and "...Conventional Yield Measures" chapters. **Implemented in:** `yieldcurve.curves.pricing`, `yieldcurve.curves.protocol.Fixings`, `yieldcurve.instruments`.

**Assumptions to flag:** day-count/basis convention changes the accrued-interest and $w$ calculation (30/360 vs ACT/ACT); all priced bonds are option-free; the single flat yield $y$ is the YTM convention, while valuation itself is curve-based.

## 4. Duration

**Macaulay duration** (in periods): the present-value-weighted average time to receipt of cash flow,
$$D_{\text{Mac}} = \frac{\sum_{t=1}^n t \cdot PVCF_t}{PVTCF}, \qquad PVTCF = \sum_t PVCF_t = P^{\text{dirty}}.$$
Converted to years by dividing by $k$ (payments/year). **Modified duration**: $D_{\text{mod}} = D_{\text{Mac}}/(1+y)$, with $\Delta P/P \approx -D_{\text{mod}}\times \Delta y$. **Dollar duration**: $\text{DV01} \equiv \text{dollar duration of 1bp} = D_{\text{mod}} \times P / 10{,}000$.

**Naming (as implemented).** `macaulay_duration` is the classical YTM-weighted mean time; `fisher_weil_duration` is the spot-curve-weighted mean time to cash flow. **Effective duration** reprices under a shifted curve and is defined for anything the pricer prices (including swaps and FRNs, where a yield-space duration is undefined). On a flat curve the families agree; on a sloped curve they differ for a real reason.

**DV01 convention.** `dv01` is the **positive loss** a long position takes when rates rise 1 bp — `base − price(+1bp)` — not a signed price change. Portfolio DV01 aggregates the same convention.

*Source:* Fabozzi, Ch. 13. **Implemented in:** `yieldcurve.risk.sensitivities` (analytic and effective families), `yieldcurve.risk.portfolio` (aggregation).

**Assumption to flag:** modified duration is a first-order local approximation — accurate for small yield changes, biased for large ones, and symmetric in $\Delta y$ while true price/yield relations are not; this is why convexity (§5) and full re-pricing reconciliation exist.

## 5. Convexity

**Definition** (in periods, at yield $y$ per period):
$$C = \frac{\sum_{t=1}^n t(t+1)\,PVCF_t}{(1+y)^2 \times PVTCF}.$$
For a zero-coupon bond this reduces to $C = n(n+1)/(1+y)^2$. Converted to years by dividing by $k^2$. Convexity is always positive for an option-free bond. **Second-order price approximation:** $\Delta P/P \approx -D_{\text{mod}}\,\Delta y + \tfrac12\, C\,(\Delta y)^2$.

*Source:* Fabozzi, Ch. 14 (eq. 14-1, 14-2). **Implemented in:** `yieldcurve.risk.sensitivities`.

**Assumption to flag:** convexity has no standardized scaling convention across vendors; the cross-check against QuantLib confirms the convention before comparing numbers.

## 6. Key rate duration (Ho, 1992)

The $j$-th key-rate duration is the approximate percentage change in value for a shock to the spot rate at maturity $t_j$, holding other key rates fixed. The implemented shifts are **triangular**: full size at the key, falling linearly to zero at the neighbouring keys, flat beyond the first and last keys. Those flat tails make the shifts a partition of unity, which makes the key-rate durations sum to the parallel-shift duration — **up to the O(bump²) truncation error of the central finite differences, not exactly** (an earlier version of the module claimed exactness; the tests pin the residual).

**Grids.** The SEK key-rate grid is 3m/6m/1y/2y/5y/7y/10y; the SEK 1y point is interpolated rather than observed (the Riksbank publishes 6m bills and 2y benchmarks with nothing between) and that is stated wherever the SEK key-rate profile is reported.

*Source:* Ho (1992); Fabozzi, Ch. 15; Nawalkha, Soto & Beliaeva, Ch. 1 (eq. 1.7) and Ch. 9. **Implemented in:** `yieldcurve.risk.keyrate`.

**Assumptions to flag:** the number and placement of key rates is arbitrary; KRD says nothing about how *likely* a non-parallel shift is; KRD uses linear interpolation between key rates by construction.

## 7. Principal-component analysis of zero-rate changes

**Method.** A principal-component decomposition is fitted to a history of daily zero-rate changes on a common tenor grid. Two curve-risk quantities are derived per component, with explicit units:

- `pca_durations` — the modified duration along the component's **unit-norm** loading direction: fractional price change per unit (1.0 decimal) shift of the zero curve along that direction, in years. Direction only: the component's empirical volatility is not involved.
- `pca_exposure` — the fractional price change for a **one-standard-deviation** move along the component: direction duration scaled by the component's empirical standard deviation.

**Direction/scale separation.** The two measures exist precisely because direction (loading shape) and scale (empirical volatility) are independent facts; conflating them is the classic PCA risk-measure error. Signs are deterministic (largest-magnitude loading entry positive), so repeated fits on the same history return identical components. Components are named PC1/PC2/PC3 when the loading's sign pattern matches the economic criterion (no sign change = level; one = slope; two = curvature); the diagnostic behind each decision is `PCAResult.loading_shape`. Degenerate histories (constant, rank-deficient, non-finite) are rejected.

**What PCA is and is not.** PCA is a statistical description of how the curve has moved historically; it is not an arbitrage-free model and cannot price anything. Scenario shocks come from the EU 2024/856 shapes (§10); PCA supplies direction and scale measures only.

*Source:* standard PCA on yield-curve changes (Nawalkha, Soto & Beliaeva, Ch. 10 discuss principal-component durations; the module's direction/scale split is documented in the module docstring). **Implemented in:** `yieldcurve.risk.pca`.

## 8. VaR / expected shortfall proxy

The historical risk numbers are a **linearized delta VaR/ES proxy**: portfolio value change is approximated by (DV01-type sensitivities) × (historical rate-change distribution), so VaR and ES are read off a linearized P&L distribution.

**Convention.** Loss-positive: a positive number is a loss. This is stated wherever the numbers appear, because the same symbol set with the opposite sign convention is a live source of wrong interpretation.

**What it is not.** The proxy uses the historical SEK zero-rate changes available in the packaged snapshot; it is explicitly a volatility proxy for educational analysis, **not** SEK VaR, not a regulatory measure, and no capital is computed from it.

**Implemented in:** `yieldcurve.risk.portfolio` (with the scenario ΔEVE comparison and the VaR/ES proxy documented in the module docstring).

## 9. CMT and benchmark par-yield mapping

Published Treasury CMT rates (>1y) and Riksbank benchmark yields are **par-yield inputs**, not raw bill/par-instrument quotes: the quoted yield becomes the coupon of a par instrument that prices to 100 (semiannual for USD CMT >1y), so the par-yield mapping is exact by construction. CMT tenors ≤1y are bills. Reconstructing a published par yield as if it were a raw instrument quote would double-count the price-yield conversion; the builders therefore quote par instruments directly.

**Implemented in:** `yieldcurve.curves.build` (`_cmt_quote`, Riksbank benchmark mapping).

## 10. Scenario shocks — EU 2024/856

The scenario module implements the six supervisory shock scenarios of Commission Delegated Regulation (EU) 2024/856: parallel up/down, short-rate up/down, steepener, flattener. The short-rate scalar decays as $e^{-t/4}$ (Article 2(2)-(3)); the rotation weights are $-0.65/+0.9$ (steepener) and $+0.8/-0.6$ (flattener) (Article 2(4)); USD and SEK parameters are parallel 200 bp, short 300 bp, long 150 bp (Annex Part A); the post-shock floor of Article 3(7) starts at −150 bp at immediate maturity, rises 3 bp per year and reaches 0% at 50 years, with the observed rate kept when it is below the floor.

The module does not claim to implement an institution-wide supervisory outlier test, IRRBB compliance, capital, NII, behavioural modelling, currency aggregation, or regulatory reporting.

**Implemented in:** `yieldcurve.risk.scenarios` (parameter source: `src/yieldcurve/risk/scenarios.toml`, which cites the regulation article per parameter).

## 11. Hull-White one-factor short-rate model

**Dynamics (risk-neutral measure).**
$$dr(t) = [\vartheta(t) - a\,r(t)]\,dt + \sigma\, dW(t),$$
with $a,\sigma>0$ constant and $\vartheta(t)$ chosen to exactly fit the initial term structure:
$$\vartheta(t) = \frac{\partial f^M(0,t)}{\partial T} + a f^M(0,t) + \frac{\sigma^2}{2a}\left(1-e^{-2at}\right),$$
where $f^M(0,t)$ is the market instantaneous forward and $P^M(0,t)$ the market discount factor. **The implementation never evaluates $\vartheta$** (it contains the second derivative of the discount curve, which is noise on a bootstrapped curve); the affine bond price needs only $P^M(0,\cdot)$ and $f^M(0,\cdot)$, and exact simulation needs only the same.

**Conditional distribution.** Defining $\alpha(t) = f^M(0,t) + \dfrac{\sigma^2}{2a^2}(1-e^{-at})^2$, the model integrates to
$$r(s) = r(t)e^{-a(s-t)} + \alpha(s) - \alpha(t)e^{-a(s-t)} + \sigma\int_t^s e^{-a(s-u)}dW(u),$$
so $r(s)\mid \mathcal F_t$ is **Gaussian** with
$$E\{r(s)\mid\mathcal F_t\} = r(t)e^{-a(s-t)} + \alpha(s)-\alpha(t)e^{-a(s-t)}, \qquad \operatorname{Var}\{r(s)\mid\mathcal F_t\} = \frac{\sigma^2}{2a}\left[1-e^{-2a(s-t)}\right].$$
With $r(0)=f^M(0,0)$ and $\alpha(0)=f^M(0,0)$, the unconditional mean collapses to $E[r(t)] = \alpha(t)$.

**Affine bond price.** Because $\int_t^T r(u)\,du \mid \mathcal F_t$ is Gaussian, the zero-coupon bond price is closed form:
$$P(t,T) = A(t,T)\,e^{-B(t,T)\,r(t)}, \qquad B(t,T) = \frac{1}{a}\left[1-e^{-a(T-t)}\right],$$
$$A(t,T) = \frac{P^M(0,T)}{P^M(0,t)}\exp\left\{B(t,T)f^M(0,t) - \frac{\sigma^2}{4a}\left(1-e^{-2at}\right)B(t,T)^2\right\}.$$
These are the formulas implemented in `HullWhite.A`/`HullWhite.B`/`HullWhite.zcb` (with `expm1`-based evaluation to avoid catastrophic cancellation) and displayed in notebook 06. The `a → 0` limits (B = T−t, variance term = σ²tB²/2) are handled explicitly.

**Simulation semantics.**
- `simulate` samples the short rate **exactly** at the requested grid points from the conditional Gaussian transitions above — no Euler-Maruyama discretisation bias for the rate itself.
- `simulate_path_discount_factors` returns per-path $\exp(-\int_t^T r(s)\,ds)$ computed by the **trapezoid rule on a monthly grid** (default). This is a *path discount-factor approximation*, **not exact zero-coupon bond simulation**: the trapezoid rule carries an O(step²) time-step bias relative to the exact bond price $P(t,T)$ (about 2.9e-7 on the packaged test fixture, roughly three orders of magnitude below the 3-SE Monte Carlo window of the packaged tests). The tests measure the bias deterministically via the closed-form expectation of $\exp(-\text{trapezoid})$ for the Gaussian path, and verify it shrinks with the step size, so Monte Carlo error and time-step bias are never conflated.

**Conditional negative-rate probability.** Since $r(t) \mid \mathcal F_0 \sim \mathcal N(\alpha(t),\, \sigma^2(1-e^{-2at})/(2a))$:
$$Q\{r(t)<0\} = \Phi\!\left(-\frac{\alpha(t)}{\sqrt{\sigma^2/(2a)\,[1-e^{-2at}]}}\right).$$
This is the formula notebook 06 computes from the conditional Hull-White mean and standard deviation (e.g., 0.1% / 5.9% / 9.4% at 1/5/10 years for $a=0.05$, $\sigma=0.01$, flat 3% — see `docs/hull-white-limitations.md`).

**Pricing.** European swaptions are priced via the Jamshidian decomposition applied to the underlying swap's cash flows with a single strike source (ATM from the packaged grid); zero-coupon bond options use the closed Jamshidian form. Calibration fits $(a,\sigma)$ to a co-terminal ATM normal-volatility strip (Bachelier convention), reporting optimizer success, active bounds, Jacobian rank/condition, residual scale and start sensitivity; boundary, rank-deficient and unsuccessful fits are rejected.

*Source:* Brigo & Mercurio, §3.3 "The Hull-White Extended Vasicek Model" (dynamics, eq. 3.32-3.38; bond pricing, eq. 3.39); Hull & White (1990b) for the time-varying-parameter extension that the constant-parameter restriction avoids. **Implemented in:** `yieldcurve.models.hullwhite`.

**Assumptions/limitations to flag:**
- **Gaussian short rate ⇒ strictly positive probability of negative rates at any finite $t$** (formula above); small in typical calibrations but non-zero by construction.
- **Constant $a,\sigma$** is the "one time-varying parameter" restriction; the model fits the **rate curve and a co-terminal ATM strip only**, not a volatility surface — no smile, no expiry-tenor grid.
- **One factor ⇒ perfect instantaneous correlation across the curve** (single Brownian driver); the model is therefore not used to generate curve scenarios (those come from the EU 2024/856 shapes, §10), and its empirical-PCA counterpart (§7) describes historical movement with more than one component.
- **Mean reversion is fitted, not observed**: calibrated $a$ is a vol-surface shape parameter and generally disagrees with a time-series estimate.
- Path discount factors are an approximation (see simulation semantics above).
- Model limitations are documented in `docs/hull-white-limitations.md`, which the module docstring points to.

## 12. Cross-validation logic (why QuantLib is used)

QuantLib-Python's curve, bond, and Hull-White objects implement the same closed-form formulas in §§1-11. The cross-checks in the README and notebooks (selected bond clean/dirty/accrued/yield comparisons, a selected Hull-White swaption NPV comparison, ECB Svensson parameter reconstruction) are therefore a **consistency check between two implementations of the same closed-form results** — selected implementation cross-checking, not an independent empirical validation. Agreement confirms absence of implementation bugs on the tested instruments; it does not validate the models against markets, and the repository never describes it as such.

*Source:* QuantLib Python Cookbook, "Interest-rate curves" and "Bonds" sections (recipes 27-33), recipes 15-16 for Hull-White.
