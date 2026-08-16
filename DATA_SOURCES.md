# Data Sources and Provenance

This repository ships one frozen, read-only market-data snapshot (dated 2026-07-24) as
packaged resources under `src/yieldcurve/data/`. The machine-readable
`snapshot_manifest.toml` next to the CSVs is the authoritative dataset list and carries
the same provenance fields; this document is its human-readable twin. The two records
are pinned to each other: `tests/market/test_data_sources_document.py` pins dataset
coverage, classification, licence status and primary URL; `tests/market/
test_snapshot_contents.py` pins observation/retrieval dates, columns, units,
classification, licence wording and the sha256 checksums against the packaged CSV
bytes. Together they cover the full provenance record, so a dataset cannot drift apart
from its provenance in one place only.

Every dataset below records: publisher, primary URL, retrieval and observation dates,
raw field meaning and units, the transformation applied, licence/redistribution status
(verified against the cited page on the stated date, or explicitly marked unverified),
classification, and known limitations.

## Classification taxonomy

Every packaged dataset is classified exactly one of:

- **public** — observed values published by a third-party publisher, recorded with a
  source and licence status;
- **constructed** — computed in this repository from recorded inputs (which may be
  public, indicative, or unverified); never presented as observed live quotes;
- **illustrative** — fabricated values with a documented shape; not market data and not
  a fit to any traded price.

All percentage quotes are stored as decimals (2.31% → 0.0231). Licence pages were
re-verified on 2026-08-05; retrieval of the data itself happened 2026-07-24. Where a
licence or redistribution status could not be verified, this document says so
explicitly instead of asserting terms.

---

## riksbank_bills

**Classification:** public

- **Publisher:** Sveriges Riksbank
- **Primary URL:** `https://api.riksbank.se/swea/v1/Observations/<seriesId>/<from>/<to>`
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 (each row's `maturity_date` is that series'
  observation date)
- **Raw fields and units:** API series `SETB1MBENCHC` (1M), `SETB3MBENCH` (3M),
  `SETB6MBENCH` (6M), GroupId 6; the API publishes percentage yields. CSV columns:
  `tenor` (label), `maturity_date` (ISO 8601), `rate` (decimal yield, 0.01933 = 1.933%).
- **Transformation:** percentage yields divided by 100 to decimals; only tenors with a
  live series are included (the 12M bill series `SETB12MBENCH` was discontinued in 2010).
**Licence/redistribution:** verified — [Open data – information available for
  re-use](https://www.riksbank.se/en-gb/about-the-riksbank/about-the-website/open-data--information-available-for-re-use/)
  (retrieved 2026-08-05): "The Riksbank's open data is freely available and may be
  further used without any special consent or agreement being required. Enter source,
  Sveriges Riksbank and date." Processed statistics must not be presented as an official
  collaboration or partnership with the Riksbank.
- **Known limitations:** frozen 2026-07-24; the 12M bill tenor is absent (discontinued
  2010); only published tenors are included.

## riksbank_gov_benchmarks

**Classification:** public

- **Publisher:** Sveriges Riksbank
- **Primary URL:** `https://api.riksbank.se/swea/v1/Observations/<seriesId>/<from>/<to>`
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 (each row's `maturity_date` is that series'
  observation date)
- **Raw fields and units:** API series `SEGVB2YC` (2Y), `SEGVB5YC` (5Y), `SEGVB7YC`
  (7Y), `SEGVB10YC` (10Y), GroupId 7; the API publishes percentage yields. CSV columns:
  `tenor` (label), `maturity_date` (ISO 8601), `yield` (decimal yield, 0.02478 = 2.478%).
- **Transformation:** percentage yields divided by 100 to decimals; the 1Y, 15Y, 20Y and
  30Y benchmark tenors are genuinely absent from the API and are not fabricated here
  (the 1Y point is interpolated downstream in curve construction and flagged as such).
**Licence/redistribution:** verified — same Riksbank open-data page as
  `riksbank_bills` (retrieved 2026-08-05): free further use, source and date to be
  entered.
- **Known limitations:** frozen 2026-07-24; only 2Y/5Y/7Y/10Y are published; a 1Y curve
  point must be interpolated and is not an observed quote.

## riksbank_swestr

**Classification:** public

- **Publisher:** Sveriges Riksbank
- **Primary URL:** `https://api.riksbank.se/swestr/`
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 — the overnight (ON) value has value day 2026-07-23
  and publication day 2026-07-24; the compounded averages have value day equal to the
  publication day.
- **Raw fields and units:** tenors ON, 1W, 1M, 2M, 3M, 6M; the API publishes percentage
  rates. CSV columns: `tenor` (label), `rate` (decimal rate, 0.0164 = 1.64%).
- **Transformation:** percentage rates divided by 100 to decimals.
**Licence/redistribution:** verified — [Conditions for the use and re-publishing of
  SWESTR](https://www.riksbank.se/en-gb/statistics/swestr/conditions-for-use-and-re-publishing/)
  (retrieved 2026-08-05): "SWESTR (including average rates and index) is provided free
  of charge without any licensing expenses being demanded for use or re-publication";
  Sveriges Riksbank must be credited as administrator and source.
- **Known limitations:** frozen 2026-07-24; unauthenticated public access is rate-limited
  and registration is required for extended access.

## riksgalden_gov_bonds

**Classification:** public

- **Publisher:** Swedish National Debt Office (Riksgalden)
- **Primary URL:**
  [Central Government Debt statistics](https://www.riksgalden.se/en/statistics/statistics-regarding-government-securities/)
  and
  [auction results](https://www.riksgalden.se/en/our-operations/central-government-borrowing/issuance/latest-auction-result/nominal-government-bonds/)
- **Retrieval date:** 2026-07-24
- **Observation date:** coupon, issue and maturity dates as published in the Central
  Government Debt report current at the freeze; outstanding nominal from the September
  2025 report (the latest available at the freeze date).
- **Raw fields and units:** CSV columns: `isin` (ISIN identifier), `coupon` (decimal
  coupon rate, 0.01 = 1%), `issue_date` (ISO 8601), `maturity_date` (ISO 8601),
  `outstanding_nominal` (outstanding nominal in SEK).
- **Transformation:** values stored as published; coupon percentages normalized to
  decimals.
**Licence/redistribution:** unverified — no explicit reuse licence was found on
  riksgalden.se (checked 2026-08-05). The contents are official Swedish government
  statistics: under the Swedish Copyright Act (1960:729, ch. 1 §9) official decisions
  and reports are not protected by copyright, but compilation/database rights may apply.
  This repository treats the values as public with attribution to Riksgalden; the
  redistribution status is stated as unverified rather than asserted.
- **Known limitations:** frozen 2026-07-24; outstanding nominal dates from the September
  2025 report; the app excludes the 2039 and 2071 bonds because they mature beyond the
  last curve pillar.

## fred_treasury_cmt

**Classification:** public

- **Publisher:** Federal Reserve Bank of St. Louis (FRED); the underlying series are
  produced by the Board of Governors of the Federal Reserve System (US) (H.15 release).
- **Primary URL:** `https://fred.stlouisfed.org/graph/fredgraph.csv`
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 (last published business-day values as of retrieval).
- **Raw fields and units:** series `DGS1MO` … `DGS30` — U.S. Treasury constant-maturity
  (CMT) **par yields**, percentage values. CSV columns: `series_id` (FRED series
  identifier), `tenor_years` (tenor in years; 1M stored as 0.0833), `rate` (decimal
  yield, 0.038 = 3.8%).
- **Transformation:** CMT par yields divided by 100 to decimals; the one-month tenor is
  stored as a fractional year. Any curve built from these inputs is a CMT-implied
  approximation, not an official Treasury bootstrap.
**Licence/redistribution:** verified facts with a recorded limitation — the underlying
  CMT rates are U.S. Government work, which is not subject to copyright (17 U.S.C. §105;
  public domain, citation requested). FRED's [Terms of
  Use](https://fred.stlouisfed.org/legal/) (retrieved 2026-08-05) prohibit storing,
  caching or archiving FRED content, data mining or scraping, and commercial use without
  permission, and state that series may be owned by third parties; FRED does not license
  redistribution of the underlying data. The public-domain status of the Treasury rates
  is asserted; the FRED retrieval terms are recorded as a limitation, not overridden.
- **Known limitations:** frozen 2026-07-24; the one-month tenor is stored as a
  fractional year; the FRED terms restrict archiving and scraping of the retrieval
  service itself.

## fred_treasury_cmt_history

**Classification:** public

- **Publisher:** Federal Reserve Bank of St. Louis (FRED); underlying series produced by
  the Board of Governors of the Federal Reserve System (US) (H.15 release).
- **Primary URL:** `https://fred.stlouisfed.org/graph/fredgraph.csv`
- **Retrieval date:** 2026-07-24
- **Observation date:** daily observations from 2021-07-25 through 2026-07-24 (five
  years ending at the snapshot date; last observation 2026-07-24).
- **Raw fields and units:** the same CMT par-yield series as `fred_treasury_cmt`, daily.
  CSV columns: `date` (ISO 8601 observation date), `tenor_years` (tenor in years), `rate`
  (decimal yield).
- **Transformation:** CMT par yields divided by 100 to decimals; observations stored
  long-form over date × tenor; U.S. market holidays are dropped (no value is published
  for them).
**Licence/redistribution:** verified facts with a recorded limitation — identical to
  `fred_treasury_cmt`: public-domain U.S. Government work, retrieved via FRED whose
  terms restrict archiving and scraping of the service (retrieved 2026-08-05).
- **Known limitations:** frozen 2026-07-24; five-year window ending that date; used for
  the PCA and historical-risk derivations in the app.

## usd_ois_swaps

**Classification:** constructed

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no public source for the complete grid — construction record lives in
  this repository (`https://github.com/wididestrianda01/fixed-income-curve-engine`); the
  base input is the FRED Treasury CMT series above.
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 (as-of date of the construction; the CMT input is
  that day's published values and the spread is that day's close).
- **Raw fields and units:** CSV columns: `tenor_years` (tenor in years), `par_rate`
  (decimal par rate, 0.0412 = 4.12%).
- **Transformation:** `par_rate = treasury_cmt_rate + ois_tsy_spread`, applied per
  tenor, where the spread is taken from the Bloomberg generic `USSW` vs `USGG` screen at
  the 2026-07-24 close (approximate, ±2 bp).
**Licence/redistribution:** unverified — constructed from public-domain Treasury CMT
  data plus a spread transcribed from a Bloomberg indicative screen; Bloomberg terminal
  data is not freely redistributable and its terms were not verified for this mark. No
  third-party market-data feed is redistributed; the grid is committed as a dated
  educational mark.
- **Known limitations:** constructed, not observed live quotes; approximate (±2 bp); not
  a tradable mark; the spread input is an indicative screen.

## usd_forecast_basis

**Classification:** constructed

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no public source — the input is an indicative screen; construction
  record lives in this repository
  (`https://github.com/wididestrianda01/fixed-income-curve-engine`).
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24 (as-of date of the construction).
- **Raw fields and units:** CSV columns: `tenor_years` (tenor in years), `basis_bp`
  (basis spread in basis points).
- **Transformation:** `basis_bp` = (3M Term SOFR − SOFR OIS) per tenor, from the
  Bloomberg `S490` screen indicative mid-market on 2026-07-24, rounded to the nearest
  0.5 bp. Downstream, the forecast curve is built as OIS par rate + `basis_bp / 1e4`.
**Licence/redistribution:** unverified — values transcribed from a Bloomberg indicative
  screen; the underlying Term SOFR benchmark values are proprietary to the benchmark
  administrator and are not committed here (only the derived, rounded basis points are).
  Redistribution terms of the screen were not verified.
- **Known limitations:** constructed, not observed live quotes; representative, not live
  tradable marks; the exact basis is quoted inter-dealer and varies intraday.

## ecb_spot_curve

**Classification:** public

- **Publisher:** European Central Bank
- **Primary URL:**
  `https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.<DATA_TYPE_FM>`
  (SDMX 2.1 REST; spot rates are the `SR_<TENOR>` data flows)
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24
- **Raw fields and units:** AAA-rated euro-area government bond spot curve, Svensson
  model, continuously compounded, yield-error minimisation. The API publishes percentage
  rates. CSV columns: `tenor_years` (tenor in years, 1–30), `zero_rate` (decimal
  continuously compounded zero rate, 0.0264139995 = 2.64139995%).
- **Transformation:** published percentage zero rates divided by 100 to decimals;
  continuously compounded per the ECB technical notes; no sub-1Y tenors are published as
  separate spot-rate series (the Svensson model extends analytically to the short end).
**Licence/redistribution:** verified — [Policy regarding the reuse of ESCB
  statistics](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)
  (retrieved 2026-08-05): "All publicly available ESCB statistics may be reused free of
  charge on the condition that the source is quoted (e.g. 'Source: ECB statistics.') and
  that the statistics (including metadata) are not modified"; free reuse does not apply
  to third-party data, and these series are ECB statistics.
- **Known limitations:** frozen 2026-07-24; no sub-1Y spot series published; the curve is
  a model fit, not a set of directly observed zero rates.

## ecb_svensson_parameters

**Classification:** public

- **Publisher:** European Central Bank
- **Primary URL:**
  `https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.<DATA_TYPE_FM>`
  (same SDMX 2.1 endpoint as `ecb_spot_curve`; data flows `BETA0`, `BETA1`, `BETA2`,
  `BETA3`, `TAU1`, `TAU2`)
- **Retrieval date:** 2026-07-24
- **Observation date:** 2026-07-24
- **Raw fields and units:** CSV columns: `parameter` (Svensson parameter name), `value`
  (BETA parameters in percent, TAU parameters in years).
- **Transformation:** stored as published: BETA parameters are percentage values
  (e.g. BETA0 = 1.326064682 means 1.326…%) and must be divided by 100 before the Svensson
  formula; TAU parameters are in years and used directly; `%.10g` formatting applied.
**Licence/redistribution:** verified — same ECB reuse policy as `ecb_spot_curve`
  (retrieved 2026-08-05): free reuse with source quoted and statistics unmodified.
- **Known limitations:** frozen 2026-07-24; betas are stored in percent and must be
  divided by 100 downstream.

## illustrative_swaption_vols

**Classification:** illustrative

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no external source — construction record and generator live in this
  repository (`https://github.com/wididestrianda01/fixed-income-curve-engine`).
- **Retrieval date:** 2026-07-24 (generation as-of; the grid is generated, not
  retrieved)
- **Observation date:** none — the values are fabricated with a documented shape, not
  observed.
- **Raw fields and units:** CSV columns: `expiry` (ISO 8601 option expiry date),
  `maturity` (ISO 8601 swap maturity date), `vol` (normal volatility in basis points).
- **Transformation:** `sigma_bp(e, m) = (60.0 + 22.0 * e * exp(1 - e / 1.5)) * exp(-0.018 * m)`
  with e the option expiry in years and m the underlying swap tenor in years, run over
  expiries (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0) and swap tenors (1.0, 2.0, 5.0, 10.0).
  Expiry/maturity dates are `asof + floor(years * 365.25)` days. Fully deterministic;
  regenerate with `python scripts/build_illustrative_vols.py --write-packaged` (the
  packaged resource is read-only at runtime, so regeneration is a deliberate maintainer
  action that must also update the manifest sha256; bare invocation only prints the
  generated grid).
**Licence/redistribution:** verified — wholly generated in this repository from the
  stated closed form; no third-party data is involved and no redistribution restrictions
  apply to the values themselves (they are not market data).
- **Known limitations:** illustrative, not observed live quotes; not market data; not a
  fit to any traded price; the shape is market-plausible by construction only.

## illustrative_swaption_smile

**Classification:** illustrative

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no external source — construction record and generator live in this
  repository (`https://github.com/wididestrianda01/fixed-income-curve-engine`).
- **Retrieval date:** 2026-07-24 (generation as-of; the grid is generated, not retrieved)
- **Observation date:** none — the values are fabricated with a documented shape, not
  observed.
- **Raw fields and units:** CSV columns: `expiry` (ISO 8601 option expiry date),
  `maturity` (ISO 8601 swap maturity date), `strike` (decimal strike rate, 0.03 = 3%),
  `vol` (normal volatility in basis points).
- **Transformation:** `sigma_N(K; e, m) = sigma_atm(e, m) * (1 - 0.40 * u + 0.25 * u^2 + 0.06 * u^4)`
  with `u = (K - 0.03) / 0.03`, the ATM construction shared with
  `illustrative_swaption_vols`, run over expiries (1.0, 2.0, 5.0) on swap tenor 5.0 and
  strikes the ATM forward plus deltas (-0.015 to +0.015) in 50 bp steps. The quartic
  term is small and kept deliberately so a SABR fit leaves a measured residual rather
  than an exact planted fit. Fully deterministic; regenerate with
  `python scripts/build_illustrative_smile.py --write-packaged`.
**Licence/redistribution:** verified — wholly generated in this repository from the
  stated closed form; no third-party data is involved and no redistribution restrictions
  apply to the values themselves (they are not market data).
- **Known limitations:** illustrative, not observed live quotes; not market data; not a
  fit to any traded price; the skew and convexity are market-plausible by construction
  only.

## illustrative_inflation_breakevens

**Classification:** illustrative

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no external source — construction record and generator live in this
  repository (`https://github.com/wididestrianda01/fixed-income-curve-engine`).
- **Retrieval date:** 2026-07-24 (generation as-of; the grid is generated, not
  retrieved)
- **Observation date:** none — the values are fabricated with a documented shape, not
  observed.
- **Raw fields and units:** CSV columns: `tenor_years` (tenor in years),
  `breakeven` (decimal continuously compounded zero-coupon breakeven rate,
  0.023 = 2.3%).
- **Transformation:** `breakeven(T) = 0.023 + 0.012 * (T / 3.0) * exp(1 - T / 3.0)`
  with T the tenor in years, run over tenors (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0,
  20.0, 30.0). Continuously compounded: a nominal zero rate `n(T)` and this
  breakeven `b(T)` give the real zero rate `r(T) = n(T) - b(T)`. Fully deterministic;
  regenerate with `python scripts/build_illustrative_inflation.py --write-packaged`.
**Licence/redistribution:** verified — wholly generated in this repository from the
  stated closed form; no third-party data is involved and no redistribution restrictions
  apply to the values themselves (they are not market data).
- **Known limitations:** illustrative, not observed live quotes; not market data; not a
  fit to any traded inflation price and not a CPI forecast; the humped shape is
  market-plausible by construction only.

## illustrative_xccy_basis

**Classification:** illustrative

- **Publisher:** constructed in this repository; no publisher observes these values.
- **Primary URL:** no external source — construction record and generator live in this
  repository (`https://github.com/wididestrianda01/fixed-income-curve-engine`).
- **Retrieval date:** 2026-07-24 (generation as-of; the grid is generated, not retrieved)
- **Observation date:** none — the values are fabricated with a documented shape, not
  observed.
- **Raw fields and units:** CSV columns: `tenor_years` (tenor in years), `basis_bp`
  (basis spread in basis points).
- **Transformation:** `basis_bp(t) = -28.0 * (1 - exp(-t / 3.0))` with t the tenor in
  years, run over tenors (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0).
  This is the EUR/USD cross-currency basis, the spread added to the EUR (€STR) leg of a
  cross-currency basis swap against USD SOFR flat; the negative sign is the
  USD-funding-premium convention. Fully deterministic; regenerate with
  `python scripts/build_illustrative_xccy.py --write-packaged`.
**Licence/redistribution:** verified — wholly generated in this repository from the
  stated closed form; no third-party data is involved and no redistribution restrictions
  apply to the values themselves (they are not market data).
- **Known limitations:** illustrative, not observed live quotes; not market data; not a
  fit to any traded price; the sign and shape are market-plausible by construction only.

---

## Known gaps and exclusions

The SEK market is incomplete in free, redistributable sources: the Riksbank publishes
government benchmark yields at only four tenors (2Y, 5Y, 7Y, 10Y); the 1Y, 15Y, 20Y and
30Y tenors are not published through any free API. STIBOR is no longer published by the
Riksbank (it moved to the Swedish Financial Benchmark Facility in May 2020), and no free
machine-readable feed is available. FRA quotes and SEK swap rates require a commercial
terminal.

That is why the project uses two markets (SEK + EUR/USD): the SEK curve is built from
the four bond benchmarks, three bill rates, SWESTR and the outstanding-bond reference
data; the EUR curve is fully specified (14 spot tenors plus Svensson parameters); the
USD curve is constructed from Treasury CMT par yields plus a dated OIS spread and a
Term-SOFR basis — enough for a two-curve (discount + forecast) construction without any
licensed market-data feed.

The snapshot is frozen and fully offline: there is no network access path in the
package and no download or update instructions are provided. Constructed datasets are
reproducible from the records above; nothing in this repository updates market data.
