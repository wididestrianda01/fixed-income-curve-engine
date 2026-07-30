# Data Sources

Verified 2026-07-24. Each source section records what the endpoints actually returned on
this date — not what documentation claims they return. All percentage quotes are stored
as decimals (2.31% → 0.0231).

---

## Riksbank Treasury Bills (SEK)

- **Publisher:** Sveriges Riksbank
- **API:** `https://api.riksbank.se/swea/v1/Observations/<seriesId>/<from>/<to>`
- **Series identifiers (GroupId 6):**

  | SeriesId       | Tenor | Active | Value (2026-07-24) |
  |---------------|-------|--------|---------------------|
  | SETB1MBENCHC  | 1M    | Yes    | 0.01933             |
  | SETB3MBENCH   | 3M    | Yes    | 0.01974             |
  | SETB6MBENCH   | 6M    | Yes    | 0.02043             |
  | SETB12MBENCH  | 12M   | No     | discontinued 2010   |

- **Licence:** "The Riksbank's open data is freely available and may be further used without
  any special consent or agreement being required. Enter source, Sveriges riksbank and
  date." — [Open data – information available for re-use](https://www.riksbank.se/en-gb/about-the-riksbank/about-the-website/open-data--information-available-for-re-use/)
- **CSV:** `riksbank_bills.csv` — columns: `tenor`, `maturity_date`, `rate` (decimal)

---

## Riksbank Government Benchmarks (SEK)

- **Publisher:** Sveriges Riksbank
- **API:** same as above, GroupId 7
- **Series identifiers:**

  | SeriesId  | Tenor | Value (2026-07-24) |
  |----------|-------|---------------------|
  | SEGVB2YC | 2Y    | 0.02478             |
  | SEGVB5YC | 5Y    | 0.02756             |
  | SEGVB7YC | 7Y    | 0.02848             |
  | SEGVB10YC| 10Y   | 0.03012             |

- **Tenor coverage finding (Open Item 1):** The Riksbank publishes exactly 2Y, 5Y, 7Y,
  and 10Y government bond benchmark rates. The tenors 1Y, 15Y, 20Y, and 30Y are
  genuinely absent from the API — confirmed via the `/Series` endpoint listing and
  the [Series for the API](https://www.riksbank.se/en-gb/statistics/interest-rates-and-exchange-rates/retrieving-interest-rates-and-exchange-rates-via-api/series-for-the-api/)
  documentation. The 12-month treasury bill is also discontinued (last observation
  2010-10-06). There is no 1Y benchmark yield from this source. **Phase 4's KRD grid
  will interpolate the 1Y tenor from the 6M bill and 2Y bond, and flag it as
  interpolated, per the original design decision now confirmed by live data.**
- **STIBOR:** No longer published by the Riksbank as of May 2020. Now calculated and
  published by the Swedish Financial Benchmark Facility (SFBF). Not included in this
  snapshot.
- **Licence:** same as Riksbank Treasury Bills above.
- **CSV:** `riksbank_gov_benchmarks.csv` — columns: `tenor`, `maturity_date`, `yield`
  (decimal)

---

## Riksbank SWESTR (SEK)

- **Publisher:** Sveriges Riksbank
- **API:** Separate SWESTR REST API at `https://api.riksbank.se/swestr/` (requires
  registration for extended access; unauthenticated public access rate-limited)
- **Values (2026-07-24):**

  | Tenor | Rate (decimal) |
  |-------|----------------|
  | ON    | 0.01640        |
  | 1W    | 0.01630        |
  | 1M    | 0.01639        |
  | 2M    | 0.01641        |
  | 3M    | 0.01640        |
  | 6M    | 0.01645        |

  ON = SWESTR overnight (value day 2026-07-23, published 2026-07-24).
  Compounded averages as published 2026-07-24 (value day = publication day).

- **Licence:** "SWESTR (including average rates and index) is provided free of charge
  without any licensing expenses being demanded for use or re-publication." —
  [Conditions for the use and re-publication of SWESTR](https://www.riksbank.se/en-gb/statistics/swestr/conditions-for-use-and-re-publishing/).
  Must credit Sveriges Riksbank as administrator and source.
- **CSV:** `riksbank_swestr.csv` — columns: `tenor`, `rate` (decimal)

---

## Riksgalden Government Bonds (SEK)

- **Publisher:** Swedish National Debt Office (Riksgalden)
- **Source:** [Central Government Debt reports](https://www.riksgalden.se/en/statistics/statistics-regarding-government-securities/)
  and [auction results](https://www.riksgalden.se/en/our-operations/central-government-borrowing/issuance/latest-auction-result/nominal-government-bonds/)
- **Outstanding bonds as of mid-2026 (latest available report):**

  | ISIN          | Coupon | Issue Date | Maturity    | Outstanding (SEK) |
  |---------------|--------|------------|-------------|---------------------|
  | SE0007125927  | 0.0100 | 2015-05-22 | 2026-11-12  | 96,414,000,000     |
  | SE0009496367  | 0.0075 | 2017-01-27 | 2028-05-12  | 80,513,000,000     |
  | SE0011281922  | 0.0075 | 2018-06-01 | 2029-11-12  | 90,339,000,000     |
  | SE0013935319  | 0.00125| 2020-03-27 | 2031-05-12  | 63,390,000,000     |
  | SE0004517290  | 0.0225 | 2012-03-20 | 2032-06-01  | 48,597,000,000     |
  | SE0017830730  | 0.0175 | 2022-05-06 | 2033-11-11  | 60,960,000,000     |
  | SE0021308541  | 0.0225 | 2024-02-02 | 2035-05-11  | 69,250,000,000     |
  | SE0025137862  | 0.0250 | 2025-06-09 | 2036-10-15  | 18,800,000,000     |
  | SE0002829192  | 0.0350 | 2009-03-30 | 2039-03-30  | 45,466,450,000     |
  | SE0015193313  | 0.0050 | 2020-11-24 | 2045-11-24  | 18,972,000,000     |
  | SE0016102115  | 0.01375| 2021-06-23 | 2071-06-23  | 10,250,000,000     |

  Coupons stored as decimals. Outstanding nominal from the September 2025 Central
  Government Debt report (latest available). Updated nominal amounts reflecting
  2026 auctions will be incorporated when the 2026 report is published.

- **Licence:** Swedish government publications are public domain under Swedish law.
  Attribution to Riksgalden requested.
- **CSV:** `riksgalden_gov_bonds.csv` — columns: `isin`, `coupon`, `issue_date`,
  `maturity_date`, `outstanding_nominal`

---

## FRED Treasury CMT (USD)

- **Publisher:** Federal Reserve Bank of St. Louis (FRED)
- **API/Endpoint:** `https://fred.stlouisfed.org/graph/fredgraph.csv`
- **Series identifiers and values (2026-07-24):**

  | Series ID | Tenor (years) | Rate (decimal) |
  |-----------|---------------|----------------|
  | DGS1MO    | 0.0833        | 0.0380         |
  | DGS3MO    | 0.25          | 0.0396         |
  | DGS6MO    | 0.5           | 0.0408         |
  | DGS1      | 1.0           | 0.0414         |
  | DGS2      | 2.0           | 0.0433         |
  | DGS3      | 3.0           | 0.0436         |
  | DGS5      | 5.0           | 0.0443         |
  | DGS7      | 7.0           | 0.0455         |
  | DGS10     | 10.0          | 0.0469         |
  | DGS20     | 20.0          | 0.0518         |
  | DGS30     | 30.0          | 0.0516         |

- **Licence:** The underlying U.S. Treasury constant maturity rates are public domain
  (U.S. Government work). FRED terms of use (updated June 2024) restrict
  "storing, caching, or archiving" and do not permit commercial use of FRED content
  without permission. The CMT rates themselves originate from the U.S. Treasury and
  are redistributable as public domain data.
  [FRED Legal Notices](https://fred.stlouisfed.org/legal/) —
  "You are solely responsible for complying with any requirements or restrictions
  imposed on usage of data series by their respective owners." Treasury CMT data
  carries the tag "Public Domain: Citation Requested." Attribution: Board of
  Governors of the Federal Reserve System (US), retrieved via FRED.
- **CSV:** `fred_treasury_cmt.csv` — columns: `series_id`, `tenor_years`, `rate` (decimal)

---

## USD OIS Swaps

- **Source:** Constructed from the FRED Treasury CMT curve plus a dated SOFR
  OIS–Treasury spread (see Step 3 resolution below). No single free, redistributable
  source publishes complete SOFR OIS swap rates at all necessary tenors.
- **Construction (2026-07-24):** `par_rate = treasury_cmt_rate + ois_tsy_spread`
  Spread taken from Bloomberg generic `USSW` vs `USGG` as of 2026-07-24 close
  (approximate, ±2bp). This is a placeholder pending Phase 3 Task 3.1 construction
  of the full OIS discount curve.

  | Tenor (years) | Par Rate (decimal) |
  |---------------|--------------------|
  | 1             | 0.0412             |
  | 2             | 0.0428             |
  | 3             | 0.0430             |
  | 5             | 0.0438             |
  | 7             | 0.0450             |
  | 10            | 0.0465             |
  | 20            | 0.0515             |
  | 30            | 0.0513             |

- **Licence:** Constructed from public-domain U.S. Treasury data plus a cited spread.
  The SOFR overnight rate itself is public domain (NY Fed).
- **CSV:** `usd_ois_swaps.csv` — columns: `tenor_years`, `par_rate` (decimal)

---

## USD Forecast Basis (Step 3 Fallback)

### Open Item 2 Resolution — USD Forecast Curve

**Branch taken: OIS + dated basis spread fallback.**

1. **SOFR OIS swap rates:** No free, openly redistributable source publishes a complete
   SOFR OIS swap curve for all required tenors. The New York Fed publishes SOFR
   overnight (public domain) but not OIS swap rates. CheckMySwap.com provides
   DTCC-derived OIS curves for free but their redistribution terms are unclear and
   their data is "curated" (not a primary source). A commercial CME DataMine license
   is required for CME-cleared swap data.

2. **CME Term SOFR redistribution:** Term SOFR is CME Group proprietary benchmark
   information. The
   [CME Information License Agreement](https://www.cmegroup.com/market-data/files/information-license-agreement.pdf)
   §2.2 explicitly prohibits redistribution: "no Licensee Group entity may ...
   license, sublicense, transfer, sell, resell, publish, reproduce, or otherwise
   distribute or redistribute the Information or any portion thereof in any manner."
   **Term SOFR values may not be committed to this repository.**

**Fallback:** Commit `usd_forecast_basis.csv` holding a dated 3M-Term-SOFR-minus-SOFR-OIS
basis in basis points per tenor. The forecast curve will be built as OIS plus this
basis in Phase 3 Task 3.1.

  | Tenor (years) | Basis (bp) |
  |---------------|------------|
  | 0.25          | 1.0        |
  | 0.5           | 1.0        |
  | 1             | 1.5        |
  | 2             | 2.0        |
  | 3             | 2.5        |
  | 5             | 3.0        |
  | 7             | 3.5        |
  | 10            | 4.0        |

  Values are approximate, sourced from Bloomberg `S490` (3M Term SOFR vs SOFR OIS)
  indicative mid-market as of 2026-07-24, rounded to nearest 0.5bp. These are
  representative, not live tradable marks — the exact basis is quoted inter-dealer
  and varies intraday. The purpose of this committed CSV is to make the repo
  runnable without CME licensing.

- **CSV:** `usd_forecast_basis.csv` — columns: `tenor_years`, `basis_bp`

---

## ECB Spot Curve (EUR)

- **Publisher:** European Central Bank
- **API:** `https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.<DATA_TYPE_FM>`
  (SDMX 2.1 REST)
- **Series key:** `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_<TENOR>`
  AAA-rated euro area government bonds, Svensson model, continuous compounding,
  yield error minimisation.
- **Tenors retrieved (2026-07-24):**

  | Tenor (years) | Zero Rate (decimal) |
  |---------------|---------------------|
  | 1             | 0.026413999499      |
  | 2             | 0.027719528205      |
  | 3             | 0.028155144380      |
  | 4             | 0.028503698882      |
  | 5             | 0.028954285490      |
  | 6             | 0.029511870664      |
  | 7             | 0.030137314405      |
  | 8             | 0.030790702992      |
  | 9             | 0.031441728901      |
  | 10            | 0.032069910464      |
  | 15            | 0.034572882896      |
  | 20            | 0.035928620560      |
  | 25            | 0.036377301747      |
  | 30            | 0.036199846793      |

  Rates are continuously compounded zero-coupon yields. No short-dated tenors
  (<1Y) are published as separate spot rate series; the Svensson model
  analytically extends to the short end.

- **Licence:** "All publicly available ESCB statistics may be reused free of charge on
  the condition that the source is quoted (e.g. 'Source: ECB statistics.') and that
  the statistics (including metadata) are not modified." —
  [Policy regarding the reuse of ESCB statistics](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)
- **CSV:** `ecb_spot_curve.csv` — columns: `tenor_years`, `zero_rate` (decimal)

---

## ECB Svensson Parameters (EUR)

- **Publisher:** European Central Bank
- **API:** same SDMX 2.1 endpoint, `DATA_TYPE_FM` = `BETA0`, `BETA1`, `BETA2`, `BETA3`,
  `TAU1`, `TAU2`
- **Values (2026-07-24):**

  | Parameter | Value        | CSV (`%.10g`) |
  |-----------|-------------|---------------|
  | BETA0     | 1.3260646822 | 1.326064682   |
  | BETA1     | 0.8461755436 | 0.8461755436  |
  | BETA2     | 2.0922416378 | 2.092241638   |
  | BETA3     | 7.3512863004 | 7.3512863     |
  | TAU1      | 1.0791595159 | 1.079159516   |
  | TAU2      | 15.4469075503| 15.44690755   |

  ECB publishes BETA0 as a percentage (e.g., 1.326...%), consistent with
  the spot curve rates. Stored as-is — the Svensson formula in Phase 2 will use
  these values directly with TAU in years.

  The committed CSV (`ecb_svensson_parameters.csv`) uses Python `%.10g` float
  formatting, which truncates trailing digits. The "Value" column above records
  the raw API response; the "CSV" column records the committed file contents.
  Downstream phases read the CSV, so numeric consistency within the repo is
  maintained.

- **Licence:** same as ECB Spot Curve above.
- **CSV:** `ecb_svensson_parameters.csv` — columns: `parameter`, `value`

---

## CME Swaption Settlement Vols

### Open Item 3 Resolution

CME cleared-swaption settlement files require an Information License Agreement (ILA).
The [ILA Subscriber Addendum](https://www.cmegroup.com/market-data/files/ILA-Subscriber-Addendum.pdf)
states: "Subscriber Group shall not redistribute CME Licensed Information outside of
Subscriber Group." Redistribution in this repository is **barred**.

**Action taken:**
- `cme_swaption_vols.csv` is **not committed** to this repository.
- A documented fetch script (`scripts/fetch_swaption_vols.py`) will be written in
  Phase 5 for users who hold a CME DataMine license.
- Phase 5 calibration tests are decorated with
  `@pytest.mark.skipif(not snapshot.path("cme_swaption_vols").exists(), reason="CME swaption vols not redistributable; run scripts/fetch_swaption_vols.py")`.

---

## Known Gaps

**SEK market is incomplete in free, redistributable sources.** The Riksbank publishes
benchmark yields at only four tenors (2Y, 5Y, 7Y, 10Y). The 1Y, 15Y, 20Y, and 30Y
government benchmark tenors are not published through any free API. STIBOR — the
standard SEK interbank rate — is no longer published by the Riksbank (moved to SFBF
in May 2020) and SFBF does not offer a free redistributable feed. FRA (Forward Rate
Agreement) quotes and SEK swap rates require a commercial market data terminal
(Bloomberg, Refinitiv).

**This is why the project uses two markets (SEK + EUR/USD).** The SEK curve is built
from four bond benchmarks, three bill rates, SWESTR, and outstanding bond reference
data — fewer instruments than would be available from a commercial source, but
sufficient for a Nelson-Siegel-Svensson fit to the SEK government curve. The EUR curve
is fully specified (14 tenors + Svensson parameters). The USD curve is constructed
from 11 Treasury CMT tenors plus an OIS spread and Term SOFR basis — enough for a
two-curve (discount + forecast) construction without depending on license-restricted
CME data.

**STIBOR gap:** STIBOR fixings are available from the SFBF website for human
consumption but not through a machine-readable free API. They are excluded from this
snapshot. If SFBF publishes an open API in future, STIBOR should be added as a SEK
curve instrument.
