# P3 — Regulatory Context Note

Scope: a factual, current map of the EU/Swedish regulatory perimeter that an observer might associate with this project's methodology, and of why the project itself is not a regulated model. Every regulatory claim below was checked against a primary source on 2026-08-05 (source URL and what was verified are given per claim; items that could not be verified are marked explicitly). Nothing in this note is a claim that the toolkit requires regulatory approval or validation.

## Jurisdiction and status of the project

The repository's market data, conventions and academic framing are Swedish/EU (Riksbank and Riksgalden data, EU 2024/856 scenarios, EUR/SEK/USD instruments), so EU/Sweden is the relevant regulatory perimeter to discuss. Within that perimeter:

- **Finansinspektionen (FI)** is the Swedish financial supervisory authority and the national competent authority for credit institutions, insurers and occupational pension institutions in Sweden. (Verified: fi.se "About FI" — "FI is a government authority tasked with monitoring the financial market", accessed 2026-08-05, https://www.fi.se/en/about-fi/. The earlier misspelling "Finansinspositionen" is corrected here and must not reappear in any repository document.)
- **This project is not a regulated model.** It is an educational toolkit. Nothing it produces is a regulatory disclosure, a capital number, an AVA, or an input that a named institution would file. The SEB, Handelsbanken and Nordea names from the original portfolio brief are **not** treated here as users of this project's output; the project is not "a live Article 448 topic" for any institution, and this note deliberately says nothing about the regulatory obligations of any named firm.

## 1. Banks: CRR/CRD framework and IRRBB

### 1.1 What the current framework actually requires

- **CRD Article 84** ("Interest risk arising from non-trading book activities") requires *competent authorities to ensure that institutions implement* internal systems or the standardised methodology to identify, evaluate, manage and mitigate IRRBB affecting both economic value of equity (EVE) and net interest income (NII). Article 84(5) mandates the EBA standardised-methodology RTS, adopted as **Commission Delegated Regulation (EU) 2024/857** (specifying the standardised and simplified standardised methodologies; verified against EUR-Lex CELEX 32024R0857). DR 2024/857 recital (2) also confirms that EBA/GL/2018/02 remains the operative reference for the management aspects. (Both verified 2026-08-05, consolidated CRD CELEX 02013L0036-20250117 and DR 2024/857.)
- **CRD Article 98(5)** puts IRRBB in the SREP and requires supervisory powers where EVE declines by more than 15% of Tier 1 capital under the six supervisory shock scenarios, or where NII suffers a "large decline" (Article 5 of DR 2024/856 defines it as a one-year NII decline >5% of Tier 1). **Article 98(5a)** mandates the EBA RTS on the six EVE scenarios, the two NII scenarios per currency, and the common modelling and parametric assumptions. (Verified, same source.)
- **Commission Delegated Regulation (EU) 2024/856** of 1 December 2023 (OJ L, 24.4.2024; in force 14 May 2024) is that RTS: six supervisory shock scenarios (parallel up/down, short-rate up/down, steepener, flattener; Article 1), the $e^{-t/4}$ short-rate parameterisation and rotation weights (Article 2), the common EVE modelling/parametric assumptions including the Article 3(7) maturity-dependent post-shock floor (from −150 bp at immediate maturity, +3 bp/year, 0% at 50 years, observed rate kept when lower), and the "large decline" definition (Article 5). Its Annex Part A lists per-currency shock sizes — USD 200/300/150 bp and SEK 200/300/150 bp (parallel/short/long) — as a flat currency list; the final regulation does **not** use the label "Category 1" or any category grouping. (Verified against EUR-Lex CELEX 32024R0856 PDF, accessed 2026-08-05.)
- **CRR Article 448** ("Disclosure of exposures to interest rate risk on positions not held in the trading book"), as amended by CRR2 and in force since 28 June 2021, requires institutions to disclose the changes in EVE under the six supervisory shock scenarios and the changes in NII under the two supervisory shock scenarios of CRD Article 98(5), plus key modelling and parametric assumptions. The original-2013 wording ("variation in earnings, economic value or other relevant measure... broken down by currency") is historical and must not be quoted as current. (Verified against the consolidated CRR, EUR-Lex CELEX 02013R0575-20250101, accessed 2026-08-05.)
- **EBA/GL/2018/02** (Guidelines on the management of interest rate risk arising from non-trading book activities, 18 July 2018, applicable from 30 June 2019) reflected the BCBS-specified supervisory shock scenarios; DR 2024/856 recital (1) says the new regulation "builds on that specification and methodology". The six-scenario parameters are now binding through DR 2024/856. (Verified via EU 2024/856 recital (1) and footnote (3), EUR-Lex, accessed 2026-08-05. Note: the EBA website's own search was unavailable due to maintenance on 2026-08-05, so the GL's standalone EBA-page status could not be re-checked there; nothing found indicates repeal.)

### 1.2 Basel IRRBB: current document and parameters (verification result)

The brief asked to verify the document number (d578 vs d568). Result, verified on bis.org (accessed 2026-08-05):

- **d568 is NOT the IRRBB standard.** bis.org/bcbs/publ/d568.htm is "Transparency and responsiveness of initial margin in centrally cleared markets: review and policy proposals" (16 January 2024, consultative report, BCBS-CPMI-IOSCO).
- The current IRRBB standard is **BCBS d368, "Interest rate risk in the banking book", 21 April 2016** (status: Consolidated — integrated into the Basel Framework, chapter SRP31, with derivation in SRP98).
- **d578, "Recalibration of shocks in the interest rate risk in the banking book standard", 16 July 2024** (status: Consolidated; implement by 1 January 2026) is the 2024 recalibration standard. It replaces the global shock factors with local currency-specific factors, moves from the 99th to the 99.9th percentile, reduces rounding to multiples of 25 bp, and extends the calibration data to December 2023. Its final Table 2 (SRP31.90) specifies both EVE shocks (R) and NII shocks (S) per currency. (Verified against d578.pdf pages 3-4 and d368.pdf Annex 2, Table 1.)

**Parameter comparison for the currencies this project uses (bp, parallel/short/long):**

| Currency | EU DR 2024/856 Annex Part A (in force 14.5.2024) | Basel d368 Table 1 (current calibration) | Basel d578 Table 2 (from 1.1.2026): EVE (R) | Basel d578 NII (S) |
|---|---|---|---|---|
| USD | 200 / 300 / 150 | 200 / 300 / 150 | 200 / 300 / 150 | 200 / 300 / 225 |
| SEK | 200 / 300 / 150 | 200 / 300 / 150 | 200 / 300 / 150 | 275 / 425 / 200 |
| EUR | 200 / 250 / 100 | 200 / 250 / 100 | 200 / 250 / 100 | 225 / 350 / 200 |

Sources: EU: EUR-Lex CELEX 32024R0856 Annex Part A. Basel: d368.pdf Annex 2 Table 1 (EUR 200/250/100, USD 200/300/150, SEK 200/300/150); d578.pdf SRP31.90 Table 2 (single-value cells mean R=S; two-value cells are R then S). All accessed 2026-08-05.

So for USD and SEK the EU 2024/856 EVE shock sizes **coincide** with the Basel EVE shock sizes, in both the current (d368) and recalibrated (d578) Basel calibrations — the EU RTS was calibrated to build on the BCBS specification (recital (1) of 2024/856). The two frameworks are nevertheless distinct instruments with different legal effects and different supporting parameters (the EU floor of Article 3(7); the Basel floors at the discretion of national supervisors). They must not be merged into a "BCBS-EBA" label: the project implements the **EU** regulation's scenarios, and references Basel parameters only in this comparative note.

Documentation note: d578's own derivation table (SRP98.59) shows intermediate shock values (e.g. USD 197/279/131) that do not round consistently to its final Table 2; the final SRP31.90 Table 2 is cited here as authoritative, and the discrepancy is recorded rather than re-litigated.

### 1.3 Where the project sits relative to that framework

- The scenario module implements the six EU 2024/856 shock shapes with the USD/SEK parameters and the Article 3(7) floor. That is the *shape layer* of the framework's shock set, nothing more: it is not an institution-wide supervisory outlier test (no balance sheet, no EVE aggregation, no 15%-of-Tier-1 threshold), not IRRBB compliance, and not a disclosure.
- The framework covers both legs of IRRBB — economic value and **earnings/NII** — and a real bank's framework includes non-maturity-deposit behavioural modelling. This project has no NII model and no NMD behavioural model; its outputs cover the economic-value side only. That boundary is stated wherever "IRRBB-style" language could appear.
- Regulatory validation of a bank's internal IRRBB measurement system requires, per the BCBS principles and CRD Pillar 2 expectations, an independent validation function, backtesting against realised P&L, and escalation on outlier-test breaches. This project's verification (re-pricing reconciliation, cross-library checks) is an internal-consistency exercise; it is not that process, and nothing here claims to be.
- **CRR Articles 313-314 are unrelated to model validation**: they define the operational-risk "business indicator component" and "business indicator" (BIC/BI) used in the new standardised approach for operational risk. A Hull-White pricing model has nothing to do with them; any text implying they govern model governance should cite instead CRD Articles 74 (governance), 84 (IRRBB) and 101 (ICAAP) for banks' internal-model obligations. (Verified against consolidated CRR, EUR-Lex CELEX 02013R0575-20250101, accessed 2026-08-05.)
- **Prudent valuation / AVA**: CRR Article 105 (as amended) applies prudent valuation standards to trading-book positions and non-trading-book positions measured at fair value, with additional valuation adjustments for complex products (Article 105(13)) and an EBA RTS mandate (Article 105(14), adopted as Commission Delegated Regulation (EU) 2016/101). This is an institution-level obligation on positions an institution holds; a curve-stress table in a notebook is not an AVA calculation, and no capital treatment is implied by anything in this repository. (Verified against consolidated CRR, accessed 2026-08-05.)

## 2. Insurance: Solvency II (Directive 2009/138/EC, consolidated)

Verified against the consolidated directive (EUR-Lex CELEX 02009L0138-20250117, accessed 2026-08-05):

- **Article 77(2)** — the best estimate of technical provisions uses "the relevant risk-free interest rate term structure".
- **Article 77a** — the relevant risk-free term structure must use deep, liquid and transparent market information where it exists and be **extrapolated** beyond; the extrapolation methodology is specified in delegated acts (EIOPA's prescribed risk-free rates, Smith-Wilson toward the Ultimate Forward Rate beyond the Last Liquid Point, published monthly by EIOPA).
- **Article 44(2a)(a)** — the ORSA must include a regular assessment of "the sensitivity of their technical provisions and eligible own funds to the assumptions underlying the extrapolation of the relevant risk-free interest rate term structure referred to in Article 77a" (paragraph 2a was inserted by the Omnibus II Directive 2014/51/EU).
- The SCR interest-rate risk sub-module applies relative up/down shocks to the risk-free curve (per the standard formula; EIOPA's underlying-assumptions document EIOPA-14-322 describes the calibration).

Where the project sits: for an insurer's liability discounting, the discount curve is the EIOPA-prescribed risk-free term structure, not a market-bootstrapped or parametric-fit curve; the two diverge most at long tenors, which is exactly where the project's KRD grid and long-maturity instruments live. The project's curve output is therefore not valuation-ready for a Solvency II liability book, and its scenario shocks are not an SCR interest-rate capital calculation (different magnitudes, shapes, and aggregation). No named insurance group is assigned to Solvency II in this note; the framework is described for whoever might use the toolkit's outputs, not for any named firm.

## 3. Occupational pensions: IORP II (Directive (EU) 2016/2341)

Verified against the directive text (EUR-Lex CELEX 32016L2341, OJ L 354, 23.12.2016, accessed 2026-08-05):

- **Article 1** — the directive governs the taking-up and pursuit of activities of institutions for occupational retirement provision (IORPs).
- **Article 13** — *technical provisions* (note: Article 13, not 15; Article 15 is "Regulatory own funds"): IORPs must hold adequate liabilities/sufficient technical provisions (paragraphs 1-2), calculated on "sufficiently prudent actuarial valuation" assumptions with "an appropriate margin for adverse deviation" (paragraph 4(a)); the **maximum rates of interest** used "shall be chosen prudently and determined in accordance with any relevant rules of the home Member State", taking into account asset yields and high-quality bond yields (paragraph 4(b)); Member States may impose additional, more detailed requirements (paragraph 5). Recitals (40)-(41) make the same point: prudent calculation of technical provisions, with the calculation subject to additional national rules.

So IORP II requires prudent actuarial technical provisions and deliberately leaves the discount-rate detail to Member States — the opposite of Solvency II's harmonised EIOPA curve. For an occupational pension fund, the "correct" discount methodology is a national-law question, not something this toolkit can assert. No named pension provider is assigned to IORP II in this note.

## 4. Swedish AP funds: national statute, not IORP II or Solvency II

Verified against the current statute text on riksdagen.se (Lag (2000:192) om allmänna pensionsfonder, amended through SFS 2026:907, accessed 2026-08-05):

- Chapter 1, Section 1 of the Act now names **four** funds: the Second, Third, Fourth and Seventh AP funds (Andra, Tredje, Fjärde och Sjunde AP-fonden; amendment Lag 2025:376). The **First AP Fund has been wound down** (it no longer appears in the Act), and the Seventh AP Fund (premium-pension default fund) is governed by this Act.
- The funds are state authorities under Swedish national law, outside both IORP II (Article 1 scope: IORPs under the directive) and Solvency II (Article 2 scope: insurance and reinsurance undertakings). An AP fund's discounting and investment rules are set by the Act and its ordinances, not by either EU framework.

The original portfolio brief's "AP1-AP4" list is outdated and must not be repeated; the current, verifiable statement is "AP2, AP3, AP4 and AP7 under Lag (2000:192)".

## 5. Frameworks considered and excluded

Each is excluded with its scope provision cited (all EUR-Lex texts, accessed 2026-08-05):

- **GDPR (Regulation (EU) 2016/679).** Article 2(1): the GDPR applies to the processing of *personal data*. The specified pipeline — quoted bond/swap rates, portfolio holdings by instrument ID and notional, curve and scenario definitions — contains no personal data of identifiable natural persons at any point. Not applicable as specified; would need re-assessment only if a future extension attached named client/counterparty data to the portfolio holdings file. (Reasoned from GDPR Article 2(1); the absence of personal data is a fact about this pipeline, not a claim about GDPR's text.)
- **MiFID II (Directive 2014/65/EU).** Article 1(1): the directive applies to *investment firms, market operators, data reporting services providers*, and third-country firms providing investment services/activities. The toolkit provides no investment services or activities (no advice, no order flow, no execution logic; it is local analytics plus a local Streamlit viewer). Not applicable to the toolkit itself; the same exclusion holds for the persons running it to the extent they are not conducting MiFID business.
- **EMIR (Regulation (EU) No 648/2012).** Article 1(1)-(2): EMIR lays down clearing, bilateral risk-management and reporting requirements for *OTC derivative contracts* and applies to CCPs, clearing members, financial counterparties, trade repositories and, where provided, non-financial counterparties and trading venues. The project uses swap quotes only as curve-construction input data and does not originate, hold, clear or report a derivative position. Not applicable to the toolkit.
- **IFRS 13 (Fair Value Measurement).** IFRS 13 applies to entities reporting under IFRS when they measure fair value, and classifies fair-value measurements into a three-level hierarchy based on the lowest-level input that is *significant* to the measurement (paragraphs 72-74). Interpolated/extrapolated curve points are unobservable inputs (Level 3 *inputs*), but there is **no automatic classification**: the hierarchy level of the measurement follows input significance. The toolkit performs no entity-level fair-value measurement, so IFRS 13 attaches only if a user's own accounting context makes it relevant — and even then the automatic-classification claim would be wrong. (Note: the IFRS 13 standard text requires registration on ifrs.org; the paragraph references and substance are cited from knowledge of the standard and were not re-fetched on 2026-08-05 — flagged as such per the verification discipline of this note.)

## 6. Validation and QuantLib parity

The README, spec and theory notes describe the cross-library comparisons (selected bond prices/yields, a selected Hull-White swaption NPV, ECB Svensson parameter reconstruction) as **software verification checks — selected implementation cross-checking, not independent empirical or regulatory model validation**. The repository never describes them as "establishing correctness" of a model against markets; that framing is binding across all documents.
