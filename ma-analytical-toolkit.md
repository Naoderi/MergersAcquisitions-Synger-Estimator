# M&A Analytical Toolkit

Techniques, software and data sources used to assess a deal across four dimensions: price paid, deal economics, target quality, and post-deal outcomes.

---

## 1. Price Paid

### Techniques
- Comparable companies and precedent transactions analysis
- DCF, sum-of-the-parts, LBO analysis as a valuation floor
- Football-field / valuation range summary
- Contribution analysis (relative contribution vs. relative ownership in stock deals)
- VWAP and unaffected-price analysis; regression of premia against deal characteristics
- Value-creation split: synergy value captured by buyer vs. paid to seller

### Software and Data
- Capital IQ, Bloomberg, FactSet, LSEG, PitchBook, Mergermarket
- Excel with FactSet / CIQ plug-ins
- SEC EDGAR and Companies House filings

---

## 2. Deal Economics

### Techniques
- Accretion/dilution model with sources-and-uses and pro forma capitalisation
- Purchase price allocation and step-up modelling; deferred tax and intangible amortisation
- Breakeven synergy analysis (synergies needed to justify the premium)
- ROIC vs. WACC, EVA, deal IRR and NPV
- Sensitivity tables, tornado charts, Monte Carlo on price, synergies and phasing
- Scenario and downside cases; real options or decision-tree analysis for staged/earnout structures
- Debt capacity and covenant headroom modelling

### Software
- Excel / VBA operating and merger models; Python for Monte Carlo
- Anaplan, Pigment, Quantrix, Mosaic for driver-based pro forma consolidation
- Model audit tools: Operis, Spreadsheet Advantage, PerfectXL

---

## 3. Target Quality

### Techniques
- Quality-of-earnings on transaction-level GL; add-back and cut-off testing
- Cohort analysis: gross/net revenue retention, logo churn, LTV/CAC
- Survival analysis (Kaplan-Meier, Cox regression) on churn
- Concentration metrics: top-10 customer share, Herfindahl index
- Price-volume-mix decomposition
- Working capital diagnostics: DSO/DIO/DPO, monthly NWC peg
- Forensic screens: Benford's law, Beneish M-score, Altman Z-score
- Win/loss analysis, NPS benchmarking, conjoint, structured reference calls

### Software and Data
- SQL, Python / pandas, Alteryx against ERP and CRM extracts; Power BI or Tableau
- Alternative data: card panels (Earnest, Facteus), Similarweb, Sensor Tower, Revelio, Glassdoor
- Contract NLP: Kira, Luminance, DraftWise
- Tech diligence: SonarQube, cloud spend review
- Cyber diligence: BitSight, SecurityScorecard, credential exposure checks
- Virtual data rooms with analytics: Intralinks, Datasite, Ansarada

---

## 4. Post-Deal

### Techniques
- Event study — cumulative abnormal returns via market model or Fama-French
- Buy-and-hold abnormal returns; calendar-time portfolio regressions
- Difference-in-differences vs. a matched peer control group
- Re-underwriting: variance bridge from deal model to actuals
- ROIC bridge, EVA, CFROI on invested capital including integration spend
- Impairment testing under ASC 350 / IAS 36; headroom sensitivity as early warning
- Retained-cohort tracking and cross-sell attach rates

### Software
- Synergy trackers: Midaxo, DealRoom, Devensoft, Smartsheet
- Integration PMO dashboards; 100-day milestone tracking
- Workiva, Anaplan for purchase accounting and pro forma consolidation
- HRIS analytics for regretted attrition; Glint or Peakon for engagement
- Eventus or CRSP data, or Python (statsmodels) for the event study itself
