# M&A Synergy Estimator

Walk a hypothetical acquirer/target pairing through the full deal lifecycle — Price Paid, Deal Economics, Target Quality, Post-Deal — for any two public company tickers, validated against real historical deals.

## What this is

Given two tickers and a deal type (Acquisition or Merger), this tool pulls real financials (SEC filings + market data) and free-data implementations of techniques from `ma-analytical-toolkit.md`, organized into four tabs matching the deal lifecycle:

1. **Price Paid** — contribution analysis; for a Merger, implied ownership split (all-stock) with a contribution-vs-ownership fairness check; for an Acquisition, premium vs. unaffected price + VWAP; a simplified valuation-range "football field" (52-week range + analyst targets).
2. **Deal Economics** — synergy build-up waterfall, NPV, EPS accretion/dilution, tornado sensitivity, breakeven synergy analysis, ROIC vs. WACC, a leverage proxy, and a Monte Carlo NPV distribution.
3. **Target Quality** — Altman Z-score and working-capital cycle (DSO/DIO/DPO) diagnostics from SEC balance-sheet data.
4. **Post-Deal** — market-model event study (cumulative abnormal returns around an announcement date); for the 8 curated validation deals, a re-underwriting comparison of estimated vs. disclosed synergies.

Most of the toolkit's "Software and Data" columns are paid institutional platforms (Capital IQ, Bloomberg, FactSet, PitchBook, Anaplan, Kira, BitSight, etc.) this project doesn't have access to and won't fake. Where a technique is mathematically reproducible from free SEC EDGAR/Yahoo Finance data, it's implemented for real; where it fundamentally needs paid data, transaction-level GL access, or alternative data (contract text, customer cohorts, cyber scans), it's called out inline in the app and in each model module's docstring instead of being stubbed out.

The synergy assumptions are benchmarked against a curated set of real historical mergers with publicly disclosed synergy targets (see `synergy_estimator/validation/`), so the model's accuracy is measured, not assumed.

## Project structure

- `synergy_estimator/data/` — data ingestion (SEC EDGAR income statement + balance sheet, Yahoo Finance market data + price history)
- `synergy_estimator/models/` — `synergies.py`/`valuation.py`/`wacc.py`/`accretion_dilution.py`/`sensitivity.py` (Deal Economics), `price_paid.py` (Price Paid), `target_quality.py` (Target Quality), `post_deal.py` (Post-Deal event study)
- `synergy_estimator/validation/` — curated real-deal backtest (model estimate vs. disclosed target)
- `synergy_estimator/app/` — Streamlit interactive app (4-tab deal lifecycle view)
- `tests/` — unit tests

## Data sources

- **SEC EDGAR** (10-K, 10-Q, S-4 merger proxies) — free, public, used for income-statement and balance-sheet financials, and for sourcing real disclosed synergy targets in the validation set.
- **Yahoo Finance** — free market data (price, market cap, shares outstanding, beta, 52-week range, analyst targets, daily price history for VWAP/event studies).

This public repo and its demo run entirely on free, public data. Institutional data sources (Capital IQ / PitchBook / Bloomberg / WRDS), where available, are used offline only for model validation/tuning and are never pulled into this codebase or its public deployment.

## Status

Working end to end; the model is currently being recalibrated (see below). Data pipeline, core synergy model (cost + revenue synergies, WACC, NPV, EPS accretion/dilution), an 8-deal validation backtest, and an interactive Streamlit app spanning the full deal lifecycle (Price Paid / Deal Economics / Target Quality / Post-Deal) are all working.

**Backtest methodology correction.** The backtest previously scored every deal against each company's *latest* 10-K rather than the last one filed before the announcement. For older deals that compared a synergy target to financials from years of subsequent organic growth — ADI's SG&A was 94% higher in FY2025 than in the FY2019 report an analyst would have had when the Maxim deal was announced in July 2020. Since cost synergies are modeled as a linear function of SG&A and COGS, that roughly doubled the estimate for older deals. `get_financials_before()` now pins both sides of every deal to the last annual report public on the announcement date, and `run_backtest` skips (rather than mis-scores) deals with no pre-announcement filing.

Correcting it changed the picture materially. Measured on pre-announcement financials, the current defaults produce:

```
scored 8 deals | median abs error 75.1% | range -91.3% to +10.0%
median signed error -75.1% | underestimates 7/8
```

The previously reported range (roughly -89% to +48%) was measuring model error and data misalignment together, and the inflated inputs were partially masking a systematic *downward* bias. The real finding is that defaults calibrated on grocery retail (Kroger/Albertsons, where store-level labor doesn't disappear in a merger) badly underestimate overhead-heavy sectors — SNPS/ANSS (software) is -91% and ADI/MXIM (semiconductors) -88%, while RSG/ECOL (waste services) is +10%. That is an argument for per-sector calibration, which is in progress. Reproduce with `python -m synergy_estimator.validation.report`.

The app's sliders let a user correct for this by hand per-deal in the meantime.

The Post-Deal event study was sanity-checked against a real, well-known case: entering Kroger/Albertsons with the actual 2022-10-14 announcement date produces a +11.8% cumulative abnormal return for Albertsons (the target pricing in the takeover premium) and a roughly flat -0.2% CAR for Kroger (the acquirer) — the textbook signature of an M&A announcement.

Known limitations:
- Banks/financial institutions use a fundamentally different income statement (no COGS/SG&A) and aren't supported — out of scope, since they're modeled differently in real M&A work too.
- Airlines (cost-by-nature: fuel, labor, aircraft rent) and upstream/integrated oil & gas (production costs, DD&A, exploration expense instead of COGS) are excluded from the validation set for the same structural reason — confirmed by testing live EDGAR data for JetBlue/Spirit, ConocoPhillips/Marathon Oil, Chevron/Hess, and ExxonMobil/Pioneer Natural Resources, all of which lack a matched COGS concept.
- Cost of debt uses a flat credit-spread proxy rather than a real interest-coverage-based rating, since standalone interest expense isn't a reliable top-level XBRL tag across filers.
- Only 2 deals have been used to calibrate the current default assumptions — the 8-deal validation set measures generalization error but doesn't re-tune the defaults per industry. Recalibration against a larger corpus mined from EDGAR merger filings is in progress.
- The backtest target is *disclosed* synergies — what management promised at announcement, which is systematically optimistic. The model predicts the promise, not the realized outcome.
- The Post-Deal tab's re-underwriting comparison and event study need both tickers to still be live/public; all 8 curated validation deals have since-delisted targets (that's inherent to using *completed* deals with disclosed outcomes), so that comparison currently only lights up for live/blocked deal pairs, not the curated 8. Their backtested results are still available in the Deal Economics tab's Validation expander.
- Target Quality diagnostics (Altman Z-score, working capital cycle) and Price Paid's football field/VWAP degrade gracefully to "not enough data" when a filer doesn't tag a needed concept (e.g. no inventory line for a services company) rather than blocking the page, since they're supplementary, not core to the synergy estimate.
- Per each model module's docstring, several toolkit techniques are explicitly out of scope because they need data this free pipeline doesn't have: comparable-companies/precedent-transaction analysis and a standalone DCF valuation floor (needs a deals database + full company DCF model), purchase price allocation and real-options/decision-tree analysis (needs actual deal-structure terms), quality-of-earnings on transaction-level GL data, cohort/churn/survival analysis, concentration metrics, the Beneish M-score, and contract/cyber diligence (all need transaction-level or alternative data), and impairment testing, retained-cohort tracking, and difference-in-differences vs. a matched peer group (need disclosures this pipeline doesn't parse).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export EDGAR_IDENTITY="Your Name your.email@example.com"  # required by SEC's fair-access policy
```

## Running the app

```bash
streamlit run synergy_estimator/app/Home.py
```

Pick a deal type (Acquisition or Merger), enter an acquirer and target ticker, and adjust the assumption/premium sliders in the sidebar — all four tabs (Price Paid, Deal Economics, Target Quality, Post-Deal) update live. To run the Post-Deal event study, also enter an announcement date in the sidebar (auto-filled if your ticker pair matches one of the 8 curated validation deals, though see the known limitation above about delisted targets). The "Validation" expander inside the Deal Economics tab runs the 8-deal backtest against the current slider assumptions.
