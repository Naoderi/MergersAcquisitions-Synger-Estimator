"""NPV of synergies, phase-in cash flows, and synergies-as-%-of-deal-value.

Also covers the free-data-feasible "Deal Economics" techniques from the M&A
analytical toolkit: breakeven synergy analysis, ROIC vs. WACC, a leverage
proxy, and a lightweight Monte Carlo over the synergy assumptions. Purchase
price allocation / step-up modelling, real options / decision-tree analysis,
and debt-covenant-headroom modelling need deal-structure inputs (allocated
goodwill, actual covenant terms, earnout triggers) this pipeline doesn't
have -- out of scope, not faked.
"""

import dataclasses
import random

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import SynergyAssumptions, SynergyEstimate, estimate_synergies


def phased_synergy_cashflows(
    total_runrate_synergies: float, phase_in_schedule: tuple[float, ...], forecast_years: int
) -> list[float]:
    """Year-by-year synergy cash flow. Once the phase-in schedule ends, synergies
    are held at their final (typically 100%) level for the rest of the forecast."""
    cashflows = []
    for year in range(1, forecast_years + 1):
        pct = phase_in_schedule[year - 1] if year <= len(phase_in_schedule) else phase_in_schedule[-1]
        cashflows.append(total_runrate_synergies * pct)
    return cashflows


def npv_of_synergies(synergy_estimate: SynergyEstimate, wacc: float, forecast_years: int = 5) -> float:
    """NPV of the synergy cash flows, net of the upfront (year-0) integration cost.

    Default 5-year explicit horizon (no terminal/perpetuity value, no decay) is
    a standard, conservative banker convention -- a longer horizon or an
    explicit terminal value would inflate NPV further and requires additional
    assumptions (e.g. a synergy decay/fade rate) not modeled here.
    """
    cashflows = phased_synergy_cashflows(
        synergy_estimate.total_runrate_synergies, synergy_estimate.assumptions.phase_in_schedule, forecast_years
    )
    discounted = sum(cf / (1 + wacc) ** year for year, cf in enumerate(cashflows, start=1))
    return discounted - synergy_estimate.integration_cost


def synergies_pct_of_deal_value(synergy_npv: float, target_market_cap: float) -> float | None:
    """How much of the target's standalone value the synergy NPV represents --
    a common framing for whether a deal premium is justified by synergies."""
    if not target_market_cap:
        return None
    return synergy_npv / target_market_cap


def payback_period_years(synergy_estimate: SynergyEstimate) -> float | None:
    """Years of run-rate synergies needed to recoup the upfront integration cost."""
    if not synergy_estimate.total_runrate_synergies:
        return None
    return synergy_estimate.integration_cost / synergy_estimate.total_runrate_synergies


def breakeven_synergies(
    premium_paid: float, assumptions: SynergyAssumptions, wacc: float, forecast_years: int = 5
) -> float:
    """Pretax run-rate synergies (at the given phase-in schedule and
    integration-cost multiple) required for the synergy NPV to fully offset a
    premium paid over the target's unaffected value -- "how much synergy do we
    need to realize to justify this price."
    """
    unit_cashflows = phased_synergy_cashflows(1.0, assumptions.phase_in_schedule, forecast_years)
    annuity_factor = sum(cf / (1 + wacc) ** year for year, cf in enumerate(unit_cashflows, start=1))
    net_factor = annuity_factor - assumptions.integration_cost_multiple
    if net_factor <= 0:
        raise ValueError(
            "Integration cost multiple exceeds the discounted synergy annuity factor -- no level of "
            "run-rate synergies breaks even under these assumptions."
        )
    return premium_paid / net_factor


def roic(nopat: float, invested_capital: float) -> float | None:
    """Return on invested capital = NOPAT / invested capital. `invested_capital`
    is expected as a market-value figure (market cap + total debt), consistent
    with how this project already treats capital structure in wacc.py, rather
    than book value."""
    if not invested_capital:
        return None
    return nopat / invested_capital


def leverage_ratio(total_debt: float, ebitda_proxy: float) -> float | None:
    """Debt / EBITDA-proxy leverage multiple. Uses operating income as an
    EBITDA proxy (the schema has no separate D&A line, the same simplification
    documented in wacc.py for cost of debt) -- a debt-capacity sanity check,
    not a real covenant-headroom model."""
    if not ebitda_proxy:
        return None
    return total_debt / ebitda_proxy


@dataclasses.dataclass
class MonteCarloResult:
    npvs: list[float]
    mean: float
    p10: float
    p50: float
    p90: float
    pct_positive: float  # fraction of trials with a positive NPV


def monte_carlo_npv(
    acquirer_ticker: str,
    acquirer_financials: AnnualFinancials,
    target_ticker: str,
    target_financials: AnnualFinancials,
    base_assumptions: SynergyAssumptions,
    wacc: float,
    forecast_years: int = 5,
    n_trials: int = 500,
    spread_pct: float = 0.30,
    seed: int | None = None,
) -> MonteCarloResult:
    """Randomizes sga_overlap_rate, cogs_overlap_rate, and
    revenue_cross_sell_rate each trial (independent triangular draws centered
    on the base assumption, +/-spread_pct), recomputes NPV, and summarizes the
    resulting distribution -- a lightweight stand-in for the toolkit's "Monte
    Carlo on price, synergies and phasing," using only this model's existing
    assumptions rather than a full deal-structure simulation.
    """
    rng = random.Random(seed)

    def _jitter(base_value: float) -> float:
        low = base_value * (1 - spread_pct)
        high = base_value * (1 + spread_pct)
        return rng.triangular(low, high, base_value)

    npvs = []
    for _ in range(n_trials):
        trial_assumptions = dataclasses.replace(
            base_assumptions,
            sga_overlap_rate=_jitter(base_assumptions.sga_overlap_rate),
            cogs_overlap_rate=_jitter(base_assumptions.cogs_overlap_rate),
            revenue_cross_sell_rate=_jitter(base_assumptions.revenue_cross_sell_rate),
        )
        estimate = estimate_synergies(
            acquirer_ticker, acquirer_financials, target_ticker, target_financials, trial_assumptions
        )
        npvs.append(npv_of_synergies(estimate, wacc, forecast_years=forecast_years))

    npvs.sort()
    n = len(npvs)
    return MonteCarloResult(
        npvs=npvs,
        mean=sum(npvs) / n,
        p10=npvs[int(0.10 * (n - 1))],
        p50=npvs[int(0.50 * (n - 1))],
        p90=npvs[int(0.90 * (n - 1))],
        pct_positive=sum(1 for v in npvs if v > 0) / n,
    )
