"""Cost + revenue synergy estimation.

Default overlap rates were calibrated against two real deals with both
companies still independently public today (so live financials are pullable):
Kroger/Albertsons (~$1B disclosed run-rate cost synergies net of divestitures,
announced Oct 2022) and Tapestry/Capri (~$200M disclosed run-rate cost
synergies, announced Aug 2023). An initial 20%/3% guess overestimated
Kroger/Albertsons by ~15x -- grocery retail SG&A is dominated by store-level
labor that doesn't disappear in a merger, unlike corporate-overhead-heavy
industries. The calibrated rates (1.5% SG&A / 0.15% COGS) land within 3% of
the Kroger/Albertsons target and within a conservative ~2x of Tapestry/Capri.
Two deals across two very different industries isn't enough to trust broadly
-- Phase 3 (the full 8-15 deal validation set) is what tunes these per-industry;
treat these as a much-better-grounded starting point, not a finished calibration.
"""

from dataclasses import dataclass

from synergy_estimator.data.schema import AnnualFinancials


@dataclass
class SynergyAssumptions:
    sga_overlap_rate: float = 0.015
    cogs_overlap_rate: float = 0.0015
    duplicate_public_company_cost: float = 15_000_000.0
    revenue_cross_sell_rate: float = 0.01  # applied to the smaller company's revenue
    revenue_confidence_weight: float = 0.5  # probability haircut vs. cost synergies
    integration_cost_multiple: float = 1.25  # x run-rate synergies, paid upfront
    phase_in_schedule: tuple[float, ...] = (0.30, 0.65, 1.0)  # % of run-rate by end of year 1, 2, 3


@dataclass
class SynergyEstimate:
    acquirer_ticker: str
    target_ticker: str
    cost_synergies_runrate: float
    revenue_synergies_runrate: float  # already probability-weighted
    total_runrate_synergies: float
    integration_cost: float
    assumptions: SynergyAssumptions


def _require(financials: AnnualFinancials, field: str) -> float:
    value = getattr(financials, field)
    if value is None:
        raise ValueError(
            f"{financials.ticker} FY{financials.fiscal_year}: missing '{field}' -- cannot estimate "
            "synergies without it. This usually means the filer tags this line item with a custom "
            "XBRL concept not yet handled in edgar_source.py."
        )
    return value


def estimate_cost_synergies(
    acquirer: AnnualFinancials, target: AnnualFinancials, assumptions: SynergyAssumptions
) -> float:
    combined_sga = _require(acquirer, "sga_expense") + _require(target, "sga_expense")
    combined_cogs = _require(acquirer, "cogs") + _require(target, "cogs")
    return (
        combined_sga * assumptions.sga_overlap_rate
        + combined_cogs * assumptions.cogs_overlap_rate
        + assumptions.duplicate_public_company_cost
    )


def estimate_revenue_synergies(
    acquirer: AnnualFinancials, target: AnnualFinancials, assumptions: SynergyAssumptions
) -> float:
    smaller_revenue = min(_require(acquirer, "revenue"), _require(target, "revenue"))
    raw_uplift = smaller_revenue * assumptions.revenue_cross_sell_rate
    return raw_uplift * assumptions.revenue_confidence_weight


def estimate_synergies(
    acquirer_ticker: str,
    acquirer_financials: AnnualFinancials,
    target_ticker: str,
    target_financials: AnnualFinancials,
    assumptions: SynergyAssumptions | None = None,
) -> SynergyEstimate:
    assumptions = assumptions or SynergyAssumptions()
    cost = estimate_cost_synergies(acquirer_financials, target_financials, assumptions)
    revenue = estimate_revenue_synergies(acquirer_financials, target_financials, assumptions)
    total = cost + revenue
    return SynergyEstimate(
        acquirer_ticker=acquirer_ticker,
        target_ticker=target_ticker,
        cost_synergies_runrate=cost,
        revenue_synergies_runrate=revenue,
        total_runrate_synergies=total,
        integration_cost=total * assumptions.integration_cost_multiple,
        assumptions=assumptions,
    )
