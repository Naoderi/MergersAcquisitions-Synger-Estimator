import pytest

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import (
    SynergyAssumptions,
    estimate_cost_synergies,
    estimate_revenue_synergies,
    estimate_synergies,
)


def _financials(ticker, revenue=1000.0, cogs=500.0, sga=200.0, **overrides):
    defaults = dict(
        ticker=ticker,
        fiscal_year=2024,
        period_end="2024-12-31",
        revenue=revenue,
        cogs=cogs,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=sga,
        operating_income=None,
        net_income=100.0,
        shares_diluted=50.0,
        pretax_income=130.0,
        income_tax_expense=30.0,
    )
    defaults.update(overrides)
    return AnnualFinancials(**defaults)


def test_estimate_cost_synergies_uses_combined_sga_and_cogs():
    acquirer = _financials("A", cogs=500.0, sga=200.0)
    target = _financials("B", cogs=300.0, sga=100.0)
    assumptions = SynergyAssumptions(
        sga_overlap_rate=0.10, cogs_overlap_rate=0.02, duplicate_public_company_cost=15.0
    )
    result = estimate_cost_synergies(acquirer, target, assumptions)
    assert result == pytest.approx((200 + 100) * 0.10 + (500 + 300) * 0.02 + 15.0)


def test_estimate_cost_synergies_raises_on_missing_sga():
    acquirer = _financials("A", sga=None)
    target = _financials("B")
    with pytest.raises(ValueError, match="sga_expense"):
        estimate_cost_synergies(acquirer, target, SynergyAssumptions())


def test_estimate_revenue_synergies_uses_smaller_companys_revenue():
    acquirer = _financials("A", revenue=1000.0)
    target = _financials("B", revenue=400.0)
    assumptions = SynergyAssumptions(revenue_cross_sell_rate=0.05, revenue_confidence_weight=0.5)
    result = estimate_revenue_synergies(acquirer, target, assumptions)
    assert result == pytest.approx(400.0 * 0.05 * 0.5)


def test_estimate_synergies_integration_cost_scales_with_multiple():
    acquirer = _financials("A")
    target = _financials("B")
    assumptions = SynergyAssumptions(integration_cost_multiple=1.5)
    estimate = estimate_synergies("A", acquirer, "B", target, assumptions)
    assert estimate.integration_cost == pytest.approx(estimate.total_runrate_synergies * 1.5)
