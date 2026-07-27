import pytest

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import SynergyAssumptions, SynergyEstimate
from synergy_estimator.models.valuation import (
    breakeven_synergies,
    leverage_ratio,
    monte_carlo_npv,
    npv_of_synergies,
    payback_period_years,
    phased_synergy_cashflows,
    roic,
    synergies_pct_of_deal_value,
)


def test_phased_synergy_cashflows_holds_final_pct_after_schedule_ends():
    cashflows = phased_synergy_cashflows(100.0, (0.5, 1.0), forecast_years=4)
    assert cashflows == [50.0, 100.0, 100.0, 100.0]


def _synergy_estimate(total_runrate=100.0, integration_cost=125.0, phase_in=(1.0,)):
    return SynergyEstimate(
        acquirer_ticker="A",
        target_ticker="B",
        cost_synergies_runrate=total_runrate,
        revenue_synergies_runrate=0.0,
        total_runrate_synergies=total_runrate,
        integration_cost=integration_cost,
        assumptions=SynergyAssumptions(phase_in_schedule=phase_in),
    )


def test_npv_of_synergies_matches_manual_discounting():
    estimate = _synergy_estimate(total_runrate=100.0, integration_cost=50.0, phase_in=(1.0,))
    wacc = 0.10
    forecast_years = 3
    expected_pv = sum(100.0 / (1.10**year) for year in range(1, forecast_years + 1)) - 50.0
    assert npv_of_synergies(estimate, wacc, forecast_years=forecast_years) == pytest.approx(expected_pv)


def test_synergies_pct_of_deal_value():
    assert synergies_pct_of_deal_value(250.0, 1000.0) == pytest.approx(0.25)
    assert synergies_pct_of_deal_value(250.0, 0.0) is None


def test_payback_period_years():
    estimate = _synergy_estimate(total_runrate=100.0, integration_cost=125.0)
    assert payback_period_years(estimate) == pytest.approx(1.25)


def test_breakeven_synergies_matches_manual_solve():
    assumptions = SynergyAssumptions(phase_in_schedule=(1.0,), integration_cost_multiple=0.5)
    wacc = 0.10
    forecast_years = 3
    annuity_factor = sum(1.0 / (1.10**year) for year in range(1, forecast_years + 1))
    net_factor = annuity_factor - 0.5
    premium_paid = 500.0
    assert breakeven_synergies(premium_paid, assumptions, wacc, forecast_years) == pytest.approx(
        premium_paid / net_factor
    )


def test_breakeven_synergies_raises_when_integration_cost_dominates():
    assumptions = SynergyAssumptions(phase_in_schedule=(1.0,), integration_cost_multiple=10.0)
    with pytest.raises(ValueError):
        breakeven_synergies(500.0, assumptions, wacc=0.10, forecast_years=1)


def test_roic_divides_nopat_by_invested_capital():
    assert roic(nopat=50.0, invested_capital=500.0) == pytest.approx(0.10)


def test_roic_none_when_no_invested_capital():
    assert roic(nopat=50.0, invested_capital=0.0) is None


def test_leverage_ratio_divides_debt_by_ebitda_proxy():
    assert leverage_ratio(total_debt=300.0, ebitda_proxy=100.0) == pytest.approx(3.0)


def test_leverage_ratio_none_without_ebitda_proxy():
    assert leverage_ratio(total_debt=300.0, ebitda_proxy=0.0) is None


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


def test_monte_carlo_npv_produces_sorted_percentiles_within_range():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    result = monte_carlo_npv(
        "A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10, n_trials=200, seed=42
    )
    assert len(result.npvs) == 200
    assert result.p10 <= result.p50 <= result.p90
    assert 0.0 <= result.pct_positive <= 1.0


def test_monte_carlo_npv_is_reproducible_with_a_seed():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    result_a = monte_carlo_npv("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10, n_trials=50, seed=7)
    result_b = monte_carlo_npv("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10, n_trials=50, seed=7)
    assert result_a.npvs == result_b.npvs
