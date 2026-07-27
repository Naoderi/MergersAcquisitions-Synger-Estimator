import pytest

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.accretion_dilution import after_tax_synergies, eps_accretion_dilution


def _financials(ticker, net_income, shares_diluted, pretax_income=None, income_tax_expense=None):
    return AnnualFinancials(
        ticker=ticker,
        fiscal_year=2024,
        period_end="2024-12-31",
        revenue=1000.0,
        cogs=500.0,
        gross_profit=500.0,
        rnd_expense=None,
        sga_expense=200.0,
        operating_income=300.0,
        net_income=net_income,
        shares_diluted=shares_diluted,
        pretax_income=pretax_income,
        income_tax_expense=income_tax_expense,
    )


def test_after_tax_synergies_blends_both_companies_effective_tax_rates():
    acquirer = _financials("A", 100.0, 50.0, pretax_income=130.0, income_tax_expense=30.0)  # ~23.1%
    target = _financials("B", 50.0, 20.0, pretax_income=70.0, income_tax_expense=10.0)  # ~14.3%
    blended_rate = ((30.0 / 130.0) + (10.0 / 70.0)) / 2
    assert after_tax_synergies(100.0, acquirer, target) == pytest.approx(100.0 * (1 - blended_rate))


def test_after_tax_synergies_falls_back_to_statutory_rate_when_tax_data_missing():
    acquirer = _financials("A", 100.0, 50.0)  # no pretax_income/income_tax_expense
    target = _financials("B", 50.0, 20.0)
    assert after_tax_synergies(100.0, acquirer, target) == pytest.approx(100.0 * (1 - 0.21))


def test_eps_accretion_dilution_all_cash_no_new_shares():
    acquirer = _financials("A", net_income=100.0, shares_diluted=50.0, pretax_income=130.0, income_tax_expense=30.0)
    target = _financials("B", net_income=50.0, shares_diluted=20.0, pretax_income=70.0, income_tax_expense=10.0)

    result = eps_accretion_dilution(acquirer, target, pretax_runrate_synergies=20.0)

    assert result.acquirer_standalone_eps == pytest.approx(2.0)
    assert result.proforma_eps_without_synergies == pytest.approx(150.0 / 50.0)
    assert result.pct_change_without_synergies == pytest.approx((3.0 - 2.0) / 2.0)
    # with synergies: combined net income + after-tax synergy benefit, same combined shares
    blended_rate = ((30.0 / 130.0) + (10.0 / 70.0)) / 2
    expected_with = (150.0 + 20.0 * (1 - blended_rate)) / 50.0
    assert result.proforma_eps_with_synergies == pytest.approx(expected_with)
