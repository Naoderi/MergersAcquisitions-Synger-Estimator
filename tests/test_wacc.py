import pytest

from synergy_estimator.data.schema import AnnualFinancials, MarketData
from synergy_estimator.models.wacc import (
    blended_wacc,
    cost_of_debt,
    cost_of_equity,
    effective_tax_rate,
    estimate_wacc,
)


def _market(beta=1.0, market_cap=800.0, total_debt=200.0):
    return MarketData(
        ticker="A",
        company_name="A Inc.",
        sector=None,
        industry=None,
        market_cap=market_cap,
        share_price=10.0,
        shares_outstanding=80.0,
        beta=beta,
        total_debt=total_debt,
    )


def _financials(pretax_income=130.0, income_tax_expense=30.0):
    return AnnualFinancials(
        ticker="A",
        fiscal_year=2024,
        period_end="2024-12-31",
        revenue=1000.0,
        cogs=500.0,
        gross_profit=500.0,
        rnd_expense=None,
        sga_expense=200.0,
        operating_income=300.0,
        net_income=100.0,
        shares_diluted=80.0,
        pretax_income=pretax_income,
        income_tax_expense=income_tax_expense,
    )


def test_effective_tax_rate_divides_tax_expense_by_pretax_income():
    assert effective_tax_rate(_financials(130.0, 30.0)) == pytest.approx(30.0 / 130.0)


def test_effective_tax_rate_none_when_pretax_loss():
    assert effective_tax_rate(_financials(-50.0, 0.0)) is None


def test_estimate_wacc_weights_by_market_capital_structure():
    market = _market(beta=1.2, market_cap=800.0, total_debt=200.0)
    financials = _financials(130.0, 30.0)
    rf = 0.04
    wacc = estimate_wacc(market, financials, rf)

    tax_rate = 30.0 / 130.0
    expected = 0.8 * cost_of_equity(rf, 1.2) + 0.2 * cost_of_debt(rf) * (1 - tax_rate)
    assert wacc == pytest.approx(expected)


def test_estimate_wacc_raises_without_beta():
    market = _market(beta=None)
    with pytest.raises(ValueError, match="beta"):
        estimate_wacc(market, _financials(), 0.04)


def test_blended_wacc_weights_by_total_capital():
    result = blended_wacc(acquirer_wacc=0.10, acquirer_capital=300.0, target_wacc=0.06, target_capital=100.0)
    assert result == pytest.approx((0.10 * 300 + 0.06 * 100) / 400)
