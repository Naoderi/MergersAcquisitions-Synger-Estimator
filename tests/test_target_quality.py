import pytest

from synergy_estimator.data.schema import AnnualFinancials, BalanceSheetData, MarketData
from synergy_estimator.models.target_quality import altman_z_score, working_capital_diagnostics


def _financials(**overrides):
    defaults = dict(
        ticker="A",
        fiscal_year=2024,
        period_end="2024-12-31",
        revenue=1000.0,
        cogs=500.0,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=200.0,
        operating_income=150.0,
        net_income=100.0,
        shares_diluted=50.0,
        pretax_income=130.0,
        income_tax_expense=30.0,
    )
    defaults.update(overrides)
    return AnnualFinancials(**defaults)


def _balance_sheet(**overrides):
    defaults = dict(
        ticker="A",
        fiscal_year=2024,
        period_end="2024-12-31",
        total_assets=2000.0,
        total_liabilities=1000.0,
        total_current_assets=600.0,
        total_current_liabilities=300.0,
        retained_earnings=400.0,
        accounts_receivable=100.0,
        inventory=80.0,
        accounts_payable=60.0,
    )
    defaults.update(overrides)
    return BalanceSheetData(**defaults)


def _market(**overrides):
    defaults = dict(
        ticker="A",
        company_name="A Inc.",
        sector=None,
        industry=None,
        market_cap=3000.0,
        share_price=10.0,
        shares_outstanding=300.0,
        beta=1.0,
        total_debt=500.0,
    )
    defaults.update(overrides)
    return MarketData(**defaults)


def test_altman_z_score_matches_manual_formula():
    financials = _financials(revenue=1000.0, operating_income=150.0)
    bs = _balance_sheet(
        total_assets=2000.0, total_current_assets=600.0, total_current_liabilities=300.0,
        retained_earnings=400.0, total_liabilities=1000.0,
    )
    market = _market(market_cap=3000.0)

    result = altman_z_score(financials, bs, market)

    working_capital = 600.0 - 300.0
    expected = (
        1.2 * (working_capital / 2000.0)
        + 1.4 * (400.0 / 2000.0)
        + 3.3 * (150.0 / 2000.0)
        + 0.6 * (3000.0 / 1000.0)
        + 1.0 * (1000.0 / 2000.0)
    )
    assert result.z_score == pytest.approx(expected)


def test_altman_z_score_zones():
    financials = _financials(operating_income=150.0)
    market = _market(market_cap=3000.0)
    safe_bs = _balance_sheet(total_assets=1000.0, retained_earnings=800.0, total_liabilities=200.0)
    assert altman_z_score(financials, safe_bs, market).zone == "safe"

    distress_bs = _balance_sheet(
        total_assets=5000.0, total_current_assets=100.0, total_current_liabilities=4000.0,
        retained_earnings=-1000.0, total_liabilities=4800.0,
    )
    distress_market = _market(market_cap=100.0)
    assert altman_z_score(financials, distress_bs, distress_market).zone == "distress"


def test_altman_z_score_none_when_balance_sheet_data_missing():
    financials = _financials()
    bs = _balance_sheet(total_assets=None)
    market = _market()
    assert altman_z_score(financials, bs, market) is None


def test_working_capital_diagnostics_matches_manual_formula():
    financials = _financials(revenue=1000.0, cogs=500.0)
    bs = _balance_sheet(accounts_receivable=100.0, inventory=80.0, accounts_payable=60.0)

    result = working_capital_diagnostics(financials, bs)

    assert result.days_sales_outstanding == pytest.approx(100.0 / 1000.0 * 365)
    assert result.days_inventory_outstanding == pytest.approx(80.0 / 500.0 * 365)
    assert result.days_payable_outstanding == pytest.approx(60.0 / 500.0 * 365)
    assert result.cash_conversion_cycle == pytest.approx(
        result.days_sales_outstanding + result.days_inventory_outstanding - result.days_payable_outstanding
    )


def test_working_capital_diagnostics_degrades_field_by_field_when_inventory_missing():
    financials = _financials(revenue=1000.0, cogs=500.0)
    bs = _balance_sheet(accounts_receivable=100.0, inventory=None, accounts_payable=60.0)

    result = working_capital_diagnostics(financials, bs)

    assert result.days_sales_outstanding is not None
    assert result.days_inventory_outstanding is None
    assert result.days_payable_outstanding is not None
    assert result.cash_conversion_cycle is None
