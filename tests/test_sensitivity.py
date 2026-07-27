import pytest

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.sensitivity import tornado_sensitivity
from synergy_estimator.models.synergies import SynergyAssumptions


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


def test_tornado_sensitivity_returns_a_row_per_driver_plus_wacc():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    rows = tornado_sensitivity("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10)
    assert {row.label for row in rows} == {
        "sga_overlap_rate",
        "cogs_overlap_rate",
        "revenue_cross_sell_rate",
        "integration_cost_multiple",
        "wacc",
    }


def test_tornado_sensitivity_sorted_by_descending_swing():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    rows = tornado_sensitivity("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10)
    swings = [abs(row.high_npv - row.low_npv) for row in rows]
    assert swings == sorted(swings, reverse=True)


def test_tornado_sensitivity_lower_wacc_gives_higher_npv():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    rows = tornado_sensitivity("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10)
    wacc_row = next(row for row in rows if row.label == "wacc")
    assert wacc_row.low_value < wacc_row.high_value
    assert wacc_row.low_npv > wacc_row.high_npv


def test_tornado_sensitivity_higher_overlap_rate_gives_higher_npv():
    acquirer = _financials("A", revenue=1000.0, cogs=500.0, sga=200.0)
    target = _financials("B", revenue=400.0, cogs=200.0, sga=100.0)
    rows = tornado_sensitivity("A", acquirer, "B", target, SynergyAssumptions(), wacc=0.10)
    sga_row = next(row for row in rows if row.label == "sga_overlap_rate")
    assert sga_row.low_npv < sga_row.high_npv
