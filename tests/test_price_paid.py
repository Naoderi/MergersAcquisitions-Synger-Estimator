import pytest

from synergy_estimator.data.schema import AnnualFinancials, MarketData, PricePoint
from synergy_estimator.models.price_paid import (
    acquisition_premium,
    contribution_analysis,
    implied_offer_price,
    implied_ownership_split,
    valuation_range,
    vwap,
)


def _financials(ticker, revenue=1000.0, operating_income=200.0, net_income=100.0, **overrides):
    defaults = dict(
        ticker=ticker,
        fiscal_year=2024,
        period_end="2024-12-31",
        revenue=revenue,
        cogs=500.0,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=200.0,
        operating_income=operating_income,
        net_income=net_income,
        shares_diluted=50.0,
        pretax_income=130.0,
        income_tax_expense=30.0,
    )
    defaults.update(overrides)
    return AnnualFinancials(**defaults)


def _market(**overrides):
    defaults = dict(
        ticker="A",
        company_name="A Inc.",
        sector=None,
        industry=None,
        market_cap=800.0,
        share_price=10.0,
        shares_outstanding=80.0,
        beta=1.0,
        total_debt=200.0,
    )
    defaults.update(overrides)
    return MarketData(**defaults)


def test_contribution_analysis_computes_pct_of_combined_for_each_metric():
    acquirer = _financials("A", revenue=800.0, operating_income=150.0, net_income=80.0)
    target = _financials("B", revenue=200.0, operating_income=50.0, net_income=20.0)
    rows = contribution_analysis(acquirer, target)
    revenue_row = next(r for r in rows if r.metric == "Revenue")
    assert revenue_row.acquirer_pct == pytest.approx(0.8)
    assert revenue_row.target_pct == pytest.approx(0.2)


def test_contribution_analysis_skips_metrics_with_missing_data():
    acquirer = _financials("A", operating_income=None)
    target = _financials("B")
    rows = contribution_analysis(acquirer, target)
    assert "Operating income" not in {r.metric for r in rows}
    assert "Revenue" in {r.metric for r in rows}


def test_implied_ownership_split_no_premium():
    result = implied_ownership_split(acquirer_market_cap=800.0, target_market_cap=200.0)
    assert result.target_ownership_pct == pytest.approx(0.2)
    assert result.acquirer_ownership_pct == pytest.approx(0.8)
    assert result.implied_target_deal_value == pytest.approx(200.0)


def test_implied_ownership_split_with_premium_increases_target_pct():
    no_premium = implied_ownership_split(acquirer_market_cap=800.0, target_market_cap=200.0, premium_pct=0.0)
    with_premium = implied_ownership_split(acquirer_market_cap=800.0, target_market_cap=200.0, premium_pct=0.30)
    assert with_premium.target_ownership_pct > no_premium.target_ownership_pct


def test_implied_ownership_split_raises_without_market_caps():
    with pytest.raises(ValueError):
        implied_ownership_split(acquirer_market_cap=0.0, target_market_cap=200.0)


def test_acquisition_premium_positive_when_offer_above_unaffected():
    assert acquisition_premium(offer_price_per_share=55.0, unaffected_price_per_share=50.0) == pytest.approx(0.10)


def test_implied_offer_price_is_inverse_of_acquisition_premium():
    unaffected = 50.0
    premium = 0.25
    offer = implied_offer_price(unaffected, premium)
    assert acquisition_premium(offer, unaffected) == pytest.approx(premium)


def test_vwap_weights_by_volume():
    points = [
        PricePoint(date="2024-01-01", close=10.0, volume=100.0),
        PricePoint(date="2024-01-02", close=20.0, volume=300.0),
    ]
    assert vwap(points) == pytest.approx((10.0 * 100.0 + 20.0 * 300.0) / 400.0)


def test_vwap_returns_none_without_volume_data():
    points = [PricePoint(date="2024-01-01", close=10.0, volume=None)]
    assert vwap(points) is None


def test_valuation_range_includes_only_available_bars():
    market = _market(fifty_two_week_low=8.0, fifty_two_week_high=12.0)
    bars = valuation_range(market)
    assert len(bars) == 1
    assert bars[0].label == "52-week range"
    assert bars[0].low == pytest.approx(8.0)
    assert bars[0].high == pytest.approx(12.0)


def test_valuation_range_empty_when_no_data_available():
    market = _market()
    assert valuation_range(market) == []
