import pytest

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import SynergyAssumptions, SynergyEstimate
from synergy_estimator.validation.backtest import run_backtest, score_deal
from synergy_estimator.validation.deals import HistoricalDeal


def _estimate(cost=80.0, revenue=20.0):
    return SynergyEstimate(
        acquirer_ticker="A",
        target_ticker="B",
        cost_synergies_runrate=cost,
        revenue_synergies_runrate=revenue,
        total_runrate_synergies=cost + revenue,
        integration_cost=(cost + revenue) * 1.25,
        assumptions=SynergyAssumptions(),
    )


def _deal(synergy_type="cost", disclosed=100.0):
    return HistoricalDeal(
        acquirer_ticker="A",
        target_ticker="B",
        target_cik=None,
        disclosed_synergy_runrate=disclosed,
        synergy_type=synergy_type,
        deal_status="completed",
        announcement_date="2024-01-01",
        source="test fixture",
    )


def test_score_deal_uses_cost_only_estimate_for_cost_type_deals():
    result = score_deal(_estimate(cost=80.0, revenue=20.0), _deal(synergy_type="cost", disclosed=100.0))
    assert result.estimated_synergies == pytest.approx(80.0)
    assert result.disclosed_synergies == pytest.approx(100.0)
    assert result.pct_error == pytest.approx((80.0 - 100.0) / 100.0)


def test_score_deal_uses_total_estimate_for_cost_and_revenue_type_deals():
    result = score_deal(
        _estimate(cost=80.0, revenue=20.0), _deal(synergy_type="cost_and_revenue", disclosed=100.0)
    )
    assert result.estimated_synergies == pytest.approx(100.0)
    assert result.pct_error == pytest.approx(0.0)


def test_score_deal_pct_error_is_positive_when_overestimating():
    result = score_deal(_estimate(cost=150.0, revenue=0.0), _deal(synergy_type="cost", disclosed=100.0))
    assert result.pct_error == pytest.approx(0.5)


def _financials(ticker, period_end, sga=1_000.0, cogs=10_000.0, revenue=20_000.0):
    return AnnualFinancials(
        ticker=ticker,
        fiscal_year=int(period_end[:4]),
        period_end=period_end,
        revenue=revenue,
        cogs=cogs,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=sga,
        operating_income=None,
        net_income=None,
        shares_diluted=None,
        pretax_income=None,
        income_tax_expense=None,
    )


def test_run_backtest_requests_financials_as_of_each_announcement_date():
    """The whole point of the date-aware refactor: both sides of the deal must
    be fetched as of the announcement, not as of today."""
    calls = []

    def fake_fetch(identifier, display_ticker, announcement_date):
        calls.append((identifier, announcement_date))
        return _financials(display_ticker, "2023-12-31")

    deal = _deal()
    run_backtest([deal], fetch=fake_fetch)

    assert calls == [("A", "2024-01-01"), ("B", "2024-01-01")]


def test_run_backtest_prefers_target_cik_over_ticker_for_delisted_targets():
    calls = []

    def fake_fetch(identifier, display_ticker, announcement_date):
        calls.append(identifier)
        return _financials(display_ticker, "2023-12-31")

    deal = _deal()
    deal.target_cik = 743316
    run_backtest([deal], fetch=fake_fetch)

    assert calls == ["A", 743316]


def test_run_backtest_skips_deals_without_pre_announcement_financials():
    """A target that IPO'd months before announcement has no pre-deal 10-K.
    Dropping it is honest; scoring it on post-deal financials is the bug."""

    def fake_fetch(identifier, display_ticker, announcement_date):
        if display_ticker == "B":
            raise ValueError("no 10-K filed before 2024-01-01")
        return _financials(display_ticker, "2023-12-31")

    (result,) = run_backtest([_deal()], fetch=fake_fetch)

    assert not result.scored
    assert result.pct_error is None
    assert result.estimated_synergies is None
    assert "no 10-K filed before" in result.skipped_reason


def test_run_backtest_scores_deals_that_have_pre_announcement_financials():
    def fake_fetch(identifier, display_ticker, announcement_date):
        return _financials(display_ticker, "2023-12-31")

    (result,) = run_backtest([_deal(disclosed=100.0)], fetch=fake_fetch)

    assert result.scored
    assert result.skipped_reason is None
    assert result.estimated_synergies > 0
