import pytest

from synergy_estimator.validation.backtest import BacktestResult
from synergy_estimator.validation.deals import HistoricalDeal
from synergy_estimator.validation.report import summarize


def _deal(acquirer="A", target="B"):
    return HistoricalDeal(
        acquirer_ticker=acquirer,
        target_ticker=target,
        target_cik=None,
        disclosed_synergy_runrate=100.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2024-01-01",
        source="test fixture",
    )


def _result(pct_error):
    if pct_error is None:
        return BacktestResult(
            deal=_deal(),
            estimated_synergies=None,
            disclosed_synergies=100.0,
            pct_error=None,
            skipped_reason="no pre-announcement 10-K",
        )
    return BacktestResult(
        deal=_deal(),
        estimated_synergies=100.0 * (1 + pct_error),
        disclosed_synergies=100.0,
        pct_error=pct_error,
    )


def test_summarize_reports_signed_bias_separately_from_absolute_error():
    """Absolute error alone hides direction. A model that is uniformly 50% low
    is a calibration problem; one that is 50% off in both directions is noise."""
    summary = summarize([_result(-0.5), _result(-0.5), _result(-0.5)])

    assert summary.median_abs_error == pytest.approx(0.5)
    assert summary.median_signed_error == pytest.approx(-0.5)
    assert summary.underestimates == 3


def test_summarize_excludes_skipped_deals_from_error_statistics():
    summary = summarize([_result(-0.5), _result(None), _result(0.5)])

    assert summary.scored == 2
    assert summary.skipped == 1
    assert summary.median_abs_error == pytest.approx(0.5)
    assert summary.underestimates == 1


def test_summarize_raises_when_nothing_was_scored():
    with pytest.raises(ValueError, match="no deals were scored"):
        summarize([_result(None)])
