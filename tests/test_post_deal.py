import pytest

from synergy_estimator.data.schema import PricePoint
from synergy_estimator.models.post_deal import event_study_car, event_study_from_price_history


def test_event_study_car_zero_when_stock_tracks_market_exactly():
    market_returns = [0.01, -0.005, 0.02, 0.0, -0.01] * 10
    stock_returns = list(market_returns)  # beta=1, alpha=0, no abnormal return
    result = event_study_car(market_returns, market_returns, stock_returns[:5], market_returns[:5])
    assert result.beta == pytest.approx(1.0)
    assert result.alpha == pytest.approx(0.0, abs=1e-9)
    assert result.cumulative_abnormal_return == pytest.approx(0.0, abs=1e-9)


def test_event_study_car_positive_when_stock_jumps_beyond_market_model():
    estimation_market = [0.01, -0.005, 0.02, 0.0, -0.01] * 10
    estimation_stock = list(estimation_market)  # beta=1 during estimation
    event_market = [0.005, 0.005]
    event_stock = [0.05, 0.03]  # well above what beta=1 predicts
    result = event_study_car(estimation_stock, estimation_market, event_stock, event_market)
    assert result.cumulative_abnormal_return > 0.05


def test_event_study_car_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        event_study_car([0.01, 0.02], [0.01], [0.01], [0.01])


def test_event_study_car_raises_on_zero_variance_market():
    flat_market = [0.0] * 10
    with pytest.raises(ValueError):
        event_study_car([0.01] * 10, flat_market, [0.01], [0.0])


def _price_points(closes: list[float], start_index: int = 0) -> list[PricePoint]:
    return [
        PricePoint(date=f"2024-{1 + (start_index + i) // 28:02d}-{1 + (start_index + i) % 28:02d}", close=c, volume=1000.0)
        for i, c in enumerate(closes)
    ]


def test_event_study_from_price_history_end_to_end():
    # 50 days of matched, correlated random-walk-ish prices (beta ~1) for stock and market
    market_closes = [100.0]
    stock_closes = [50.0]
    for i in range(59):
        market_move = 0.001 * ((i % 7) - 3)
        market_closes.append(market_closes[-1] * (1 + market_move))
        stock_closes.append(stock_closes[-1] * (1 + market_move))

    market_points = _price_points(market_closes)
    stock_points = _price_points(stock_closes)
    event_date = stock_points[45].date

    result = event_study_from_price_history(stock_points, market_points, event_date, event_window_days=2)
    assert result.beta == pytest.approx(1.0, abs=0.05)
    assert len(result.abnormal_returns) == 5  # 2 before + event day + 2 after


def test_event_study_from_price_history_raises_when_event_date_missing():
    points = _price_points([100.0] * 40)
    with pytest.raises(ValueError, match="not found"):
        event_study_from_price_history(points, points, "1999-01-01")


def test_event_study_from_price_history_raises_with_too_little_estimation_data():
    points = _price_points([100.0 + i for i in range(10)])
    with pytest.raises(ValueError, match="pre-event history"):
        event_study_from_price_history(points, points, points[5].date, event_window_days=1)
