"""Post-Deal: event-study cumulative abnormal returns (market model).

Covers the one free-data-feasible technique from the toolkit's "Post-Deal"
section that generalizes to any two tickers with an announcement date: the
market-model event study (cumulative abnormal returns around announcement),
computed from Yahoo Finance daily price history against a market-index
benchmark (e.g. "^GSPC" for the S&P 500).

Two other Post-Deal items are already covered elsewhere in this codebase,
not duplicated here:
- "Re-underwriting: variance bridge from deal model to actuals" is exactly
  what synergy_estimator/validation/backtest.py does (estimated vs. disclosed
  synergies) for the 8 curated historical deals.
- "ROIC bridge... on invested capital" reuses models/valuation.py's roic()
  and leverage_ratio(), applied to pre- vs. post-deal financials.

Out of scope, and not faked: buy-and-hold abnormal returns and calendar-time
portfolio regressions (need a broader peer/factor universe than a single
market index), difference-in-differences vs. a matched peer control group
(needs a peer-matching methodology), impairment testing under ASC 350/IAS 36
(needs goodwill/reporting-unit disclosures this pipeline doesn't parse), and
retained-cohort/cross-sell tracking (needs customer-level data no public
filing discloses).
"""

from dataclasses import dataclass

from synergy_estimator.data.schema import PricePoint


def _ols_alpha_beta(market_returns: list[float], stock_returns: list[float]) -> tuple[float, float]:
    """Simple OLS market-model regression: stock_return = alpha + beta * market_return."""
    n = len(market_returns)
    if n == 0 or n != len(stock_returns):
        raise ValueError("market_returns and stock_returns must be the same nonzero length.")
    mean_market = sum(market_returns) / n
    mean_stock = sum(stock_returns) / n
    covariance = sum((m - mean_market) * (s - mean_stock) for m, s in zip(market_returns, stock_returns))
    variance_market = sum((m - mean_market) ** 2 for m in market_returns)
    if variance_market == 0:
        raise ValueError("market_returns has zero variance -- cannot estimate beta.")
    beta = covariance / variance_market
    alpha = mean_stock - beta * mean_market
    return alpha, beta


@dataclass
class EventStudyResult:
    alpha: float
    beta: float
    abnormal_returns: list[float]  # one per trading day in the event window
    cumulative_abnormal_return: float


def event_study_car(
    estimation_stock_returns: list[float],
    estimation_market_returns: list[float],
    event_stock_returns: list[float],
    event_market_returns: list[float],
) -> EventStudyResult:
    """Market-model cumulative abnormal return (CAR): fits alpha/beta on a
    pre-event estimation window (unaffected by the deal), then measures how
    much the stock's actual return in the event window deviated from what
    that market model predicts. A positive target CAR around announcement is
    the classic signature of the market pricing in a deal premium.
    """
    alpha, beta = _ols_alpha_beta(estimation_market_returns, estimation_stock_returns)
    if len(event_stock_returns) != len(event_market_returns):
        raise ValueError("event_stock_returns and event_market_returns must be the same length.")
    abnormal_returns = [
        stock_r - (alpha + beta * market_r) for stock_r, market_r in zip(event_stock_returns, event_market_returns)
    ]
    return EventStudyResult(
        alpha=alpha,
        beta=beta,
        abnormal_returns=abnormal_returns,
        cumulative_abnormal_return=sum(abnormal_returns),
    )


def _align_and_compute_returns(
    stock_points: list[PricePoint], market_points: list[PricePoint]
) -> tuple[list[str], list[float], list[float]]:
    """Aligns two price series by date (inner join) and returns
    (dates, stock_returns, market_returns), where each entry is the
    day-over-day return ending on that date."""
    stock_by_date = {p.date: p.close for p in stock_points}
    market_by_date = {p.date: p.close for p in market_points}
    common_dates = sorted(set(stock_by_date) & set(market_by_date))

    dates, stock_returns, market_returns = [], [], []
    for prev_date, curr_date in zip(common_dates, common_dates[1:]):
        stock_returns.append((stock_by_date[curr_date] - stock_by_date[prev_date]) / stock_by_date[prev_date])
        market_returns.append((market_by_date[curr_date] - market_by_date[prev_date]) / market_by_date[prev_date])
        dates.append(curr_date)
    return dates, stock_returns, market_returns


def event_study_from_price_history(
    stock_points: list[PricePoint],
    market_points: list[PricePoint],
    event_date: str,
    event_window_days: int = 5,
    min_estimation_days: int = 30,
) -> EventStudyResult:
    """Orchestrates event_study_car() from raw price history: aligns the
    stock and market-index series by date, treats everything before the event
    window as the estimation window, and the +/-event_window_days trading
    days around event_date as the event window.

    Raises ValueError if event_date isn't in the aligned trading history
    (e.g. the target has since delisted and Yahoo Finance no longer serves
    price data past its last trading day) or if there isn't enough pre-event
    history to estimate a reliable beta.
    """
    dates, stock_returns, market_returns = _align_and_compute_returns(stock_points, market_points)
    if event_date not in dates:
        raise ValueError(
            f"{event_date} not found in the aligned trading history -- either it's not a trading day, "
            "or price data isn't available that far back/forward (common for delisted tickers)."
        )

    event_index = dates.index(event_date)
    start = max(0, event_index - event_window_days)
    end = min(len(dates), event_index + event_window_days + 1)

    estimation_stock = stock_returns[:start]
    estimation_market = market_returns[:start]
    if len(estimation_stock) < min_estimation_days:
        raise ValueError(
            f"Only {len(estimation_stock)} days of pre-event history available; need at least "
            f"{min_estimation_days} to estimate a reliable market-model beta."
        )

    return event_study_car(estimation_stock, estimation_market, stock_returns[start:end], market_returns[start:end])
