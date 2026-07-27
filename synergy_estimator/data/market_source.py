"""Pull live market data from Yahoo Finance (free-tier data source)."""

import yfinance as yf

from synergy_estimator.data.schema import MarketData, PricePoint


def get_market_data(ticker: str) -> MarketData:
    info = yf.Ticker(ticker).info
    return MarketData(
        ticker=ticker.upper(),
        company_name=info.get("shortName"),
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        share_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        shares_outstanding=info.get("sharesOutstanding"),
        beta=info.get("beta"),
        total_debt=info.get("totalDebt"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        analyst_target_low=info.get("targetLowPrice"),
        analyst_target_mean=info.get("targetMeanPrice"),
        analyst_target_high=info.get("targetHighPrice"),
    )


def get_risk_free_rate() -> float:
    """Live 10-year Treasury yield (as a decimal, e.g. 0.0468), used as the CAPM risk-free rate."""
    ten_year = yf.Ticker("^TNX").info.get("regularMarketPrice")
    if ten_year is None:
        raise RuntimeError("Could not fetch ^TNX (10-year Treasury yield) from Yahoo Finance.")
    return ten_year / 100


def get_price_history(ticker: str, start: str, end: str) -> list[PricePoint]:
    """Daily close price (split/dividend-adjusted) and volume between start and
    end (YYYY-MM-DD, inclusive), oldest first. Works for equity tickers and
    index tickers alike (e.g. "^GSPC" for the S&P 500, used as the market
    benchmark in event studies).

    Note: like all Yahoo Finance data, this is NOT retained for delisted
    tickers -- a target that's since been acquired and delisted will return
    an empty list for any window after its last trading day.
    """
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    points = []
    for index, row in df.iterrows():
        close = row.get("Close")
        volume = row.get("Volume")
        if close != close:  # NaN
            continue
        points.append(
            PricePoint(
                date=index.strftime("%Y-%m-%d"),
                close=float(close),
                volume=float(volume) if volume == volume else None,
            )
        )
    return points
