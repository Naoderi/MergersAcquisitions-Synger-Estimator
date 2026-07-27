"""Combine EDGAR filings + free market data into one CompanyProfile per ticker."""

from synergy_estimator.data.edgar_source import get_annual_financials
from synergy_estimator.data.market_source import get_market_data
from synergy_estimator.data.schema import CompanyProfile


def get_company_profile(ticker: str, years: int = 3) -> CompanyProfile:
    return CompanyProfile(
        ticker=ticker.upper(),
        market=get_market_data(ticker),
        financials=get_annual_financials(ticker, years=years),
    )
