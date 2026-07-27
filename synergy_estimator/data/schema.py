"""Common financials schema that all data sources normalize into."""

from dataclasses import dataclass


@dataclass
class AnnualFinancials:
    """One fiscal year of income-statement data for a single company."""

    ticker: str
    fiscal_year: int
    period_end: str
    revenue: float | None
    cogs: float | None
    gross_profit: float | None
    rnd_expense: float | None
    sga_expense: float | None
    operating_income: float | None
    net_income: float | None
    shares_diluted: float | None
    pretax_income: float | None
    income_tax_expense: float | None


@dataclass
class MarketData:
    """Live market snapshot for a single company."""

    ticker: str
    company_name: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None
    share_price: float | None
    shares_outstanding: float | None
    beta: float | None
    total_debt: float | None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    analyst_target_low: float | None = None
    analyst_target_mean: float | None = None
    analyst_target_high: float | None = None


@dataclass
class BalanceSheetData:
    """One fiscal year of balance-sheet data for a single company.

    Used only by the target-quality diagnostics (Altman Z-score, working
    capital cycle) -- the core synergy/NPV/accretion model never requires
    balance-sheet data, only the income statement.
    """

    ticker: str
    fiscal_year: int
    period_end: str
    total_assets: float | None
    total_liabilities: float | None
    total_current_assets: float | None
    total_current_liabilities: float | None
    retained_earnings: float | None
    accounts_receivable: float | None
    inventory: float | None
    accounts_payable: float | None


@dataclass
class PricePoint:
    """One trading day's closing price (and volume, where available)."""

    date: str  # YYYY-MM-DD
    close: float
    volume: float | None


@dataclass
class CompanyProfile:
    """Everything the synergy model needs for one company."""

    ticker: str
    market: MarketData
    financials: list[AnnualFinancials]  # most recent first
