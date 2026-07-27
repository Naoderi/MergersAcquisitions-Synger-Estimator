"""Pull company income-statement history from SEC EDGAR via edgartools.

Note: this targets standard corporate income statements (revenue/COGS/SG&A).
Banks and other financial institutions use a fundamentally different statement
structure (net interest income, provision for credit losses, no COGS) and are
out of scope -- they're valued/modeled differently in real M&A work too.
"""

import os
import re

import edgar
from edgar.financials import Financials

from synergy_estimator.data.schema import AnnualFinancials, BalanceSheetData

_IDENTITY_ENV_VAR = "EDGAR_IDENTITY"
_identity_set = False

# Each line item has a list of candidate XBRL concepts, in priority order,
# since different filers tag the same line item differently.
_CONCEPT_CANDIDATES: dict[str, list[str]] = {
    "revenue": [
        "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap_Revenues",
        "us-gaap_SalesRevenueNet",
    ],
    "cogs": [
        "us-gaap_CostOfGoodsAndServicesSold",
        "us-gaap_CostOfRevenue",
        "us-gaap_CostOfServices",
        "us-gaap_CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "us-gaap_CostOfGoodsSold",
        "us-gaap_DirectOperatingCosts",
    ],
    "gross_profit": [
        "us-gaap_GrossProfit",
    ],
    "rnd_expense": [
        "us-gaap_ResearchAndDevelopmentExpense",
    ],
    "sga_expense": [
        "us-gaap_SellingGeneralAndAdministrativeExpense",
        "us-gaap_GeneralAndAdministrativeExpense",
    ],
    "operating_income": [
        "us-gaap_OperatingIncomeLoss",
    ],
    "net_income": [
        "us-gaap_NetIncomeLoss",
    ],
    "shares_diluted": [
        "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "pretax_income": [
        "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax_expense": [
        "us-gaap_IncomeTaxExpenseBenefit",
    ],
}

_PERIOD_COLUMN_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) \(FY\)$")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Balance-sheet columns are point-in-time (an instant), not a duration, so
# edgartools labels them as a plain date with no "(FY)" suffix.
_BALANCE_SHEET_PERIOD_COLUMN_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_BALANCE_SHEET_CONCEPT_CANDIDATES: dict[str, list[str]] = {
    "total_assets": ["us-gaap_Assets"],
    "total_liabilities": ["us-gaap_Liabilities"],
    "total_current_assets": ["us-gaap_AssetsCurrent"],
    "total_current_liabilities": ["us-gaap_LiabilitiesCurrent"],
    "retained_earnings": ["us-gaap_RetainedEarningsAccumulatedDeficit"],
    "accounts_receivable": ["us-gaap_ReceivablesNetCurrent", "us-gaap_AccountsReceivableNetCurrent"],
    "inventory": ["us-gaap_InventoryNet", "us-gaap_FIFOInventoryAmount"],
    "accounts_payable": ["us-gaap_AccountsPayableCurrent", "us-gaap_AccountsPayableTradeCurrent"],
}

# Fallback concepts for total shareholders' equity, used only to derive
# total_liabilities via the accounting identity (Liabilities = Assets - Equity)
# when a filer doesn't tag a standalone Liabilities line (e.g. Republic
# Services reports only "total liabilities and stockholders' equity").
_EQUITY_CONCEPT_CANDIDATES = [
    "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "us-gaap_StockholdersEquity",
]

# Fallback for filers who tag SG&A/COGS with a company-specific custom XBRL
# extension concept instead of a standard us-gaap one (e.g. Kroger tags SG&A
# as `kr_OperatingGeneralAndAdministrativeExpense`). When no candidate concept
# matches, fall back to matching the human-readable label text instead.
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "cogs": ["merchandise cost", "cost of goods", "cost of revenue", "cost of sales", "cost of services"],
    "sga_expense": ["general and administrative", "selling, general"],
}


def set_edgar_identity(identity: str | None = None) -> None:
    """Configure the SEC-required requester identity ("Name email@example.com").

    Reads from the identity argument, then the EDGAR_IDENTITY env var. SEC's
    fair-access policy requires a real contact string on every request.
    """
    global _identity_set
    resolved = identity or os.environ.get(_IDENTITY_ENV_VAR)
    if not resolved:
        raise RuntimeError(
            f"Set the {_IDENTITY_ENV_VAR} environment variable to 'Your Name your@email.com' "
            "before pulling data from SEC EDGAR."
        )
    edgar.set_identity(resolved)
    _identity_set = True


def _fiscal_year_columns(df) -> list[str]:
    return [c for c in df.columns if _PERIOD_COLUMN_RE.match(c)]


def _first_available_value(df, concepts: list[str], column: str) -> float | None:
    top_level = df[df["dimension"] == False]  # noqa: E712
    for concept in concepts:
        rows = top_level[top_level["concept"] == concept]
        if not rows.empty:
            value = rows.iloc[0][column]
            if value == value:  # filters out NaN
                return float(value)
    return None


def _value_by_label_keyword(df, keywords: list[str], column: str) -> float | None:
    top_level = df[df["dimension"] == False]  # noqa: E712
    labels = top_level["label"].fillna("").str.lower()
    for keyword in keywords:
        rows = top_level[labels.str.contains(keyword, regex=False)]
        if not rows.empty:
            value = rows.iloc[0][column]
            if value == value:  # filters out NaN
                return float(value)
    return None


def _resolve_field_value(df, field: str, concepts: list[str], column: str) -> float | None:
    value = _first_available_value(df, concepts, column)
    if value is not None:
        return value
    keywords = _LABEL_KEYWORDS.get(field)
    if keywords:
        return _value_by_label_keyword(df, keywords, column)
    return None


def _annual_financials_from_dataframe(df, display_ticker: str, years: int) -> list[AnnualFinancials]:
    results = []
    for column in _fiscal_year_columns(df)[:years]:
        match = _PERIOD_COLUMN_RE.match(column)
        year, month, day = match.groups()
        values = {
            field: _resolve_field_value(df, field, concepts, column)
            for field, concepts in _CONCEPT_CANDIDATES.items()
        }
        results.append(
            AnnualFinancials(
                ticker=display_ticker,
                fiscal_year=int(year),
                period_end=f"{year}-{month}-{day}",
                **values,
            )
        )
    return results


def _financials_from_company(company, display_ticker: str, years: int) -> list[AnnualFinancials]:
    income_statement = company.get_financials().income_statement()
    return _annual_financials_from_dataframe(income_statement.to_dataframe(), display_ticker, years)


def get_annual_financials(ticker: str, years: int = 3) -> list[AnnualFinancials]:
    """Return up to `years` most-recent fiscal years of income-statement data."""
    if not _identity_set:
        set_edgar_identity()

    company = edgar.Company(ticker)
    return _financials_from_company(company, ticker.upper(), years)


def get_annual_financials_by_cik(cik: int, display_ticker: str, years: int = 3) -> list[AnnualFinancials]:
    """Like get_annual_financials, but looks up by CIK -- required for delisted
    tickers (e.g. an acquired target), since SEC keeps filings by CIK forever
    but ticker resolution stops working once a ticker is deactivated."""
    if not _identity_set:
        set_edgar_identity()

    company = edgar.Company(cik)
    return _financials_from_company(company, display_ticker.upper(), years)


def _newest_reported_before(
    candidates: list[AnnualFinancials], announcement_date: str
) -> AnnualFinancials | None:
    """Pick the latest fiscal year that had already closed on `announcement_date`.

    A 10-K's comparative columns run backwards from its own fiscal year, so the
    newest qualifying column is the one the market was actually pricing on.
    """
    reported = [f for f in candidates if f.period_end < announcement_date]
    if not reported:
        return None
    return max(reported, key=lambda f: f.period_end)


def get_financials_before(
    identifier: int | str, display_ticker: str, announcement_date: str, forms: tuple[str, ...] = ("10-K",)
) -> AnnualFinancials:
    """Return the newest fiscal year that had already been *reported* as of
    `announcement_date` (YYYY-MM-DD).

    This is the figure an analyst underwriting the deal would actually have had
    in front of them. It is deliberately not `get_annual_financials(...)[0]`:
    `Company.get_financials()` reads whichever 10-K is latest *today*, so
    scoring a 2020 deal that way compares a 2020 synergy target against 2025
    financials -- several years of organic growth the acquirer never claimed
    credit for. Here we instead pick the last 10-K filed before the
    announcement and read its most recent fiscal-year column.

    Raises ValueError when the filer has no qualifying annual report (e.g. a
    target that IPO'd months before being acquired) -- that deal genuinely
    can't be scored on pre-deal financials and should be dropped, not fudged.
    """
    if not _DATE_RE.match(announcement_date):
        raise ValueError(f"announcement_date must be YYYY-MM-DD, got {announcement_date!r}")
    if not _identity_set:
        set_edgar_identity()

    company = edgar.Company(identifier)
    # edgartools range syntax: ":YYYY-MM-DD" means "filed on or before".
    filings = company.get_filings(form=list(forms), filing_date=f":{announcement_date}")
    if filings is None or len(filings) == 0:
        raise ValueError(
            f"{display_ticker} ({identifier}): no {'/'.join(forms)} filed before {announcement_date} -- "
            "no pre-announcement financials exist for this deal."
        )

    # get_filings returns newest-first; the first hit is the last annual report
    # available to the market on the announcement date.
    financials = Financials.extract(filings[0])
    if financials is None:
        raise ValueError(
            f"{display_ticker} ({identifier}): could not parse financials from "
            f"{filings[0].form} filed {filings[0].filing_date}."
        )

    # XBRL only became mandatory around 2009-2011; older annual reports parse
    # into a Financials object whose statements are all None.
    income_statement = financials.income_statement()
    if income_statement is None:
        raise ValueError(
            f"{display_ticker} ({identifier}): {filings[0].form} filed {filings[0].filing_date} "
            "has no machine-readable income statement (pre-XBRL filing)."
        )

    df = income_statement.to_dataframe()
    # Ask for every column, then filter -- a fiscal year that ended before the
    # announcement but whose 10-K landed after it was not yet public, and the
    # comparative columns run backwards from there.
    candidates = _annual_financials_from_dataframe(df, display_ticker, years=len(_fiscal_year_columns(df)))
    reported = _newest_reported_before(candidates, announcement_date)
    if reported is None:
        raise ValueError(
            f"{display_ticker} ({identifier}): {filings[0].form} filed {filings[0].filing_date} "
            f"contains no fiscal year ending before {announcement_date}."
        )
    return reported


def _resolve_total_liabilities(df, column: str) -> float | None:
    direct = _first_available_value(df, _BALANCE_SHEET_CONCEPT_CANDIDATES["total_liabilities"], column)
    if direct is not None:
        return direct
    # Identity fallback: Liabilities = Assets - Equity, for filers (e.g. Republic
    # Services) that only tag "total liabilities and stockholders' equity"
    # rather than a standalone total-liabilities line.
    total_assets = _first_available_value(df, _BALANCE_SHEET_CONCEPT_CANDIDATES["total_assets"], column)
    equity = _first_available_value(df, _EQUITY_CONCEPT_CANDIDATES, column)
    if total_assets is not None and equity is not None:
        return total_assets - equity
    return None


def _balance_sheet_from_company(company, display_ticker: str, years: int) -> list[BalanceSheetData]:
    balance_sheet = company.get_financials().balance_sheet()
    df = balance_sheet.to_dataframe()

    results = []
    columns = [c for c in df.columns if _BALANCE_SHEET_PERIOD_COLUMN_RE.match(c)]
    for column in columns[:years]:
        match = _BALANCE_SHEET_PERIOD_COLUMN_RE.match(column)
        year, month, day = match.groups()
        values = {
            field: _resolve_field_value(df, field, concepts, column)
            for field, concepts in _BALANCE_SHEET_CONCEPT_CANDIDATES.items()
            if field != "total_liabilities"
        }
        values["total_liabilities"] = _resolve_total_liabilities(df, column)
        results.append(
            BalanceSheetData(
                ticker=display_ticker,
                fiscal_year=int(year),
                period_end=f"{year}-{month}-{day}",
                **values,
            )
        )
    return results


def get_balance_sheet(ticker: str, years: int = 3) -> list[BalanceSheetData]:
    """Return up to `years` most-recent fiscal-year-end balance sheets.

    Used only by target-quality diagnostics (Altman Z-score, working capital
    cycle) -- optional, supplementary data, not required for the core
    synergy/NPV/accretion model.
    """
    if not _identity_set:
        set_edgar_identity()

    company = edgar.Company(ticker)
    return _balance_sheet_from_company(company, ticker.upper(), years)


def get_balance_sheet_by_cik(cik: int, display_ticker: str, years: int = 3) -> list[BalanceSheetData]:
    """Like get_balance_sheet, but looks up by CIK for delisted tickers."""
    if not _identity_set:
        set_edgar_identity()

    company = edgar.Company(cik)
    return _balance_sheet_from_company(company, display_ticker.upper(), years)
