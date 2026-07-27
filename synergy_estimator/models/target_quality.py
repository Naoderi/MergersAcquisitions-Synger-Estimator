"""Target Quality: Altman Z-score and working-capital cycle diagnostics.

Covers the free-data-feasible techniques from the M&A analytical toolkit's
"Target Quality" section that are computable from a public company's XBRL
income statement + balance sheet: bankruptcy-risk screening (Altman Z-score)
and the cash conversion cycle (DSO/DIO/DPO).

Out of scope, and not faked: quality-of-earnings on transaction-level GL data,
cohort/survival analysis on customer churn, concentration/Herfindahl metrics,
price-volume-mix decomposition, Beneish M-score (needs multi-year depreciation
and cash-flow-statement line items this pipeline doesn't pull), win/loss and
NPS analysis, contract NLP, and cyber diligence -- all require either
transaction-level data, alternative data (card panels, web traffic, HR
systems), or contract/security-scan access that SEC EDGAR and Yahoo Finance
don't provide.
"""

from dataclasses import dataclass

from synergy_estimator.data.schema import AnnualFinancials, BalanceSheetData, MarketData


@dataclass
class AltmanZScoreResult:
    z_score: float
    zone: str  # "safe" | "grey" | "distress"


def altman_z_score(
    financials: AnnualFinancials, balance_sheet: BalanceSheetData, market: MarketData
) -> AltmanZScoreResult | None:
    """Altman Z-score (public-company "Model A"): a 1968 bankruptcy-risk
    screen still widely used as a quick target-quality sanity check.

    Z = 1.2*(working capital/assets) + 1.4*(retained earnings/assets)
      + 3.3*(EBIT/assets) + 0.6*(market equity/liabilities) + 1.0*(sales/assets)

    Zones: Z > 2.99 "safe", 1.81-2.99 "grey", Z < 1.81 "distress".
    Uses operating_income as an EBIT proxy (no separate EBIT line in this
    schema) and market cap as the market value of equity. Returns None if any
    required field is missing, rather than raising -- this is a supplementary
    diagnostic, not a hard requirement for the core synergy estimate.
    """
    total_assets = balance_sheet.total_assets
    if not total_assets:
        return None
    if balance_sheet.total_current_assets is None or balance_sheet.total_current_liabilities is None:
        return None
    if balance_sheet.retained_earnings is None:
        return None
    if financials.operating_income is None:
        return None
    if not balance_sheet.total_liabilities:
        return None
    if not market.market_cap:
        return None
    if financials.revenue is None:
        return None

    working_capital = balance_sheet.total_current_assets - balance_sheet.total_current_liabilities
    a = working_capital / total_assets
    b = balance_sheet.retained_earnings / total_assets
    c = financials.operating_income / total_assets
    d = market.market_cap / balance_sheet.total_liabilities
    e = financials.revenue / total_assets

    z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
    zone = "safe" if z > 2.99 else "distress" if z < 1.81 else "grey"
    return AltmanZScoreResult(z_score=z, zone=zone)


@dataclass
class WorkingCapitalDiagnostics:
    days_sales_outstanding: float | None
    days_inventory_outstanding: float | None
    days_payable_outstanding: float | None
    cash_conversion_cycle: float | None  # DSO + DIO - DPO


def working_capital_diagnostics(
    financials: AnnualFinancials, balance_sheet: BalanceSheetData
) -> WorkingCapitalDiagnostics:
    """DSO/DIO/DPO and the cash conversion cycle. Each metric degrades to None
    independently if its underlying balance-sheet field is missing (e.g. a
    services company with no inventory line) rather than failing the whole
    diagnostic -- unlike Altman Z-score, which needs every input to mean
    anything as a composite score.
    """
    dso = (
        balance_sheet.accounts_receivable / financials.revenue * 365
        if balance_sheet.accounts_receivable is not None and financials.revenue
        else None
    )
    dio = (
        balance_sheet.inventory / financials.cogs * 365
        if balance_sheet.inventory is not None and financials.cogs
        else None
    )
    dpo = (
        balance_sheet.accounts_payable / financials.cogs * 365
        if balance_sheet.accounts_payable is not None and financials.cogs
        else None
    )
    ccc = dso + dio - dpo if dso is not None and dio is not None and dpo is not None else None
    return WorkingCapitalDiagnostics(
        days_sales_outstanding=dso, days_inventory_outstanding=dio, days_payable_outstanding=dpo,
        cash_conversion_cycle=ccc,
    )
