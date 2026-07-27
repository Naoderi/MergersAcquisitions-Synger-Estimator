"""WACC estimation: CAPM cost of equity on a real capital structure.

Cost of debt is a documented simplification (see project plan): risk-free
rate + a flat credit-spread proxy, since standalone interest expense isn't a
reliable top-level XBRL tag across filers (confirmed while building the data
pipeline -- e.g. Apple doesn't break it out separately). A synthetic-rating
approach (interest coverage -> spread table) is a possible future refinement,
not required for the MVP.
"""

from synergy_estimator.data.schema import AnnualFinancials, MarketData

# Damodaran's long-run implied US equity risk premium, approximate -- a cited
# constant input, not computed. Should be refreshed periodically from
# https://pages.stern.nyu.edu/~adamodar/
EQUITY_RISK_PREMIUM = 0.047

DEFAULT_CREDIT_SPREAD = 0.015

# US federal statutory corporate rate, used as a fallback when a company's
# effective tax rate can't be computed (e.g. a pretax loss year).
FALLBACK_TAX_RATE = 0.21


def cost_of_equity(risk_free_rate: float, beta: float) -> float:
    return risk_free_rate + beta * EQUITY_RISK_PREMIUM


def cost_of_debt(risk_free_rate: float, credit_spread: float = DEFAULT_CREDIT_SPREAD) -> float:
    return risk_free_rate + credit_spread


def effective_tax_rate(financials: AnnualFinancials) -> float | None:
    if not financials.pretax_income or not financials.income_tax_expense:
        return None
    if financials.pretax_income <= 0:
        return None
    return financials.income_tax_expense / financials.pretax_income


def estimate_wacc(market: MarketData, financials: AnnualFinancials, risk_free_rate: float) -> float:
    if market.beta is None:
        raise ValueError(f"{market.ticker}: missing beta, cannot compute WACC")

    equity = market.market_cap or 0.0
    debt = market.total_debt or 0.0
    total_capital = equity + debt
    if total_capital == 0:
        raise ValueError(f"{market.ticker}: missing market cap and debt, cannot compute WACC")

    tax_rate = effective_tax_rate(financials)
    if tax_rate is None:
        tax_rate = FALLBACK_TAX_RATE

    e_weight = equity / total_capital
    d_weight = debt / total_capital

    return e_weight * cost_of_equity(risk_free_rate, market.beta) + d_weight * cost_of_debt(
        risk_free_rate
    ) * (1 - tax_rate)


def blended_wacc(
    acquirer_wacc: float,
    acquirer_capital: float,
    target_wacc: float,
    target_capital: float,
) -> float:
    """Combined-entity WACC, weighted by each company's total capital (equity + debt)."""
    total_capital = acquirer_capital + target_capital
    if total_capital == 0:
        raise ValueError("Combined capital is zero, cannot blend WACC")
    return (acquirer_wacc * acquirer_capital + target_wacc * target_capital) / total_capital
