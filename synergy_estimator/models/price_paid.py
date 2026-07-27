"""Price Paid: contribution analysis, implied ownership, premium, VWAP, valuation range.

Covers the free-data-feasible techniques from the M&A analytical toolkit's
"Price Paid" section: contribution analysis, VWAP/unaffected-price premium
analysis, and a simplified "football field" valuation range (52-week trading
range + analyst target range). Full comparable-companies/precedent-transaction
analysis and DCF/LBO valuation floors require a deals database (PitchBook /
Mergermarket) or a full standalone-company DCF model, neither of which this
free-data pipeline has -- out of scope, not faked.
"""

from dataclasses import dataclass

from synergy_estimator.data.schema import AnnualFinancials, MarketData, PricePoint


@dataclass
class ContributionRow:
    metric: str
    acquirer_value: float
    target_value: float
    acquirer_pct: float
    target_pct: float


def contribution_analysis(acquirer: AnnualFinancials, target: AnnualFinancials) -> list[ContributionRow]:
    """Each company's % contribution to the combined entity's revenue,
    operating income, and net income -- the classic "is the ownership split
    fair relative to what each side brings" check in a merger of equals.
    Skips any metric where either company's value is missing or the combined
    total is zero, rather than failing the whole analysis.
    """
    rows = []
    for label, attr in (("Revenue", "revenue"), ("Operating income", "operating_income"), ("Net income", "net_income")):
        acquirer_value = getattr(acquirer, attr)
        target_value = getattr(target, attr)
        if acquirer_value is None or target_value is None:
            continue
        combined = acquirer_value + target_value
        if combined == 0:
            continue
        rows.append(
            ContributionRow(
                metric=label,
                acquirer_value=acquirer_value,
                target_value=target_value,
                acquirer_pct=acquirer_value / combined,
                target_pct=target_value / combined,
            )
        )
    return rows


@dataclass
class OwnershipSplit:
    acquirer_ownership_pct: float
    target_ownership_pct: float
    implied_target_deal_value: float


def implied_ownership_split(
    acquirer_market_cap: float, target_market_cap: float, premium_pct: float = 0.0
) -> OwnershipSplit:
    """Target shareholders' implied % ownership of the combined entity in an
    all-stock merger, given a premium paid over the target's current market
    cap. Compare against contribution_analysis()'s % contributions: a target
    receiving materially more ownership than it contributes in revenue/income
    is a common merger-of-equals red flag (or a sign the premium is rich).
    """
    if not acquirer_market_cap or not target_market_cap:
        raise ValueError("Both acquirer and target market cap are required to compute an ownership split.")
    target_deal_value = target_market_cap * (1 + premium_pct)
    combined = acquirer_market_cap + target_deal_value
    target_pct = target_deal_value / combined
    return OwnershipSplit(
        acquirer_ownership_pct=1 - target_pct,
        target_ownership_pct=target_pct,
        implied_target_deal_value=target_deal_value,
    )


def acquisition_premium(offer_price_per_share: float, unaffected_price_per_share: float) -> float:
    """% premium of an offer price over the target's unaffected (pre-rumor/
    pre-announcement) share price -- the standard "is this a rich premium"
    metric in a straight acquisition (as opposed to a stock-for-stock merger
    of equals, where implied_ownership_split is the more relevant lens)."""
    if not unaffected_price_per_share:
        raise ValueError("unaffected_price_per_share must be nonzero.")
    return (offer_price_per_share - unaffected_price_per_share) / unaffected_price_per_share


def implied_offer_price(unaffected_price_per_share: float, premium_pct: float) -> float:
    """Inverse of acquisition_premium: the offer price implied by a target premium %."""
    return unaffected_price_per_share * (1 + premium_pct)


def vwap(price_points: list[PricePoint]) -> float | None:
    """Volume-weighted average price over a window of trading days -- a common
    reference point for "what was this stock really worth" that's less
    sensitive to a single noisy closing print than the raw unaffected price.
    Returns None if no points have volume data.
    """
    weighted_sum = 0.0
    total_volume = 0.0
    for point in price_points:
        if point.volume:
            weighted_sum += point.close * point.volume
            total_volume += point.volume
    if total_volume == 0:
        return None
    return weighted_sum / total_volume


@dataclass
class ValuationRangeBar:
    label: str
    low: float
    high: float


def valuation_range(market: MarketData) -> list[ValuationRangeBar]:
    """A simplified "football field": the target's 52-week trading range and
    sell-side analyst target range, packaged for a range-bar chart. A full
    football field (DCF, comps multiples, precedent transaction premia) needs
    a standalone DCF model and a deals database this pipeline doesn't have;
    these two free, Yahoo-Finance-sourced ranges are a partial substitute.
    """
    bars = []
    if market.fifty_two_week_low is not None and market.fifty_two_week_high is not None:
        bars.append(ValuationRangeBar("52-week range", market.fifty_two_week_low, market.fifty_two_week_high))
    if market.analyst_target_low is not None and market.analyst_target_high is not None:
        bars.append(
            ValuationRangeBar("Analyst target range", market.analyst_target_low, market.analyst_target_high)
        )
    return bars
