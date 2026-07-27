"""Backtest estimate_synergies() against real disclosed deal synergy targets.

Every deal is scored on the financials that were public *on its announcement
date* -- see `get_financials_before`. Scoring against today's latest 10-K
instead (the original implementation) inflated older deals badly: ADI's SG&A
was 94% higher in FY2025 than in the FY2019 report an analyst would have had
when the Maxim deal was announced in July 2020, and the cost-synergy model is
linear in SG&A, so that deal's estimate was roughly doubled by the mismatch.
"""

from dataclasses import dataclass

from synergy_estimator.data.cache import CachedFailure, cached
from synergy_estimator.data.edgar_source import get_financials_before
from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import SynergyAssumptions, SynergyEstimate, estimate_synergies
from synergy_estimator.validation.deals import DEALS, HistoricalDeal


@dataclass
class BacktestResult:
    deal: HistoricalDeal
    estimated_synergies: float | None
    disclosed_synergies: float
    pct_error: float | None  # (estimated - disclosed) / disclosed
    skipped_reason: str | None = None  # set when pre-deal financials were unavailable

    @property
    def scored(self) -> bool:
        return self.pct_error is not None


def score_deal(estimate: SynergyEstimate, deal: HistoricalDeal) -> BacktestResult:
    estimated = (
        estimate.cost_synergies_runrate if deal.synergy_type == "cost" else estimate.total_runrate_synergies
    )
    disclosed = deal.disclosed_synergy_runrate
    return BacktestResult(
        deal=deal,
        estimated_synergies=estimated,
        disclosed_synergies=disclosed,
        pct_error=(estimated - disclosed) / disclosed,
    )


def _pre_deal_financials(identifier: int | str, display_ticker: str, announcement_date: str):
    """Cached fetch of the last fiscal year reported before `announcement_date`."""
    return cached(
        "financials_before",
        f"{identifier}_{announcement_date}",
        AnnualFinancials,
        lambda: get_financials_before(identifier, display_ticker, announcement_date),
    )


def run_backtest(
    deals: list[HistoricalDeal] | None = None,
    assumptions: SynergyAssumptions | None = None,
    fetch=_pre_deal_financials,
) -> list[BacktestResult]:
    """Network-touching orchestration: for each deal, fetch the financials that
    were public on its announcement date and score estimate_synergies() against
    the disclosed target.

    Deals whose filers have no pre-announcement annual report are skipped rather
    than scored on post-deal data -- dropping them is honest, silently
    substituting today's financials is not. `fetch` is injectable so tests can
    run the orchestration without touching the network.
    """
    deals = deals if deals is not None else DEALS
    results = []
    for deal in deals:
        try:
            acquirer_financials = fetch(
                deal.acquirer_cik or deal.acquirer_ticker,
                deal.acquirer_ticker,
                deal.announcement_date,
            )
            target_financials = fetch(
                deal.target_cik or deal.target_ticker, deal.target_ticker, deal.announcement_date
            )
        except (ValueError, CachedFailure) as exc:
            results.append(
                BacktestResult(
                    deal=deal,
                    estimated_synergies=None,
                    disclosed_synergies=deal.disclosed_synergy_runrate,
                    pct_error=None,
                    skipped_reason=str(exc),
                )
            )
            continue
        estimate = estimate_synergies(
            deal.acquirer_ticker, acquirer_financials, deal.target_ticker, target_financials, assumptions
        )
        results.append(score_deal(estimate, deal))
    return results
