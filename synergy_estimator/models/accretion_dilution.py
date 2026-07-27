"""Pro-forma EPS accretion/dilution -- the classic "is this deal accretive?" check.

Simplifying assumption: all-cash / no new share issuance by default
(new_shares_issued=0.0), since actual deal financing structure (cash vs.
stock, debt raised) isn't modeled here.
"""

from dataclasses import dataclass

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.wacc import FALLBACK_TAX_RATE, effective_tax_rate


@dataclass
class AccretionDilutionResult:
    acquirer_standalone_eps: float | None
    proforma_eps_without_synergies: float | None
    proforma_eps_with_synergies: float | None
    pct_change_without_synergies: float | None
    pct_change_with_synergies: float | None


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def after_tax_synergies(
    pretax_synergies: float, acquirer_financials: AnnualFinancials, target_financials: AnnualFinancials
) -> float:
    """Blends both companies' effective tax rates (falling back to the US
    statutory rate when a company's effective rate can't be computed)."""
    rates = [
        r
        for r in (effective_tax_rate(acquirer_financials), effective_tax_rate(target_financials))
        if r is not None
    ]
    blended_rate = sum(rates) / len(rates) if rates else FALLBACK_TAX_RATE
    return pretax_synergies * (1 - blended_rate)


def eps_accretion_dilution(
    acquirer_financials: AnnualFinancials,
    target_financials: AnnualFinancials,
    pretax_runrate_synergies: float,
    new_shares_issued: float = 0.0,
) -> AccretionDilutionResult:
    acquirer_shares = acquirer_financials.shares_diluted
    acquirer_eps = (
        acquirer_financials.net_income / acquirer_shares
        if acquirer_shares and acquirer_financials.net_income is not None
        else None
    )

    combined_shares = (acquirer_shares or 0.0) + new_shares_issued
    combined_net_income = (acquirer_financials.net_income or 0.0) + (target_financials.net_income or 0.0)
    synergy_benefit = after_tax_synergies(pretax_runrate_synergies, acquirer_financials, target_financials)

    proforma_wo = combined_net_income / combined_shares if combined_shares else None
    proforma_w = (combined_net_income + synergy_benefit) / combined_shares if combined_shares else None

    return AccretionDilutionResult(
        acquirer_standalone_eps=acquirer_eps,
        proforma_eps_without_synergies=proforma_wo,
        proforma_eps_with_synergies=proforma_w,
        pct_change_without_synergies=_pct_change(proforma_wo, acquirer_eps),
        pct_change_with_synergies=_pct_change(proforma_w, acquirer_eps),
    )
