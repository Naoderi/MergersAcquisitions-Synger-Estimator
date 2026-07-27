"""Tornado sensitivity: how much synergy NPV moves as each key assumption swings."""

import dataclasses
from dataclasses import dataclass

from synergy_estimator.data.schema import AnnualFinancials
from synergy_estimator.models.synergies import SynergyAssumptions, estimate_synergies
from synergy_estimator.models.valuation import npv_of_synergies

# WACC is perturbed in absolute basis points (the standard banker convention)
# rather than as a relative percentage, since a 25%-of-rate swing on an
# already-small decimal is a much less meaningful stress test than +/-150bps.
_WACC_BPS_SWING = 0.015

_PERTURBABLE_FIELDS = (
    "sga_overlap_rate",
    "cogs_overlap_rate",
    "revenue_cross_sell_rate",
    "integration_cost_multiple",
)


@dataclass
class SensitivityRow:
    label: str
    low_value: float
    high_value: float
    low_npv: float
    base_npv: float
    high_npv: float


def _npv_for_assumptions(
    acquirer_ticker: str,
    acquirer_financials: AnnualFinancials,
    target_ticker: str,
    target_financials: AnnualFinancials,
    assumptions: SynergyAssumptions,
    wacc: float,
    forecast_years: int,
) -> float:
    estimate = estimate_synergies(acquirer_ticker, acquirer_financials, target_ticker, target_financials, assumptions)
    return npv_of_synergies(estimate, wacc, forecast_years=forecast_years)


def tornado_sensitivity(
    acquirer_ticker: str,
    acquirer_financials: AnnualFinancials,
    target_ticker: str,
    target_financials: AnnualFinancials,
    base_assumptions: SynergyAssumptions,
    wacc: float,
    forecast_years: int = 5,
    perturbation_pct: float = 0.25,
) -> list[SensitivityRow]:
    """Perturbs each key driver +/-perturbation_pct (WACC uses +/-150bps instead,
    the standard banker convention) holding all others at base, recomputes NPV,
    and returns rows sorted by |high_npv - low_npv| descending (tornado order).
    """
    base_npv = _npv_for_assumptions(
        acquirer_ticker, acquirer_financials, target_ticker, target_financials, base_assumptions, wacc, forecast_years
    )

    rows = []
    for field in _PERTURBABLE_FIELDS:
        base_value = getattr(base_assumptions, field)
        low_value = base_value * (1 - perturbation_pct)
        high_value = base_value * (1 + perturbation_pct)

        low_npv = _npv_for_assumptions(
            acquirer_ticker,
            acquirer_financials,
            target_ticker,
            target_financials,
            dataclasses.replace(base_assumptions, **{field: low_value}),
            wacc,
            forecast_years,
        )
        high_npv = _npv_for_assumptions(
            acquirer_ticker,
            acquirer_financials,
            target_ticker,
            target_financials,
            dataclasses.replace(base_assumptions, **{field: high_value}),
            wacc,
            forecast_years,
        )
        rows.append(
            SensitivityRow(
                label=field, low_value=low_value, high_value=high_value, low_npv=low_npv, base_npv=base_npv, high_npv=high_npv
            )
        )

    low_wacc, high_wacc = wacc - _WACC_BPS_SWING, wacc + _WACC_BPS_SWING
    rows.append(
        SensitivityRow(
            label="wacc",
            low_value=low_wacc,
            high_value=high_wacc,
            # Lower WACC discounts less -> higher NPV, so low_value maps to the larger NPV here.
            low_npv=_npv_for_assumptions(
                acquirer_ticker, acquirer_financials, target_ticker, target_financials, base_assumptions, low_wacc, forecast_years
            ),
            base_npv=base_npv,
            high_npv=_npv_for_assumptions(
                acquirer_ticker, acquirer_financials, target_ticker, target_financials, base_assumptions, high_wacc, forecast_years
            ),
        )
    )

    rows.sort(key=lambda row: abs(row.high_npv - row.low_npv), reverse=True)
    return rows
