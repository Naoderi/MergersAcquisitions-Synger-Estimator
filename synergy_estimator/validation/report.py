"""Summarize backtest results into the error statistics used to compare models.

Kept separate from `backtest.py` so the Phase 2 recalibration can score itself
against exactly the same metrics as the pre-calibration baseline. Run directly
for a table:

    python -m synergy_estimator.validation.report
"""

import statistics
from dataclasses import dataclass

from synergy_estimator.validation.backtest import BacktestResult


@dataclass
class BacktestSummary:
    scored: int
    skipped: int
    median_abs_error: float
    mean_abs_error: float
    median_signed_error: float  # negative => model systematically underestimates
    min_error: float
    max_error: float
    underestimates: int

    def describe(self) -> str:
        return (
            f"scored {self.scored} deals ({self.skipped} skipped) | "
            f"median abs error {self.median_abs_error:.1%} | "
            f"range {self.min_error:+.1%} to {self.max_error:+.1%} | "
            f"median signed {self.median_signed_error:+.1%} | "
            f"underestimates {self.underestimates}/{self.scored}"
        )


def summarize(results: list[BacktestResult]) -> BacktestSummary:
    errors = [r.pct_error for r in results if r.scored]
    if not errors:
        raise ValueError("no deals were scored -- cannot summarize an empty backtest")
    return BacktestSummary(
        scored=len(errors),
        skipped=sum(1 for r in results if not r.scored),
        median_abs_error=statistics.median(abs(e) for e in errors),
        mean_abs_error=statistics.mean(abs(e) for e in errors),
        median_signed_error=statistics.median(errors),
        min_error=min(errors),
        max_error=max(errors),
        underestimates=sum(1 for e in errors if e < 0),
    )


def format_table(results: list[BacktestResult]) -> str:
    header = (
        f"{'DEAL':<12} {'ANNOUNCED':<12} {'EST $M':>8} {'DISC $M':>8} {'ERROR':>9}\n" + "-" * 53
    )
    rows = []
    for r in results:
        pair = f"{r.deal.acquirer_ticker}/{r.deal.target_ticker}"
        if not r.scored:
            rows.append(f"{pair:<12} {r.deal.announcement_date:<12} SKIPPED: {r.skipped_reason[:40]}")
            continue
        rows.append(
            f"{pair:<12} {r.deal.announcement_date:<12} "
            f"{r.estimated_synergies / 1e6:>8,.0f} {r.disclosed_synergies / 1e6:>8,.0f} "
            f"{r.pct_error:>+8.1%}"
        )
    return "\n".join([header, *rows])


if __name__ == "__main__":
    from synergy_estimator.validation.backtest import run_backtest

    results = run_backtest()
    print(format_table(results))
    print("-" * 53)
    print(summarize(results).describe())
