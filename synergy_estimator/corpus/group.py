"""Collapse a filing-level search result into deal-level extraction candidates.

One transaction generates a chain of filings -- the announcement 8-K, merger
communications on 425, the S-4 registration, the DEFM14A proxy -- all restating
the same synergy target. Extracting each separately would pay several times
over for one label and then over-weight that deal in the fit.

Deals are identified by the set of filers on the document plus a time window:
grouping on the CIK set alone would merge every acquisition a serial acquirer
ever made into a single "deal", so filings by the same parties separated by
more than `gap_days` are treated as distinct transactions.

The earliest filing in each cluster is the extraction target -- it is closest
to the announcement, where the original as-announced figure is stated, before
later documents revise it.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from synergy_estimator.corpus.discover import FilingHit

# Filings for one deal cluster tightly; announcement to closing proxy is
# typically a few months. Six months separates deal chains without splitting
# a long-running regulatory review into two.
_DEFAULT_GAP_DAYS = 180


@dataclass
class DealCandidate:
    """One transaction: the filing to extract from, plus its supporting chain."""

    primary: FilingHit  # earliest filing -- closest to the as-announced figure
    supporting: list[FilingHit]
    ciks: frozenset[str]

    @property
    def filing_count(self) -> int:
        return 1 + len(self.supporting)

    @property
    def sics(self) -> tuple[str, ...]:
        """SIC codes seen across the chain, most complete first."""
        seen: list[str] = []
        for hit in [self.primary, *self.supporting]:
            for sic in hit.sics:
                if sic not in seen:
                    seen.append(sic)
        return tuple(seen)


def _parse(day: str) -> date:
    return date.fromisoformat(day)


def group_into_deals(
    hits: list[FilingHit], gap_days: int = _DEFAULT_GAP_DAYS
) -> list[DealCandidate]:
    """Cluster filings into transactions by (shared party, time proximity).

    Filings are joined when they share *any* filer and fall within `gap_days`
    of each other, not when their full CIK sets match. One transaction is
    normally announced by an acquirer-only 8-K and a both-parties 425 on the
    same day; matching on the exact set would file those as two separate deals,
    and the 8-K-only one could never be resolved to a target.

    The time window is what keeps a serial acquirer's deals apart -- it shares
    its own CIK across all of them.
    """
    dated = sorted((h for h in hits if h.filing_date), key=lambda h: h.filing_date)
    gap = timedelta(days=gap_days)

    # Each open cluster is (cik set, last filing date, filings). A filing joins
    # the most recent open cluster it shares a party with.
    clusters: list[tuple[set[str], str, list[FilingHit]]] = []
    finished: list[tuple[set[str], list[FilingHit]]] = []

    for hit in dated:
        ciks = set(hit.ciks)
        joined = None
        still_open = []
        for cluster_ciks, last_date, filings in clusters:
            if _parse(hit.filing_date) - _parse(last_date) > gap:
                finished.append((cluster_ciks, filings))
                continue
            if joined is None and ciks & cluster_ciks:
                joined = (cluster_ciks, filings)
            still_open.append((cluster_ciks, last_date, filings))

        clusters = [c for c in still_open if not (joined and c[2] is joined[1])]
        if joined:
            joined[0].update(ciks)
            joined[1].append(hit)
            clusters.append((joined[0], hit.filing_date, joined[1]))
        else:
            clusters.append((ciks, hit.filing_date, [hit]))

    finished.extend((cik_set, filings) for cik_set, _, filings in clusters)

    deals = [
        DealCandidate(filings[0], filings[1:], frozenset(cik_set))
        for cik_set, filings in finished
        if filings
    ]
    return sorted(deals, key=lambda d: d.primary.filing_date)
