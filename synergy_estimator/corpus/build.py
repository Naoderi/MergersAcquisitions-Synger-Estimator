"""Assemble the calibration corpus: one row per deal, all from free EDGAR data.

Pipeline per deal candidate:

    filings -> parties (SGML header) -> disclosed figure (regex)
            -> pre-announcement financials for both sides -> row

Every stage drops deals, and the reason is recorded rather than swallowed. The
attrition table is a result in its own right: it is the honest answer to "how
many US public-public mergers actually put a number on the record, in a form
you can tie to both parties' financials?"

Run:

    python -m synergy_estimator.corpus.build --out data/deals_corpus.csv
"""

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

import requests

from synergy_estimator.corpus.discover import FilingHit, load
from synergy_estimator.corpus.filings import fetch_filing_text
from synergy_estimator.corpus.group import DealCandidate, group_into_deals
from synergy_estimator.corpus.parties import DealParties, Party, fetch_parties
from synergy_estimator.corpus.patterns import extract_figures, headline_figure
from synergy_estimator.corpus.sectors import sector_for_sic
from synergy_estimator.data.cache import CachedFailure, cached
from synergy_estimator.data.edgar_source import get_financials_before
from synergy_estimator.data.schema import AnnualFinancials

# How far down a deal's filing chain to look for a header naming both parties.
_MAX_CHAIN_LOOKUPS = 5


@dataclass
class CorpusRow:
    """One calibrated observation, with its provenance."""

    deal_id: str  # primary accession
    announcement_date: str
    acquirer_name: str
    acquirer_cik: int
    acquirer_sic: str | None
    target_name: str
    target_cik: int
    target_sic: str | None
    sector: str  # from the target's SIC -- the absorbed cost base
    disclosed_synergy_usd: float
    synergy_type: str
    acquirer_sga: float
    acquirer_cogs: float
    target_sga: float
    target_cogs: float
    combined_sga: float
    combined_cogs: float
    # Revenue is not used by the synergy model itself -- it is here so the
    # naive baselines ("synergies as a % of combined revenue") can be scored
    # on exactly the same rows as the fitted model.
    acquirer_revenue: float | None
    target_revenue: float | None
    combined_revenue: float | None
    acquirer_fy: int
    target_fy: int
    source_quote: str
    source_url: str


def _resolve_parties(
    session: requests.Session, deal: DealCandidate
) -> tuple[DealParties | None, str | None]:
    """Identify acquirer and target, or explain why not."""
    acquirer: Party | None = None
    for hit in [deal.primary, *deal.supporting][:_MAX_CHAIN_LOOKUPS]:
        try:
            parties = fetch_parties(session, hit.accession, hit.ciks[0] if hit.ciks else "0")
        except (requests.RequestException, OSError):
            continue
        if parties.is_resolved:
            return parties, None
        if parties.acquirer and acquirer is None:
            acquirer = parties.acquirer

    # Fallback: a filing listing exactly two CIKs, one of them the known filer.
    # The other party is the counterparty by elimination.
    if acquirer is not None:
        for hit in [deal.primary, *deal.supporting][:_MAX_CHAIN_LOOKUPS]:
            ciks = {int(c) for c in hit.ciks}
            if len(ciks) == 2 and acquirer.cik in ciks:
                (other,) = ciks - {acquirer.cik}
                sic = next((s for s in deal.sics if s != acquirer.sic), None)
                return DealParties(acquirer, Party(name=f"CIK {other}", cik=other, sic=sic)), None

    return None, "no filing names both parties"


def _financials(identifier: int, ticker: str, announcement_date: str) -> AnnualFinancials:
    return cached(
        "financials_before",
        f"{identifier}_{announcement_date}",
        AnnualFinancials,
        lambda: get_financials_before(identifier, ticker, announcement_date),
    )


def build_row(
    session: requests.Session, deal: DealCandidate, fetch_text
) -> tuple[CorpusRow | None, str]:
    """Turn one deal candidate into a corpus row, or say why it was dropped.

    `fetch_text` is called per filing so the figure can be looked for across the
    whole chain: the earliest filing announces the deal, but the document that
    actually quantifies the target is often a later investor presentation or
    the merger proxy.
    """
    parties, failure = _resolve_parties(session, deal)
    if parties is None:
        return None, failure or "unresolved parties"

    figure = None
    for hit in [deal.primary, *deal.supporting][:_MAX_CHAIN_LOOKUPS]:
        try:
            figure = headline_figure(extract_figures(fetch_text(hit)))
        except (requests.RequestException, OSError):
            continue
        if figure is not None:
            break
    if figure is None:
        return None, "no quantified synergy figure"

    sector = sector_for_sic(parties.target.sic or parties.acquirer.sic)
    if not sector.supported:
        return None, f"sector out of scope ({sector.key})"

    announced = deal.primary.filing_date
    try:
        acquirer_fin = _financials(parties.acquirer.cik, parties.acquirer.name, announced)
        target_fin = _financials(parties.target.cik, parties.target.name, announced)
    except (ValueError, CachedFailure) as exc:
        return None, f"no pre-announcement financials ({str(exc)[:60]})"

    missing = [
        name
        for name, value in (
            ("acquirer SG&A", acquirer_fin.sga_expense),
            ("acquirer COGS", acquirer_fin.cogs),
            ("target SG&A", target_fin.sga_expense),
            ("target COGS", target_fin.cogs),
        )
        if value is None
    ]
    if missing:
        return None, f"missing line items ({', '.join(missing)})"

    return (
        CorpusRow(
            deal_id=deal.primary.accession,
            announcement_date=announced,
            acquirer_name=parties.acquirer.name,
            acquirer_cik=parties.acquirer.cik,
            acquirer_sic=parties.acquirer.sic,
            target_name=parties.target.name,
            target_cik=parties.target.cik,
            target_sic=parties.target.sic,
            sector=sector.key,
            disclosed_synergy_usd=figure.amount_usd,
            synergy_type=figure.synergy_type,
            acquirer_sga=acquirer_fin.sga_expense,
            acquirer_cogs=acquirer_fin.cogs,
            target_sga=target_fin.sga_expense,
            target_cogs=target_fin.cogs,
            combined_sga=acquirer_fin.sga_expense + target_fin.sga_expense,
            combined_cogs=acquirer_fin.cogs + target_fin.cogs,
            acquirer_revenue=acquirer_fin.revenue,
            target_revenue=target_fin.revenue,
            combined_revenue=(
                acquirer_fin.revenue + target_fin.revenue
                if acquirer_fin.revenue is not None and target_fin.revenue is not None
                else None
            ),
            acquirer_fy=acquirer_fin.fiscal_year,
            target_fy=target_fin.fiscal_year,
            source_quote=figure.quote,
            source_url=deal.primary.document_url,
        ),
        "ok",
    )


def deduplicate(rows: list[CorpusRow], window_days: int = 400) -> list[CorpusRow]:
    """Collapse rows describing the same transaction.

    One deal can survive the pipeline more than once: both parties file their
    own merger communications, so the same transaction appears with acquirer
    and target swapped, and a long deal generates filing clusters far enough
    apart in time to group separately. A hand audit found 14% of rows were
    duplicates like this -- left in, they double-weight those deals in the fit
    and overstate the sample size.

    Deals are keyed on the *unordered* party pair, so direction does not
    matter, and the earliest announcement in each cluster is kept as the
    as-announced figure.
    """
    by_pair: dict[frozenset[int], list[CorpusRow]] = {}
    for row in rows:
        by_pair.setdefault(frozenset({row.acquirer_cik, row.target_cik}), []).append(row)

    kept: list[CorpusRow] = []
    for cluster in by_pair.values():
        cluster.sort(key=lambda r: r.announcement_date)
        current = cluster[0]
        kept.append(current)
        for row in cluster[1:]:
            gap = date.fromisoformat(row.announcement_date) - date.fromisoformat(current.announcement_date)
            if gap.days > window_days:
                kept.append(row)
                current = row
    return sorted(kept, key=lambda r: r.announcement_date)


def write_csv(rows: list[CorpusRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(CorpusRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _cli() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates.json"))
    parser.add_argument("--out", type=Path, default=Path("data/deals_corpus.csv"))
    parser.add_argument("--limit", type=int, default=None)
    # XBRL only became mandatory around 2009-2011. Earlier annual reports have
    # no machine-readable income statement, so their deals can never yield a
    # row -- excluding them up front keeps the attrition table meaningful.
    parser.add_argument("--since", default="2010-01-01", help="Drop deals announced before this date.")
    args = parser.parse_args()

    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise SystemExit("Set EDGAR_IDENTITY='Your Name your@email.com'.")

    session = requests.Session()
    session.headers["User-Agent"] = identity

    deals = [d for d in group_into_deals(load(args.candidates)) if d.primary.filing_date >= args.since]
    print(f"{len(deals)} deals announced on/after {args.since}")
    deals = deals[: args.limit]
    rows: list[CorpusRow] = []
    reasons: Counter = Counter()

    def fetch_text(hit: FilingHit) -> str:
        return fetch_filing_text(session, hit)

    for index, deal in enumerate(deals, 1):
        try:
            row, reason = build_row(session, deal, fetch_text)
        except Exception as exc:  # a single bad filer must not end the run
            reasons[f"error: {type(exc).__name__}"] += 1
            continue
        reasons[reason.split(" (")[0]] += 1
        if row:
            rows.append(row)
        if index % 100 == 0:
            print(f"  {index}/{len(deals)} -- {len(rows)} rows so far", flush=True)

    deduped = deduplicate(rows)
    if len(deduped) != len(rows):
        print(f"deduplicated {len(rows)} -> {len(deduped)} rows (same deal filed by both sides)")
    write_csv(deduped, args.out)
    summary_path = args.out.with_suffix(".attrition.json")
    summary_path.write_text(json.dumps(dict(reasons.most_common()), indent=2))

    print(f"\n{len(deduped)} corpus rows -> {args.out}")
    print(f"attrition -> {summary_path}")
    for reason, count in reasons.most_common():
        print(f"  {count:>5}  {reason}")


if __name__ == "__main__":
    _cli()
