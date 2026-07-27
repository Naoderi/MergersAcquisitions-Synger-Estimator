"""Find merger filings that disclose a synergy target, via EDGAR full-text search.

Disclosed synergies are not a field in any commercial deals database -- they
live in the prose of merger 8-Ks, 425s and S-4s. EDGAR's full-text search
covers 2001-present, so searching for the disclosure language directly yields
the deal universe *and* the label source in one step, and finds precisely the
deals that put a number on the record.

Search hits already carry the filer's CIK and SIC code, so sector grouping for
calibration comes free here rather than needing a separate lookup pass.

Run directly to build the candidate set:

    python -m synergy_estimator.corpus.discover --start 2015-01-01 --out data/candidates.json
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import requests

_FTS_URL = "https://efts.sec.gov/LATEST/search-index"
# EDGAR returns 100 hits per page. Advancing `from` by any smaller step
# re-reads an overlapping window and inflates the result count several-fold.
_PAGE_SIZE = 100
# EDGAR full-text search refuses to paginate past 10,000 results.
_MAX_FROM = 9990
# SEC fair-access allows 10 requests/second; stay comfortably under it.
_REQUEST_INTERVAL_SECONDS = 0.15

# Phrases acquirers actually use when quantifying a synergy target. Deliberately
# overlapping -- the same filing surfacing under several phrases is deduped by
# accession number, and recall matters more than query count here.
DISCLOSURE_PHRASES: tuple[str, ...] = (
    '"run-rate cost synergies"',
    '"annual cost synergies"',
    '"annualized cost synergies"',
    '"cost synergies of approximately"',
    '"in cost synergies"',
    '"run-rate synergies"',
    '"synergies of approximately"',
    '"expected cost synergies"',
    '"estimated cost synergies"',
)
# Note: EDGAR tokenizes on hyphens, so "run-rate" and "run rate" return
# identical result sets -- punctuation variants are not worth a query.

# 8-K/EX-99 carries the announcement press release; 425 is merger communications;
# S-4 and DEFM14A are the registration/proxy documents.
DEAL_FORMS: tuple[str, ...] = ("8-K", "425", "S-4", "DEFM14A")


@dataclass(frozen=True)
class FilingHit:
    """One full-text search hit, keyed by the filing it belongs to."""

    accession: str
    document: str  # primary document filename within the filing
    form: str
    filing_date: str  # YYYY-MM-DD
    ciks: tuple[str, ...]  # all filers on the document -- often both deal parties
    display_names: tuple[str, ...]  # e.g. "3D SYSTEMS CORP  (DDD)  (CIK 0000910638)"
    sics: tuple[str, ...]  # filer SIC codes, used for sector grouping in calibration
    matched_phrase: str

    @property
    def document_url(self) -> str:
        stripped = self.accession.replace("-", "")
        cik = self.ciks[0].lstrip("0") if self.ciks else ""
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{stripped}/{self.document}"


class FullTextSearchError(RuntimeError):
    pass


def _session(identity: str) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = identity
    return session


def _get(session: requests.Session, params: dict, attempts: int = 6) -> dict:
    """One search request, retrying transient backend failures.

    EDGAR's search gateway reports upstream OpenSearch outages as **HTTP 200
    with an error body**, so a status check alone silently yields zero results.
    Detect on the payload instead.
    """
    last_error = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            response = session.get(_FTS_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            last_error = str(exc)
            continue
        try:
            payload = response.json()
        except ValueError:
            last_error = f"non-JSON response (HTTP {response.status_code})"
            continue
        if "hits" in payload:
            return payload
        last_error = payload.get("errorType", f"HTTP {response.status_code}")
    raise FullTextSearchError(f"EDGAR full-text search failed after {attempts} attempts: {last_error}")


def search(
    session: requests.Session,
    phrase: str,
    forms: str,
    start: str | None = None,
    end: str | None = None,
    max_results: int = 1000,
) -> list[FilingHit]:
    """Page through full-text search results for one phrase/form combination."""
    hits: list[FilingHit] = []
    offset = 0
    while offset < min(max_results, _MAX_FROM + _PAGE_SIZE):
        params = {"q": phrase, "forms": forms}
        # EDGAR silently ignores startdt unless enddt accompanies it, so a
        # lone start date returns the full 2001-present history rather than
        # the requested window. Always send the pair.
        if start or end:
            params["startdt"] = start or "2001-01-01"
            params["enddt"] = end or date.today().isoformat()
        if offset:
            params["from"] = offset

        payload = _get(session, params)
        page = payload["hits"]["hits"]
        if not page:
            break

        for hit in page:
            accession, _, document = hit["_id"].partition(":")
            source = hit["_source"]
            hits.append(
                FilingHit(
                    accession=accession,
                    document=document,
                    form=source.get("form", ""),
                    filing_date=source.get("file_date", ""),
                    ciks=tuple(source.get("ciks", ())),
                    display_names=tuple(source.get("display_names", ())),
                    sics=tuple(source.get("sics", ())),
                    matched_phrase=phrase,
                )
            )

        total = payload["hits"]["total"]["value"]
        offset += _PAGE_SIZE
        if offset >= total:
            break
        time.sleep(_REQUEST_INTERVAL_SECONDS)

    return hits


def discover(
    identity: str,
    phrases: tuple[str, ...] = DISCLOSURE_PHRASES,
    forms: tuple[str, ...] = DEAL_FORMS,
    start: str | None = "2015-01-01",
    end: str | None = None,
    max_results_per_query: int = 1000,
    progress=None,
) -> list[FilingHit]:
    """Search every phrase x form combination and dedupe to one hit per filing.

    A filing matching several phrases is kept once, under the first phrase that
    found it -- the phrase is recorded only for provenance, not as a signal.
    """
    session = _session(identity)
    by_accession: dict[str, FilingHit] = {}
    failures: list[str] = []
    for phrase in phrases:
        for form in forms:
            try:
                found = search(session, phrase, form, start, end, max_results_per_query)
            except FullTextSearchError as exc:
                # EDGAR's search backend has intermittent outages. One bad
                # query must not discard the other 39 queries' worth of work.
                failures.append(f"{phrase} [{form}]: {exc}")
                if progress:
                    progress(f"{phrase} [{form}]: FAILED, continuing -- {exc}")
                continue

            new = 0
            for hit in found:
                if hit.accession not in by_accession:
                    by_accession[hit.accession] = hit
                    new += 1
            if progress:
                progress(f"{phrase} [{form}]: {len(found)} docs, {new} new filings "
                         f"({len(by_accession)} unique so far)")
            time.sleep(_REQUEST_INTERVAL_SECONDS)

    if failures and progress:
        progress(f"WARNING: {len(failures)} of {len(phrases) * len(forms)} queries failed; "
                 "coverage is incomplete. Re-run to fill the gaps.")
    return sorted(by_accession.values(), key=lambda h: h.filing_date)


def save(hits: list[FilingHit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(h) for h in hits], indent=2))


def load(path: Path) -> list[FilingHit]:
    return [FilingHit(**{**h, "ciks": tuple(h["ciks"]), "display_names": tuple(h["display_names"]),
                         "sics": tuple(h["sics"])})
            for h in json.loads(Path(path).read_text())]


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data/candidates.json", type=Path)
    parser.add_argument("--max-per-query", type=int, default=1000)
    args = parser.parse_args()

    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise SystemExit("Set EDGAR_IDENTITY='Your Name your@email.com' (SEC fair-access policy).")

    found = discover(
        identity,
        start=args.start,
        end=args.end,
        max_results_per_query=args.max_per_query,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    save(found, args.out)
    print(f"\n{len(found)} unique filings -> {args.out}")
