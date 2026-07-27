"""Identify who is buying whom, from EDGAR's filing header.

Merger filings carry the answer structurally. The SGML header of a 425, S-4 or
DEFM14A declares:

    SUBJECT COMPANY:                     <- the target
       COMPANY CONFORMED NAME:  IAA, Inc.
       CENTRAL INDEX KEY:       0001745041
       STANDARD INDUSTRIAL CLASSIFICATION: ... [5500]
    FILED BY:                            <- the acquirer
       COMPANY CONFORMED NAME:  RITCHIE BROS AUCTIONEERS INC
       CENTRAL INDEX KEY:       0001046102
       STANDARD INDUSTRIAL CLASSIFICATION: ... [7389]

That is exact, free, and more reliable than reading names out of prose -- and
it supplies each side's SIC separately. The target's SIC is the one that
matters most for calibration, since the acquired cost base is what a merger
absorbs; grouping on the filer's code alone would file every deal under the
acquirer's sector.

Filings with no SUBJECT COMPANY (most plain 8-Ks, which the acquirer files
alone) cannot be resolved this way and are dropped -- with no identified
target there are no target financials to fetch.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

_HEADER_CACHE = Path("data/cache/headers")
_SEC_RATE_LIMIT_SECONDS = 0.15

_BLOCK_RE = re.compile(r"^\s*(SUBJECT COMPANY|FILED BY|FILER|REPORTING-OWNER)\s*:", re.IGNORECASE)
_NAME_RE = re.compile(r"COMPANY CONFORMED NAME:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CIK_RE = re.compile(r"CENTRAL INDEX KEY:\s*(\d+)", re.IGNORECASE)
_SIC_RE = re.compile(r"STANDARD INDUSTRIAL CLASSIFICATION:.*?\[(\d{3,4})\]", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Party:
    name: str
    cik: int
    sic: str | None


@dataclass(frozen=True)
class DealParties:
    acquirer: Party | None  # FILED BY
    target: Party | None  # SUBJECT COMPANY

    @property
    def is_resolved(self) -> bool:
        """Both sides identified, and not the same company filing about itself."""
        return (
            self.acquirer is not None
            and self.target is not None
            and self.acquirer.cik != self.target.cik
        )


# The full-submission .txt begins with the SGML header, and exists for every
# filing era. The tidier `-index-headers.html` is only generated for roughly
# 2012-onward filings, so relying on it silently drops the older half of the
# corpus. Only the head of the .txt is read -- full submissions run to
# megabytes and everything needed is in the first few KB.
_HEADER_BYTES = 65_536


def header_urls(accession: str, cik: str | int) -> list[str]:
    """Candidate URLs for a filing's SGML header, newest layout first."""
    stripped = accession.replace("-", "")
    normalized = str(cik).lstrip("0")
    base = f"https://www.sec.gov/Archives/edgar/data/{normalized}"
    return [
        f"{base}/{stripped}/{accession}.txt",  # modern nested layout
        f"{base}/{accession}.txt",  # legacy flat layout
    ]


def parse_header(text: str) -> DealParties:
    """Read the FILED BY / SUBJECT COMPANY blocks out of a filing header."""
    plain = _TAG_RE.sub("", text)

    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in plain.splitlines():
        match = _BLOCK_RE.match(line)
        if match:
            current = match.group(1).upper()
            blocks.setdefault(current, [])
            continue
        if current:
            blocks[current].append(line)

    def party_from(label: str) -> Party | None:
        lines = blocks.get(label)
        if not lines:
            return None
        # Only scan until the next party's fields would begin; the first
        # name/CIK after the label belongs to that party.
        chunk = "\n".join(lines[:25])
        name = _NAME_RE.search(chunk)
        cik = _CIK_RE.search(chunk)
        if not name or not cik:
            return None
        sic = _SIC_RE.search(chunk)
        return Party(name=name.group(1).strip(), cik=int(cik.group(1)), sic=sic.group(1) if sic else None)

    # A lone "FILER" (no SUBJECT COMPANY) is a self-filed 8-K: the filer is the
    # acquirer and the target is simply not declared.
    acquirer = party_from("FILED BY") or party_from("FILER")
    return DealParties(acquirer=acquirer, target=party_from("SUBJECT COMPANY"))


def fetch_parties(session: requests.Session, accession: str, cik: str | int) -> DealParties:
    """Fetch and parse one filing's SGML header, caching the head on disk."""
    cache_path = _HEADER_CACHE / f"{accession}.txt"
    if cache_path.exists():
        return parse_header(cache_path.read_text())

    head = None
    for url in header_urls(accession, cik):
        response = session.get(url, timeout=30, stream=True)
        if response.status_code != 200:
            response.close()
            continue
        # Read only the head: the SGML header is at the top and full
        # submissions can run to megabytes.
        head = response.raw.read(_HEADER_BYTES, decode_content=True).decode("utf-8", "replace")
        response.close()
        break

    time.sleep(_SEC_RATE_LIMIT_SECONDS)
    if head is None:
        raise requests.HTTPError(f"no SGML header found for {accession}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(head)
    return parse_header(head)
