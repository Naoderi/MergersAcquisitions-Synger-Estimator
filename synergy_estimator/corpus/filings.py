"""Fetch filing documents from EDGAR and reduce them to plain text.

Filings are HTML (occasionally XML), often with tables and inline markup
splitting sentences apart. Collapsing to whitespace-normalized text is what
lets the disclosure patterns in `patterns.py` match across a phrase that the
source document happened to wrap in a `<font>` tag mid-sentence.

Documents are cached on disk. A corpus run reads several filings per deal
across a few thousand deals, and SEC's fair-access policy caps request rate.
"""

import re
import time
from pathlib import Path

import requests

from synergy_estimator.corpus.discover import FilingHit

_FILING_CACHE = Path("data/cache/filings")
_SEC_RATE_LIMIT_SECONDS = 0.15


def clean_text(html: str) -> str:
    """Strip markup and collapse whitespace."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    # Some EDGAR exhibits are XML. The HTML parser still extracts their text
    # correctly, which is all this needs.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def fetch_filing_text(session: requests.Session, hit: FilingHit) -> str:
    """Download and de-HTML one filing document, caching the plain text."""
    cache_path = _FILING_CACHE / f"{hit.accession}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    response = session.get(hit.document_url, timeout=60)
    response.raise_for_status()
    text = clean_text(response.text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    time.sleep(_SEC_RATE_LIMIT_SECONDS)
    return text
