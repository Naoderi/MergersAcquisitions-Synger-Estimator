import pytest
import requests

from synergy_estimator.corpus.discover import (
    _PAGE_SIZE,
    FilingHit,
    FullTextSearchError,
    _get,
    discover,
    search,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Stands in for requests.Session, replaying a queued list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _page(ids, total, sic="7372", cik="0000910638"):
    return FakeResponse(
        {
            "hits": {
                "total": {"value": total},
                "hits": [
                    {
                        "_id": f"{accession}:doc.htm",
                        "_source": {
                            "form": "8-K",
                            "file_date": "2023-07-13",
                            "ciks": [cik],
                            "display_names": ["3D SYSTEMS CORP  (DDD)  (CIK 0000910638)"],
                            "sics": [sic],
                        },
                    }
                    for accession in ids
                ],
            }
        }
    )


def test_get_retries_when_sec_returns_an_error_body_with_http_200(monkeypatch):
    """EDGAR reports upstream outages as HTTP 200 with an error payload, so a
    status-only check silently yields zero results."""
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([FakeResponse({"errorType": "ConnectionError"}), _page(["0001-23-1"], 1)])

    payload = _get(session, {"q": "x"})

    assert payload["hits"]["total"]["value"] == 1
    assert len(session.calls) == 2


def test_get_retries_transient_network_errors(monkeypatch):
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([requests.ConnectionError("reset"), _page(["0001-23-1"], 1)])

    assert _get(session, {"q": "x"})["hits"]["total"]["value"] == 1


def test_get_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([FakeResponse({"errorType": "ConnectionError"})] * 6)

    with pytest.raises(FullTextSearchError, match="ConnectionError"):
        _get(session, {"q": "x"}, attempts=6)


def test_search_advances_from_by_the_actual_page_size(monkeypatch):
    """EDGAR returns 100 hits per page. Stepping `from` by any smaller amount
    re-reads an overlapping window and inflates the result count several-fold."""
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    total = _PAGE_SIZE + 5
    session = FakeSession(
        [
            _page([f"0001-23-{i}" for i in range(_PAGE_SIZE)], total),
            _page([f"0001-23-{i}" for i in range(_PAGE_SIZE, total)], total),
        ]
    )

    hits = search(session, '"cost synergies"', "8-K")

    assert len(hits) == total
    assert len({h.accession for h in hits}) == total  # no window overlap
    assert session.calls[1]["from"] == _PAGE_SIZE


def test_search_stops_on_an_empty_page(monkeypatch):
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([_page(["0001-23-1"], 999), _page([], 999)])

    assert len(search(session, '"cost synergies"', "8-K")) == 1


def test_search_parses_accession_cik_and_sic_from_the_hit(monkeypatch):
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([_page(["0001193125-23-186316"], 1, sic="7372")])

    (hit,) = search(session, '"cost synergies"', "8-K")

    assert hit.accession == "0001193125-23-186316"
    assert hit.document == "doc.htm"
    assert hit.sics == ("7372",)
    assert hit.filing_date == "2023-07-13"


def test_discover_dedupes_filings_found_by_multiple_phrases(monkeypatch):
    """Phrases deliberately overlap for recall; the same filing must be kept once."""
    monkeypatch.setattr("synergy_estimator.corpus.discover.time.sleep", lambda _: None)
    session = FakeSession([_page(["0001-23-1"], 1), _page(["0001-23-1"], 1)])
    monkeypatch.setattr("synergy_estimator.corpus.discover._session", lambda identity: session)

    hits = discover("me me@example.com", phrases=('"a"', '"b"'), forms=("8-K",), start=None)

    assert len(hits) == 1


def test_document_url_builds_an_archives_path():
    hit = FilingHit(
        accession="0001193125-23-186316",
        document="d526113dex992.htm",
        form="8-K",
        filing_date="2023-07-13",
        ciks=("0000910638",),
        display_names=(),
        sics=("7372",),
        matched_phrase='"x"',
    )

    assert hit.document_url == (
        "https://www.sec.gov/Archives/edgar/data/910638/000119312523186316/d526113dex992.htm"
    )
