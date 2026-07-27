from synergy_estimator.corpus.parties import header_urls, parse_header

# Shape taken verbatim from a real 425 header (Ritchie Bros / IAA).
_MERGER_HEADER = """
<html><pre>
SUBJECT COMPANY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			IAA, Inc.
		CENTRAL INDEX KEY:			0001745041
		STANDARD INDUSTRIAL CLASSIFICATION:	RETAIL-AUTO DEALERS &amp; GASOLINE STATIONS [5500]

FILED BY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			RITCHIE BROS AUCTIONEERS INC
		CENTRAL INDEX KEY:			0001046102
		STANDARD INDUSTRIAL CLASSIFICATION:	SERVICES-BUSINESS SERVICES, NEC [7389]
</pre></html>
"""

_SELF_FILED_8K = """
<html><pre>
FILER:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			HEWLETT PACKARD ENTERPRISE CO
		CENTRAL INDEX KEY:			0001645590
		STANDARD INDUSTRIAL CLASSIFICATION:	COMPUTER HARDWARE [3570]
</pre></html>
"""


def test_reads_target_from_subject_company():
    parties = parse_header(_MERGER_HEADER)

    assert parties.target.name == "IAA, Inc."
    assert parties.target.cik == 1745041
    assert parties.target.sic == "5500"


def test_reads_acquirer_from_filed_by():
    parties = parse_header(_MERGER_HEADER)

    assert parties.acquirer.name == "RITCHIE BROS AUCTIONEERS INC"
    assert parties.acquirer.cik == 1046102
    assert parties.acquirer.sic == "7389"


def test_merger_filing_is_resolved():
    assert parse_header(_MERGER_HEADER).is_resolved


def test_self_filed_8k_has_no_target_and_is_unresolved():
    """Most plain 8-Ks are filed by the acquirer alone. Without a declared
    target there are no target financials to fetch, so the deal is dropped."""
    parties = parse_header(_SELF_FILED_8K)

    assert parties.acquirer.cik == 1645590
    assert parties.target is None
    assert not parties.is_resolved


def test_company_filing_about_itself_is_not_a_deal():
    same = _MERGER_HEADER.replace("0001745041", "0001046102")

    assert not parse_header(same).is_resolved


def test_missing_sic_is_tolerated():
    without_sic = "\n".join(
        line for line in _MERGER_HEADER.splitlines() if "STANDARD INDUSTRIAL" not in line
    )

    parties = parse_header(without_sic)

    assert parties.target.cik == 1745041
    assert parties.target.sic is None


def test_empty_header_yields_nothing():
    parties = parse_header("<html></html>")

    assert parties.acquirer is None
    assert not parties.is_resolved


def test_header_urls_cover_both_archive_layouts():
    """The nested layout is only generated for roughly 2012-onward filings;
    without the legacy flat fallback the older half of the corpus is lost."""
    modern, legacy = header_urls("0001104659-23-005579", "0001046102")

    assert modern == (
        "https://www.sec.gov/Archives/edgar/data/1046102/000110465923005579/"
        "0001104659-23-005579.txt"
    )
    assert legacy == (
        "https://www.sec.gov/Archives/edgar/data/1046102/0001104659-23-005579.txt"
    )


def test_parses_the_sgml_header_of_a_full_submission():
    """Real .txt submissions wrap the header in <SEC-HEADER> and are followed
    by document bodies; parsing must not depend on the HTML index shape."""
    submission = """<SEC-DOCUMENT>0001104659-23-005579.txt : 20230118
<SEC-HEADER>0001104659-23-005579.hdr.sgml : 20230118
ACCESSION NUMBER:		0001104659-23-005579
CONFORMED SUBMISSION TYPE:	425
SUBJECT COMPANY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			IAA, Inc.
		CENTRAL INDEX KEY:			0001745041
		STANDARD INDUSTRIAL CLASSIFICATION:	RETAIL-AUTO DEALERS [5500]

FILED BY:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			RITCHIE BROS AUCTIONEERS INC
		CENTRAL INDEX KEY:			0001046102
		STANDARD INDUSTRIAL CLASSIFICATION:	SERVICES-BUSINESS SERVICES [7389]
</SEC-HEADER>
<DOCUMENT>irrelevant body text</DOCUMENT>
"""

    parties = parse_header(submission)

    assert parties.is_resolved
    assert parties.target.cik == 1745041
    assert parties.acquirer.cik == 1046102
