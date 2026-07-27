from synergy_estimator.corpus.discover import FilingHit
from synergy_estimator.corpus.group import group_into_deals


def _hit(accession, filing_date, ciks=("0000123",), form="8-K", sics=("7372",)):
    return FilingHit(
        accession=accession,
        document="doc.htm",
        form=form,
        filing_date=filing_date,
        ciks=tuple(ciks),
        display_names=(),
        sics=tuple(sics),
        matched_phrase='"cost synergies"',
    )


def test_collapses_one_deals_filing_chain_into_a_single_candidate():
    """The 8-K, 425 and S-4 for one transaction all restate the same target."""
    hits = [
        _hit("a", "2024-01-09", form="8-K"),
        _hit("b", "2024-01-15", form="425"),
        _hit("c", "2024-03-01", form="S-4"),
    ]

    (deal,) = group_into_deals(hits)

    assert deal.filing_count == 3
    assert deal.primary.accession == "a"  # earliest = closest to as-announced


def test_separates_a_serial_acquirers_distinct_deals():
    """Grouping on the CIK set alone would merge every acquisition a serial
    acquirer ever made into one deal."""
    hits = [_hit("a", "2018-01-09"), _hit("b", "2018-02-01"), _hit("c", "2023-06-01")]

    deals = group_into_deals(hits)

    assert len(deals) == 2
    assert [d.primary.accession for d in deals] == ["a", "c"]


def test_does_not_split_a_long_running_regulatory_review():
    hits = [_hit("a", "2024-01-09"), _hit("b", "2024-05-01")]

    assert len(group_into_deals(hits)) == 1


def test_merges_an_acquirer_only_8k_with_the_same_deals_425():
    """One transaction is announced by an acquirer-only 8-K and a both-parties
    425 on the same day. Matching on the exact CIK set files those as two
    deals, and the 8-K-only one can never be resolved to a target."""
    hits = [
        _hit("a", "2024-01-09", ciks=("111",), form="8-K"),
        _hit("b", "2024-01-09", ciks=("111", "222"), form="425"),
    ]

    (deal,) = group_into_deals(hits)

    assert deal.filing_count == 2
    assert deal.ciks == frozenset({"111", "222"})


def test_unrelated_parties_stay_separate():
    hits = [_hit("a", "2024-01-09", ciks=("111", "222")), _hit("b", "2024-01-10", ciks=("333", "444"))]

    assert len(group_into_deals(hits)) == 2


def test_shared_acquirer_far_apart_in_time_stays_separate():
    """A serial acquirer shares its own CIK across every deal it does; only
    the time window separates them."""
    hits = [
        _hit("a", "2018-01-09", ciks=("111", "222")),
        _hit("b", "2023-06-01", ciks=("111", "333")),
    ]

    assert len(group_into_deals(hits)) == 2


def test_collects_sic_codes_across_the_whole_chain():
    """The announcement 8-K often lists only the acquirer; the S-4 carries both
    parties, and the target's SIC is what sector grouping needs."""
    hits = [
        _hit("a", "2024-01-09", ciks=("111", "222"), sics=("7372",)),
        _hit("b", "2024-02-09", ciks=("111", "222"), sics=("7372", "3674")),
    ]

    (deal,) = group_into_deals(hits)

    assert deal.sics == ("7372", "3674")


def test_ignores_filings_with_no_date():
    assert group_into_deals([_hit("a", "")]) == []
