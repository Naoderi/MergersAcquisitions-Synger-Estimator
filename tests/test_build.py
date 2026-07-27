import pytest
from synergy_estimator.corpus.build import CorpusRow, deduplicate


def _row(deal_id, acq, tgt, date, synergy=100e6):
    return CorpusRow(
        deal_id=deal_id, announcement_date=date,
        acquirer_name=f"A{acq}", acquirer_cik=acq, acquirer_sic="7372",
        target_name=f"T{tgt}", target_cik=tgt, target_sic="3674",
        sector="software_services", disclosed_synergy_usd=synergy, synergy_type="cost",
        acquirer_sga=1.0, acquirer_cogs=1.0, target_sga=1.0, target_cogs=1.0,
        combined_sga=2.0, combined_cogs=2.0,
        acquirer_revenue=10.0, target_revenue=10.0, combined_revenue=20.0,
        acquirer_fy=2023, target_fy=2023, source_quote="q", source_url="u",
    )


def test_collapses_the_same_deal_filed_by_both_sides():
    """Getty/Shutterstock appeared twice with the parties swapped."""
    rows = [_row("a", 111, 222, "2025-01-07"), _row("b", 222, 111, "2025-01-07")]

    assert len(deduplicate(rows)) == 1


def test_keeps_the_earliest_as_announced_figure():
    rows = [_row("late", 111, 222, "2025-03-01"), _row("early", 111, 222, "2025-01-07")]

    (kept,) = deduplicate(rows)

    assert kept.deal_id == "early"


def test_keeps_separate_deals_between_the_same_parties_years_apart():
    rows = [_row("first", 111, 222, "2015-01-07"), _row("second", 111, 222, "2023-01-07")]

    assert len(deduplicate(rows)) == 2


def test_leaves_unrelated_deals_alone():
    rows = [_row("a", 111, 222, "2024-01-01"), _row("b", 333, 444, "2024-01-01")]

    assert len(deduplicate(rows)) == 2
