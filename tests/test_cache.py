import pytest

from synergy_estimator.data import cache as cache_module
from synergy_estimator.data.cache import CachedFailure, cached
from synergy_estimator.data.schema import AnnualFinancials


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_DIR", tmp_path)


def _financials(ticker="ADI", period_end="2019-11-02", sga=648.0):
    return AnnualFinancials(
        ticker=ticker,
        fiscal_year=int(period_end[:4]),
        period_end=period_end,
        revenue=5_991.0,
        cogs=1_977.0,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=sga,
        operating_income=None,
        net_income=None,
        shares_diluted=None,
        pretax_income=None,
        income_tax_expense=None,
    )


def test_second_call_reads_from_disk_instead_of_refetching():
    calls = []

    def produce():
        calls.append(1)
        return _financials()

    first = cached("financials_before", "ADI_2020-07-13", AnnualFinancials, produce)
    second = cached("financials_before", "ADI_2020-07-13", AnnualFinancials, produce)

    assert len(calls) == 1
    assert second == first
    assert second.sga_expense == pytest.approx(648.0)


def test_distinct_keys_do_not_collide():
    cached("financials_before", "ADI_2020-07-13", AnnualFinancials, lambda: _financials(sga=648.0))
    other = cached("financials_before", "ADI_2025-01-01", AnnualFinancials, lambda: _financials(sga=1_255.0))

    assert other.sga_expense == pytest.approx(1_255.0)


def test_round_trips_a_list_of_dataclasses():
    produced = [_financials(period_end="2019-11-02"), _financials(period_end="2018-11-03")]
    cached("annual", "ADI_2", AnnualFinancials, lambda: produced)

    restored = cached("annual", "ADI_2", AnnualFinancials, lambda: [])

    assert restored == produced


def test_negative_results_are_cached_and_replayed():
    """At corpus scale, rediscovering 'this filer has no pre-deal 10-K' costs a
    full filings index load per deal per run."""
    calls = []

    def produce():
        calls.append(1)
        raise ValueError("no 10-K filed before 2024-01-01")

    with pytest.raises(ValueError):
        cached("financials_before", "NEWCO_2024-01-01", AnnualFinancials, produce)

    with pytest.raises(CachedFailure, match="no 10-K filed before"):
        cached("financials_before", "NEWCO_2024-01-01", AnnualFinancials, produce)

    assert len(calls) == 1


def test_keys_with_path_separators_do_not_escape_the_cache_directory(tmp_path):
    cached("financials_before", "../../etc/passwd", AnnualFinancials, lambda: _financials())

    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1
    assert tmp_path in written[0].parents


def test_tolerates_a_field_added_since_the_entry_was_written(tmp_path):
    """A cache written by an older schema should degrade to None on new fields
    rather than blowing up an otherwise-valid hit."""
    stale = tmp_path / "annual" / "OLD.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"ok": true, "data": {"ticker": "OLD", "fiscal_year": 2019, "period_end": "2019-11-02"}}')

    restored = cached("annual", "OLD", AnnualFinancials, lambda: _financials())

    assert restored.ticker == "OLD"
    assert restored.sga_expense is None
