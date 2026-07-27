import pandas as pd

from synergy_estimator.data.edgar_source import (
    _fiscal_year_columns,
    _first_available_value,
    _newest_reported_before,
    _resolve_total_liabilities,
)
from synergy_estimator.data.schema import AnnualFinancials


def _sample_df():
    return pd.DataFrame(
        [
            {"concept": "us-gaap_Revenues", "dimension": False, "2024-12-31 (FY)": 100.0},
            {"concept": "us-gaap_Revenues", "dimension": True, "2024-12-31 (FY)": 40.0},
            {"concept": "us-gaap_CostOfRevenue", "dimension": False, "2024-12-31 (FY)": 60.0},
        ]
    )


def test_first_available_value_skips_dimensional_rows_and_falls_back():
    df = _sample_df()
    assert _first_available_value(df, ["us-gaap_Revenues"], "2024-12-31 (FY)") == 100.0
    assert (
        _first_available_value(
            df, ["us-gaap_CostOfGoodsAndServicesSold", "us-gaap_CostOfRevenue"], "2024-12-31 (FY)"
        )
        == 60.0
    )
    assert _first_available_value(df, ["us-gaap_DoesNotExist"], "2024-12-31 (FY)") is None


def test_fiscal_year_columns_matches_only_fy_period_headers():
    df = pd.DataFrame(columns=["concept", "2024-12-31 (FY)", "2024-12-31 (Q4)", "label"])
    assert _fiscal_year_columns(df) == ["2024-12-31 (FY)"]


def test_resolve_total_liabilities_prefers_direct_concept():
    df = pd.DataFrame(
        [
            {"concept": "us-gaap_Liabilities", "dimension": False, "2024-12-31": 60.0},
            {"concept": "us-gaap_Assets", "dimension": False, "2024-12-31": 100.0},
            {"concept": "us-gaap_StockholdersEquity", "dimension": False, "2024-12-31": 40.0},
        ]
    )
    assert _resolve_total_liabilities(df, "2024-12-31") == 60.0


def _fy(period_end):
    return AnnualFinancials(
        ticker="ADI",
        fiscal_year=int(period_end[:4]),
        period_end=period_end,
        revenue=None,
        cogs=None,
        gross_profit=None,
        rnd_expense=None,
        sga_expense=None,
        operating_income=None,
        net_income=None,
        shares_diluted=None,
        pretax_income=None,
        income_tax_expense=None,
    )


def test_newest_reported_before_picks_the_last_closed_fiscal_year():
    # ADI's FY2019 ended 2019-11-02; the Maxim deal was announced 2020-07-13.
    candidates = [_fy("2019-11-02"), _fy("2018-11-03"), _fy("2017-10-28")]
    assert _newest_reported_before(candidates, "2020-07-13").fiscal_year == 2019


def test_newest_reported_before_excludes_fiscal_years_ending_after_announcement():
    # A fiscal year that closed after the announcement was not knowable to an
    # analyst underwriting the deal -- this exclusion is the bug fix.
    candidates = [_fy("2020-10-31"), _fy("2019-11-02")]
    assert _newest_reported_before(candidates, "2020-07-13").fiscal_year == 2019


def test_newest_reported_before_treats_the_announcement_date_as_exclusive():
    assert _newest_reported_before([_fy("2020-07-13")], "2020-07-13") is None


def test_newest_reported_before_returns_none_when_nothing_qualifies():
    # e.g. a target that IPO'd months before being acquired.
    assert _newest_reported_before([_fy("2024-12-31")], "2024-01-01") is None


def test_resolve_total_liabilities_falls_back_to_assets_minus_equity():
    # No standalone Liabilities line (e.g. Republic Services only tags Assets
    # and StockholdersEquity) -- should derive it via the accounting identity.
    df = pd.DataFrame(
        [
            {"concept": "us-gaap_Assets", "dimension": False, "2024-12-31": 100.0},
            {"concept": "us-gaap_StockholdersEquity", "dimension": False, "2024-12-31": 40.0},
        ]
    )
    assert _resolve_total_liabilities(df, "2024-12-31") == 60.0
