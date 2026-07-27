"""Curated set of real M&A deals with publicly disclosed synergy targets, used to
backtest estimate_synergies() against reality.

Deliberately excludes Kroger/Albertsons and Tapestry/Capri (synergies.py) -- those
two calibrated the model's overlap-rate constants, so reusing them here would be
circular.

Sector coverage note: candidates were also researched in airlines (JetBlue/Spirit
Airlines) and upstream/integrated oil & gas (ConocoPhillips/Marathon Oil,
Chevron/Hess, ExxonMobil/Pioneer Natural Resources) but dropped after fetching
their live financials -- those sectors report costs by nature (fuel, labor,
production costs, DD&A) rather than by function (COGS/SG&A), so they have no
COGS concept at all, the same underlying reason banks are out of scope. This
isn't a data-fetch gap; the model's cost/revenue-synergy framing genuinely
doesn't apply to them.
"""

from dataclasses import dataclass


@dataclass
class HistoricalDeal:
    acquirer_ticker: str
    target_ticker: str
    target_cik: int | None  # set only for delisted targets; None = live ticker lookup
    disclosed_synergy_runrate: float  # annual pretax $, as publicly disclosed
    synergy_type: str  # "cost" or "cost_and_revenue" -- which estimate to compare against
    deal_status: str  # "blocked" | "abandoned" | "completed"
    announcement_date: str
    source: str  # citation: press release / 8-K / investor deck
    disclosed_deal_value: float | None = None  # optional, for "% of deal value" context
    notes: str | None = None
    # Set only when the acquirer's ticker no longer resolves to the entity that
    # announced the deal -- e.g. a merger-of-equals holdco reorg, where the
    # ticker now points at a new CIK with no pre-announcement filing history.
    acquirer_cik: int | None = None


DEALS: list[HistoricalDeal] = [
    HistoricalDeal(
        acquirer_ticker="RSG",
        target_ticker="ECOL",
        target_cik=1783400,  # US Ecology, Inc. (post-2019 holdco reorg CIK)
        disclosed_synergy_runrate=40_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2022-02-09",
        source=(
            "Republic Services press release, Feb 9, 2022: $40M cost synergies within 3 years -- "
            "https://www.prnewswire.com/news-releases/republic-services-to-acquire-us-ecology-a-leading-environmental-solutions-company-301478661.html"
        ),
        disclosed_deal_value=2_200_000_000.0,
        notes="Also disclosed $75-100M cross-sell revenue opportunity over 3 years; not included here (cost-only comparison).",
    ),
    HistoricalDeal(
        acquirer_ticker="HPE",
        target_ticker="JNPR",
        target_cik=1043604,  # Juniper Networks Inc
        disclosed_synergy_runrate=450_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2024-01-09",
        source=(
            "HPE press release, Jan 9, 2024: $450M run-rate cost synergies within 36 months -- "
            "https://www.hpe.com/us/en/newsroom/press-release/2024/01/hpe-to-acquire-juniper-networks-to-accelerate-ai-driven-innovation.html"
        ),
        notes="Post-close guidance later raised to $600M+ by FY2028; using original as-announced figure for a pre-deal-financials comparison.",
    ),
    HistoricalDeal(
        acquirer_ticker="SNPS",
        target_ticker="ANSS",
        target_cik=1013462,  # Ansys Inc
        disclosed_synergy_runrate=400_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2024-01-16",
        source=(
            "Synopsys press release, Jan 16, 2024: ~$400M run-rate cost synergies by year 3 -- "
            "https://investor.synopsys.com/news/news-details/2024/Synopsys-to-Acquire-Ansys-Creating-a-Leader-in-Silicon-to-Systems-Design-Solutions/default.aspx"
        ),
        notes="Also disclosed ~$400M revenue synergies by year 4; using cost-only figure.",
    ),
    HistoricalDeal(
        acquirer_ticker="NE",
        target_ticker="DO",
        target_cik=949039,  # Diamond Offshore Drilling Inc
        disclosed_synergy_runrate=100_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2024-06-10",
        source=(
            "Noble Corporation plc press release, June 2024: $100M annual pretax cost synergies, "
            "75% within one year -- "
            "https://www.prnewswire.com/news-releases/noble-corporation-plc-announces-agreement-to-acquire-diamond-offshore-drilling-inc-302167958.html"
        ),
    ),
    HistoricalDeal(
        acquirer_ticker="FUN",
        target_ticker="SIX",
        target_cik=701374,  # Six Flags Entertainment Corp/OLD
        disclosed_synergy_runrate=120_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2023-11-02",
        source=(
            "Cedar Fair/Six Flags press release, Nov 2, 2023: $200M total synergies "
            "($120M cost + $80M revenue) -- "
            "https://www.businesswire.com/news/home/20231102695482/en/Cedar-Fair-and-Six-Flags-to-Combine-in-Merger-of-Equals-Creating-a-Leading-Amusement-Park-Operator"
        ),
        notes=(
            "Merger of equals: Cedar Fair was the legal acquirer/survivor and renamed itself "
            "Six Flags Entertainment Corp (ticker FUN). Using the cost-only portion of the "
            "disclosed $200M total synergies."
        ),
        # FUN now resolves to "Six Flags Entertainment Corporation/NEW" (CIK
        # 1999001), a holdco created for the merger whose first 10-K postdates
        # the announcement. Pin to Cedar Fair L.P., the entity that announced.
        acquirer_cik=811532,
    ),
    HistoricalDeal(
        acquirer_ticker="CPB",
        target_ticker="SOVO",
        target_cik=1856608,  # Sovos Brands, Inc.
        disclosed_synergy_runrate=50_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2023-08-06",
        source=(
            "Campbell's press release, Aug 8, 2023: ~$50M annualized cost synergies over 2 years -- "
            "https://www.thecampbellscompany.com/newsroom/press-releases/campbell-to-acquire-sovos-brands-leader-in-high-growth-premium-italian-sauces/"
        ),
    ),
    HistoricalDeal(
        acquirer_ticker="ADI",
        target_ticker="MXIM",
        target_cik=743316,  # Maxim Integrated Products Inc
        disclosed_synergy_runrate=275_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2020-07-13",
        source=(
            "Analog Devices investor presentation, July 2020: $275M cost synergies by end of year 2 -- "
            "https://investor.analog.com/static-files/52340411-401f-4554-ad4e-a7b4299501b9"
        ),
    ),
    HistoricalDeal(
        acquirer_ticker="SJM",
        target_ticker="TWNK",
        target_cik=1644406,  # Hostess Brands, Inc.
        disclosed_synergy_runrate=100_000_000.0,
        synergy_type="cost",
        deal_status="completed",
        announcement_date="2023-09-11",
        source=(
            "J.M. Smucker press release, Sept 2023: ~$100M annual run-rate cost synergies within "
            "the first 2 years -- "
            "https://www.prnewswire.com/news-releases/the-j-m-smucker-co-to-acquire-hostess-brands-to-accelerate-focus-on-convenient-consumer-occasions-301923243.html"
        ),
    ),
]


def find_matching_deal(acquirer_ticker: str, target_ticker: str) -> HistoricalDeal | None:
    """Look up a curated deal by ticker pair (case-insensitive) -- used by the
    Post-Deal app tab to auto-populate a known announcement date and show the
    re-underwriting variance bridge (estimated vs. disclosed synergies) when
    the user's ticker pair happens to match one of the validated deals."""
    acquirer_ticker, target_ticker = acquirer_ticker.upper(), target_ticker.upper()
    for deal in DEALS:
        if deal.acquirer_ticker == acquirer_ticker and deal.target_ticker == target_ticker:
            return deal
    return None
