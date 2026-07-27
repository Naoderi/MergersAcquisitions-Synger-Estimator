"""Interactive M&A deal lifecycle explorer: Price Paid -> Deal Economics -> Target Quality -> Post-Deal.

Implements the free-data-feasible techniques from ma-analytical-toolkit.md
using only SEC EDGAR + Yahoo Finance data. Techniques that fundamentally
require paid institutional platforms (Capital IQ, Bloomberg, PitchBook,
Anaplan, Kira, BitSight, etc.) or data this pipeline doesn't collect
(transaction-level GL, alternative data, contract text, customer cohorts) are
noted inline as out of scope rather than faked -- see the module docstrings
in models/price_paid.py, models/valuation.py, models/target_quality.py, and
models/post_deal.py for the specific list per phase.
"""

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from synergy_estimator.data.edgar_source import get_balance_sheet
from synergy_estimator.data.market_source import get_price_history, get_risk_free_rate
from synergy_estimator.data.pipeline import get_company_profile
from synergy_estimator.models.accretion_dilution import eps_accretion_dilution
from synergy_estimator.models.post_deal import event_study_from_price_history
from synergy_estimator.models.price_paid import (
    acquisition_premium,
    contribution_analysis,
    implied_offer_price,
    implied_ownership_split,
    valuation_range,
    vwap,
)
from synergy_estimator.models.sensitivity import tornado_sensitivity
from synergy_estimator.models.synergies import SynergyAssumptions, estimate_synergies
from synergy_estimator.models.target_quality import altman_z_score, working_capital_diagnostics
from synergy_estimator.models.valuation import (
    breakeven_synergies,
    leverage_ratio,
    monte_carlo_npv,
    npv_of_synergies,
    payback_period_years,
    roic,
    synergies_pct_of_deal_value,
)
from synergy_estimator.models.wacc import FALLBACK_TAX_RATE, blended_wacc, effective_tax_rate, estimate_wacc
from synergy_estimator.validation.backtest import run_backtest, score_deal
from synergy_estimator.validation.deals import find_matching_deal

st.set_page_config(page_title="M&A Synergy Estimator", layout="wide")

# Chart colours. The diverging pair encodes polarity (value destroyed vs
# created) and is deliberately blue/red rather than the conventional
# red/green -- red/green is the classic colour-vision failure, and this pair
# validates at CVD deltaE 21.6 against a >=8 target. Bars carry a 2px surface-
# coloured outline so adjacent bars read as separate marks.
_DIVERGING_NEGATIVE = "#e34948"
_DIVERGING_POSITIVE = "#2a78d6"
_SURFACE = "#fcfcfb"
_INK_MUTED = "#52514e"
_GRID = "#e8e7e3"


@st.cache_data(ttl=3600)
def _cached_company_profile(ticker: str):
    return get_company_profile(ticker)


@st.cache_data(ttl=3600)
def _cached_risk_free_rate():
    return get_risk_free_rate()


@st.cache_data(ttl=3600)
def _cached_balance_sheet(ticker: str):
    return get_balance_sheet(ticker, years=1)[0]


@st.cache_data(ttl=3600)
def _cached_price_history(ticker: str, start: str, end: str):
    return get_price_history(ticker, start, end)


@st.cache_data(ttl=86400)
def _cached_backtest(assumptions: SynergyAssumptions):
    return run_backtest(assumptions=assumptions)


st.title("M&A Synergy Estimator")
st.caption(
    "Walks a hypothetical acquirer/target pairing through the deal lifecycle -- Price Paid, Deal "
    "Economics, Target Quality, Post-Deal -- using live SEC EDGAR and Yahoo Finance data."
)

# The sidebar holds only what defines the *transaction* -- inputs every tab
# depends on. Analysis assumptions live in the tab they drive, so a control is
# never on screen for a view it has no effect on. (Streamlit renders all tabs
# on every run and never reports which one is focused, so a sidebar widget
# cannot react to tab selection; putting each control in its own tab is what
# actually achieves that.)
with st.sidebar:
    st.header("Deal terms")
    st.caption("These define the transaction and drive every tab.")
    deal_type = st.radio(
        "Deal type",
        ["Acquisition", "Merger"],
        horizontal=True,
        help=(
            "**Acquisition** -- one company buys the other. Direction matters: the buyer pays a "
            "premium for the seller, and earnings-per-share is measured for the buyer.\n\n"
            "**Merger** -- a merger of equals. The two are treated symmetrically and you get an "
            "implied ownership split instead of a premium."
        ),
    )

    # In an acquisition the two sides are not interchangeable -- one pays the
    # premium and one gets absorbed -- so the labels stay directional there. In
    # a merger of equals they are symmetric, and neutral labels are honest.
    if deal_type == "Merger":
        first_label, second_label = "Company 1 ticker", "Company 2 ticker"
        first_help = "Either party -- a merger of equals is modelled symmetrically."
        second_help = (
            "The other party. Swapping the two changes almost nothing here, since "
            "cost synergies are based on the combined cost base."
        )
    else:
        first_label, second_label = "Acquirer ticker", "Target ticker"
        first_help = "The buyer. Earnings-per-share accretion/dilution is measured from this company's perspective."
        second_help = (
            "The company being bought. The control premium is paid on **this** company's share "
            "price, so the order of the two tickers matters in an acquisition."
        )

    acquirer_ticker = st.text_input(first_label, "KR", help=first_help).strip().upper()
    target_ticker = st.text_input(second_label, "ACI", help=second_help).strip().upper()
    premium_pct = (
        st.slider(
            "Assumed control premium (%)",
            0,
            100,
            25,
            step=1,
            help=(
                "How much above the current share price the buyer offers. Control of a company "
                "almost always costs more than the market price -- 20-40% is typical.\n\n"
                "Raising it makes the deal more expensive, so more synergies are needed just to "
                "break even."
            ),
        )
        / 100.0
    )
    st.divider()
    st.caption(
        "Analysis assumptions sit inside the tab they affect -- synergy and NPV inputs on "
        "**Deal Economics**, event-study inputs on **Post-Deal**."
    )

matching_deal = find_matching_deal(acquirer_ticker, target_ticker) if acquirer_ticker and target_ticker else None

# How to name the two sides in output labels, matching the sidebar's wording.
first_role, second_role = ("Company 1", "Company 2") if deal_type == "Merger" else ("Acquirer", "Target")

if not acquirer_ticker or not target_ticker:
    st.info("Enter both an acquirer and target ticker in the sidebar to run the analysis.")
    st.stop()

st.caption(f"**{acquirer_ticker} {'acquires' if deal_type == 'Acquisition' else 'merges with'} {target_ticker}**")

try:
    with st.spinner(f"Fetching {acquirer_ticker} and {target_ticker} data from SEC EDGAR and Yahoo Finance..."):
        acquirer_profile = _cached_company_profile(acquirer_ticker)
        target_profile = _cached_company_profile(target_ticker)
        risk_free_rate = _cached_risk_free_rate()

    acquirer_financials = acquirer_profile.financials[0]
    target_financials = target_profile.financials[0]

    acquirer_wacc = estimate_wacc(acquirer_profile.market, acquirer_financials, risk_free_rate)
    target_wacc = estimate_wacc(target_profile.market, target_financials, risk_free_rate)
    combined_wacc = blended_wacc(
        acquirer_wacc,
        acquirer_profile.market.market_cap or 0.0,
        target_wacc,
        target_profile.market.market_cap or 0.0,
    )
except Exception as exc:  # network / lookup failures (bad ticker, EDGAR/Yahoo outage, etc.)
    message = f"Could not fetch or compute data for {acquirer_ticker}/{target_ticker}: {exc}"
    if matching_deal:
        message += (
            f" This matches a validated historical deal ({matching_deal.source.split(' -- ')[0]}), but its "
            "target is delisted post-close -- live tickers are required for this interactive view. See the "
            "Validation section below for its backtested result instead."
        )
    st.error(message)
    st.stop()

tab_price_paid, tab_deal_economics, tab_target_quality, tab_post_deal = st.tabs(
    ["1. Price Paid", "2. Deal Economics", "3. Target Quality", "4. Post-Deal"]
)

# ---------------------------------------------------------------------------
# Per-tab controls. Written before any tab's output so the values are available
# to the shared calculation below; each tab container is re-entered further
# down to append its charts underneath its own controls.
# ---------------------------------------------------------------------------
with tab_deal_economics:
    with st.expander("Assumptions -- drive every figure on this tab", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        sga_overlap_rate = col1.slider(
            "SG&A overlap rate", 0.0, 0.10, 0.015, step=0.001, format="%.3f",
            help=(
                "**The single biggest driver of the synergy estimate.**\n\n"
                "SG&A is overhead -- head office, finance, HR, marketing, duplicated executives. "
                "This is the share of the two companies' *combined* SG&A that disappears when they "
                "merge, because you only need one of each function.\n\n"
                "It varies hugely by industry: a software merger removes a lot of duplicated "
                "overhead, whereas a supermarket merger removes very little, because most of "
                "grocery 'SG&A' is store staff who still have to serve the same customers."
            ),
        )
        cogs_overlap_rate = col2.slider(
            "COGS overlap rate", 0.0, 0.02, 0.0015, step=0.0001, format="%.4f",
            help=(
                "COGS (cost of goods sold) is the direct cost of making or buying what the company "
                "sells. Merging saves some of it through buying power and scale -- but far less "
                "than on overhead, since you still make the same number of units.\n\n"
                "Kept deliberately small: the slider tops out at 2% for a reason."
            ),
        )
        revenue_cross_sell_rate = col3.slider(
            "Revenue cross-sell rate", 0.0, 0.10, 0.01, step=0.001, format="%.3f",
            help=(
                "Extra sales from selling each company's products to the other's customers, as a "
                "share of the smaller company's revenue.\n\n"
                ":warning: **Not calibrated.** Revenue synergies are rarely disclosed with a "
                "number, so unlike the cost sliders there is no evidence base behind this. Treat "
                "anything it produces as your assumption, not the model's."
            ),
        )
        revenue_confidence_weight = col4.slider(
            "Revenue confidence weight", 0.0, 1.0, 0.5, step=0.05,
            help=(
                "A haircut on revenue synergies, from 0 (ignore them entirely) to 1 (take them at "
                "face value).\n\n"
                "Revenue synergies are much less reliable than cost savings -- cutting a duplicated "
                "office is within management's control, persuading customers to buy more is not. "
                "Acquirers routinely miss them, which is why the default discounts them by half."
            ),
        )

        col5, col6, col7, col8 = st.columns(4)
        integration_cost_multiple = col5.slider(
            "Integration cost (x run-rate)", 0.0, 3.0, 1.25, step=0.05,
            help=(
                "The one-off cost of actually capturing the synergies -- redundancies, systems "
                "migration, advisers, rebranding -- expressed as a multiple of the annual savings.\n\n"
                "1.25x means it costs about $1.25 up front for every $1 of yearly saving. It is "
                "subtracted from the NPV as a day-one cash outflow."
            ),
        )
        forecast_years = col6.slider(
            "Forecast years", 1, 10, 5,
            help=(
                "How many years of synergies to count when discounting them back to today's money.\n\n"
                "There is no terminal value beyond this horizon, so a longer forecast mechanically "
                "raises the NPV. Five years is the convention and is deliberately conservative."
            ),
        )
        duplicate_public_company_cost = col7.number_input(
            "Duplicate public-company cost ($)", value=15_000_000.0, step=1_000_000.0, format="%.0f",
            help=(
                "The fixed cost of being a second listed company -- audit, listing fees, investor "
                "relations, a second board and annual report. It vanishes when two public companies "
                "combine, whatever their size, so it is added as a flat dollar amount rather than "
                "a percentage."
            ),
        )
        new_shares_issued = col8.number_input(
            "New shares issued (stock deal)", value=0.0, step=1_000_000.0, format="%.0f",
            help=(
                "If the buyer pays with its own shares rather than cash, enter how many new shares "
                "it issues.\n\n"
                "More shares split the combined profit more ways, which pushes earnings-per-share "
                "down. Leave at 0 for an all-cash deal."
            ),
        )

with tab_post_deal:
    with st.expander("Event study window", expanded=True):
        col1, col2 = st.columns(2)
        default_announcement = (
            datetime.strptime(matching_deal.announcement_date, "%Y-%m-%d").date() if matching_deal else None
        )
        announcement_date = col1.date_input(
            "Announcement date",
            value=default_announcement,
            min_value=date(2015, 1, 1),
            max_value=date.today(),
            help=(
                "The day the deal was made public. Only meaningful for a real announced deal -- a "
                "hypothetical pairing has no announcement to measure.\n\n"
                "The event study compares each company's share price around this date against how "
                "the S&P 500 moved, to isolate the part of the move the deal itself caused."
            ),
        )
        event_window_days = col2.slider(
            "Event window (trading days each side)", 1, 10, 5,
            help=(
                "How many trading days either side of the announcement to add up.\n\n"
                "A short window is cleaner -- it captures the reaction to the news and little "
                "else. A longer one catches leaks and later reaction, but also picks up unrelated "
                "noise that has nothing to do with the deal."
            ),
        )

assumptions = SynergyAssumptions(
    sga_overlap_rate=sga_overlap_rate,
    cogs_overlap_rate=cogs_overlap_rate,
    duplicate_public_company_cost=duplicate_public_company_cost,
    revenue_cross_sell_rate=revenue_cross_sell_rate,
    revenue_confidence_weight=revenue_confidence_weight,
    integration_cost_multiple=integration_cost_multiple,
)

try:
    estimate = estimate_synergies(
        acquirer_ticker, acquirer_financials, target_ticker, target_financials, assumptions
    )
    npv = npv_of_synergies(estimate, combined_wacc, forecast_years=forecast_years)
    pct_of_deal_value = synergies_pct_of_deal_value(npv, target_profile.market.market_cap or 0.0)
    payback = payback_period_years(estimate)
    ad_result = eps_accretion_dilution(
        acquirer_financials, target_financials, estimate.total_runrate_synergies, new_shares_issued
    )
except ValueError as exc:
    with tab_deal_economics:
        st.error(f"Could not compute synergies: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# 1. Price Paid
# ---------------------------------------------------------------------------
with tab_price_paid:
    contribution_rows = contribution_analysis(acquirer_financials, target_financials)
    if contribution_rows:
        st.subheader("Contribution analysis")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=[r.metric for r in contribution_rows],
                x=[r.acquirer_pct for r in contribution_rows],
                name=acquirer_ticker,
                orientation="h",
            )
        )
        fig.add_trace(
            go.Bar(
                y=[r.metric for r in contribution_rows],
                x=[r.target_pct for r in contribution_rows],
                name=target_ticker,
                orientation="h",
            )
        )
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%", xaxis_title="% of combined")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough financial data to compute contribution analysis.")

    if deal_type == "Merger":
        st.subheader("Implied ownership split (all-stock)")
        try:
            split = implied_ownership_split(
                acquirer_profile.market.market_cap, target_profile.market.market_cap, premium_pct
            )
            col1, col2 = st.columns(2)
            col1.metric(f"{target_ticker} implied ownership", f"{split.target_ownership_pct:.1%}")
            col2.metric(f"{acquirer_ticker} implied ownership", f"{split.acquirer_ownership_pct:.1%}")
            st.caption(
                f"At a {premium_pct:.0%} premium, implied target deal value ~ "
                f"${split.implied_target_deal_value:,.0f}."
            )
            revenue_row = next((r for r in contribution_rows if r.metric == "Revenue"), None)
            if revenue_row and abs(split.target_ownership_pct - revenue_row.target_pct) > 0.05:
                st.warning(
                    f"{target_ticker} would receive {split.target_ownership_pct:.1%} ownership while "
                    f"contributing {revenue_row.target_pct:.1%} of combined revenue -- worth scrutinizing "
                    "in a true merger of equals."
                )
        except ValueError as exc:
            st.info(str(exc))
    else:
        st.subheader("Acquisition premium")
        unaffected_price = target_profile.market.share_price
        if unaffected_price:
            offer_price = implied_offer_price(unaffected_price, premium_pct)
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{target_ticker} current price", f"${unaffected_price:,.2f}")
            col2.metric("Implied offer price", f"${offer_price:,.2f}")
            col3.metric(
                "Premium", f"{acquisition_premium(offer_price, unaffected_price):.0%}"
            )
            try:
                window_end = date.today().isoformat()
                window_start = (date.today() - timedelta(days=45)).isoformat()
                target_points = _cached_price_history(target_ticker, window_start, window_end)
                target_vwap = vwap(target_points)
                if target_vwap:
                    st.caption(f"~30-trading-day VWAP: ${target_vwap:,.2f}")
            except Exception:
                pass
        else:
            st.info(f"No current share price available for {target_ticker}.")

    st.subheader("Valuation range (football field)")
    bars = valuation_range(target_profile.market)
    if bars:
        fig = go.Figure()
        for bar in bars:
            fig.add_trace(
                go.Bar(x=[bar.high - bar.low], y=[bar.label], base=[bar.low], orientation="h", name=bar.label)
            )
        if target_profile.market.share_price:
            fig.add_vline(x=target_profile.market.share_price, line_dash="dash", annotation_text="Current price")
        fig.update_layout(showlegend=False, xaxis_title="$ per share")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No 52-week range or analyst target data available for {target_ticker}.")
    st.caption(
        "Full comparable-companies/precedent-transaction analysis and a standalone DCF valuation floor "
        "require a deals database and a full company DCF model this pipeline doesn't have."
    )

# ---------------------------------------------------------------------------
# 2. Deal Economics
# ---------------------------------------------------------------------------
with tab_deal_economics:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total run-rate synergies", f"${estimate.total_runrate_synergies:,.0f}")
    col2.metric("Synergy NPV", f"${npv:,.0f}")
    col3.metric(
        "% of target market cap", f"{pct_of_deal_value:.1%}" if pct_of_deal_value is not None else "n/a"
    )
    col4.metric("Payback period", f"{payback:.1f} yrs" if payback is not None else "n/a")
    col5.metric(
        "EPS accretion/(dilution)",
        f"{ad_result.pct_change_with_synergies:+.1%}" if ad_result.pct_change_with_synergies is not None else "n/a",
    )

    st.subheader("Synergy build-up")
    waterfall = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Cost synergies", "Revenue synergies", "Integration cost", "Net"],
            y=[estimate.cost_synergies_runrate, estimate.revenue_synergies_runrate, -estimate.integration_cost, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )
    waterfall.update_layout(showlegend=False, yaxis_title="$")
    st.plotly_chart(waterfall, use_container_width=True)
    st.caption(
        "Cost and revenue synergies are annual run-rate figures; integration cost is a one-time upfront "
        "outlay shown here for scale comparison, not an annual expense."
    )

    st.subheader("NPV sensitivity")
    sensitivity_rows = tornado_sensitivity(
        acquirer_ticker, acquirer_financials, target_ticker, target_financials, assumptions, combined_wacc,
        forecast_years=forecast_years,
    )
    tornado = go.Figure()
    labels = [row.label for row in sensitivity_rows]
    tornado.add_trace(
        go.Bar(
            y=labels, x=[row.high_npv - row.base_npv for row in sensitivity_rows],
            base=[row.base_npv for row in sensitivity_rows], orientation="h", name="High", marker_color="seagreen",
        )
    )
    tornado.add_trace(
        go.Bar(
            y=labels, x=[row.low_npv - row.base_npv for row in sensitivity_rows],
            base=[row.base_npv for row in sensitivity_rows], orientation="h", name="Low", marker_color="indianred",
        )
    )
    tornado.update_layout(barmode="overlay", xaxis_title="Synergy NPV ($)")
    st.plotly_chart(tornado, use_container_width=True)
    st.caption("Each driver is perturbed +/-25% (WACC +/-150bps) from its current slider value, holding others fixed.")

    st.subheader("Breakeven synergy analysis")
    premium_paid_dollars = (target_profile.market.market_cap or 0.0) * premium_pct
    try:
        breakeven = breakeven_synergies(premium_paid_dollars, assumptions, combined_wacc, forecast_years)
        col1, col2 = st.columns(2)
        col1.metric("Breakeven run-rate synergies needed", f"${breakeven:,.0f}")
        delta = f"{(estimate.total_runrate_synergies - breakeven) / breakeven:+.0%}" if breakeven else None
        col2.metric("Estimated run-rate synergies", f"${estimate.total_runrate_synergies:,.0f}", delta)
        st.caption(
            f"Run-rate synergies required for NPV to fully offset a {premium_pct:.0%} premium "
            f"(~${premium_paid_dollars:,.0f}) over {target_ticker}'s current market cap."
        )
    except ValueError as exc:
        st.info(str(exc))

    st.subheader("ROIC vs. WACC")
    acquirer_tax_rate = effective_tax_rate(acquirer_financials) or FALLBACK_TAX_RATE
    target_tax_rate = effective_tax_rate(target_financials) or FALLBACK_TAX_RATE
    combined_nopat = (acquirer_financials.operating_income or 0.0) * (1 - acquirer_tax_rate) + (
        target_financials.operating_income or 0.0
    ) * (1 - target_tax_rate)
    combined_invested_capital = (
        (acquirer_profile.market.market_cap or 0.0)
        + (acquirer_profile.market.total_debt or 0.0)
        + (target_profile.market.market_cap or 0.0)
        + (target_profile.market.total_debt or 0.0)
    )
    combined_roic = roic(combined_nopat, combined_invested_capital)
    col1, col2, col3 = st.columns(3)
    col1.metric("Combined ROIC", f"{combined_roic:.1%}" if combined_roic is not None else "n/a")
    col2.metric("Blended WACC", f"{combined_wacc:.1%}")
    col3.metric("Spread", f"{combined_roic - combined_wacc:+.1%}" if combined_roic is not None else "n/a")

    st.subheader("Leverage")
    combined_debt = (acquirer_profile.market.total_debt or 0.0) + (target_profile.market.total_debt or 0.0)
    combined_ebitda_proxy = (acquirer_financials.operating_income or 0.0) + (target_financials.operating_income or 0.0)
    leverage = leverage_ratio(combined_debt, combined_ebitda_proxy)
    st.metric("Combined debt / operating-income proxy", f"{leverage:.1f}x" if leverage is not None else "n/a")
    st.caption(
        "Operating income is used as an EBITDA proxy (no separate D&A line in this pipeline) -- a debt-capacity "
        "sanity check, not a real covenant-headroom model."
    )

    st.subheader("Monte Carlo: NPV distribution")
    mc_result = monte_carlo_npv(
        acquirer_ticker, acquirer_financials, target_ticker, target_financials, assumptions, combined_wacc,
        forecast_years=forecast_years, n_trials=300, seed=42,
    )
    # Bars are split at zero and coloured on a validated diverging pair, so the
    # share of trials that destroy value is legible at a glance rather than
    # something you have to read off the caption. Bin edges are aligned to zero
    # so no single bar straddles the boundary and lands in both colours.
    npvs = mc_result.npvs
    low, high = min(npvs), max(npvs)
    bin_size = (high - low) / 32 if high > low else 1.0
    bins_below_zero = math.ceil((0 - low) / bin_size) if low < 0 else 0
    xbins = dict(start=-bins_below_zero * bin_size, end=high + bin_size, size=bin_size)

    hist = go.Figure()
    for values, name, colour in (
        ([v for v in npvs if v < 0], "Destroys value (NPV < 0)", _DIVERGING_NEGATIVE),
        ([v for v in npvs if v >= 0], "Creates value (NPV ≥ 0)", _DIVERGING_POSITIVE),
    ):
        if not values:
            continue
        hist.add_trace(
            go.Histogram(
                x=values,
                xbins=xbins,
                name=name,
                marker=dict(color=colour, line=dict(color=_SURFACE, width=2)),
                hovertemplate="NPV around %{x:$,.3s}<br>%{y} of 300 trials<extra></extra>",
            )
        )
    for value, label in ((mc_result.p10, "P10"), (mc_result.p50, "P50"), (mc_result.p90, "P90")):
        hist.add_vline(
            x=value, line_dash="dot", line_width=1, line_color=_INK_MUTED,
            annotation_text=label, annotation_font_color=_INK_MUTED,
        )
    hist.update_layout(
        barmode="overlay",
        bargap=0.12,
        xaxis_title="Synergy NPV ($)",
        yaxis_title="Trials",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=48),
    )
    hist.update_xaxes(showgrid=False, zeroline=False, tickformat="$,.3s")
    hist.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    st.plotly_chart(hist, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("P10 NPV", f"${mc_result.p10:,.0f}")
    col2.metric("P50 NPV", f"${mc_result.p50:,.0f}")
    col3.metric("P90 NPV", f"${mc_result.p90:,.0f}")
    st.caption(
        f"{mc_result.pct_positive:.0%} of 300 trials had positive NPV -- SG&A/COGS overlap and revenue "
        "cross-sell rates are jittered independently +/-30% (triangular distribution) around your slider values."
    )

    with st.expander("Validation: backtest against real disclosed deals"):
        st.caption(
            "Compares this model's estimates (using this tab's current assumptions) against publicly "
            "disclosed synergy targets for 8 real, completed M&A deals not used to calibrate the model. "
            "Results are cached for 24 hours."
        )
        if st.button("Run validation backtest"):
            with st.spinner("Fetching financials for 8 validation deals from SEC EDGAR..."):
                backtest_results = _cached_backtest(assumptions)
            st.dataframe(
                [
                    {
                        "Acquirer": r.deal.acquirer_ticker,
                        "Target": r.deal.target_ticker,
                        "Estimated": r.estimated_synergies,
                        "Disclosed": r.disclosed_synergies,
                        "% Error": r.pct_error,
                    }
                    for r in backtest_results
                ],
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# 3. Target Quality
# ---------------------------------------------------------------------------
with tab_target_quality:
    st.caption(
        "No assumptions to set here -- every figure on this tab is read straight from the two "
        "companies' filed balance sheets."
    )
    try:
        with st.spinner("Fetching balance sheet data from SEC EDGAR..."):
            acquirer_bs = _cached_balance_sheet(acquirer_ticker)
            target_bs = _cached_balance_sheet(target_ticker)

        st.subheader("Altman Z-score")
        col1, col2 = st.columns(2)
        for col, ticker, financials, bs, market in (
            (col1, acquirer_ticker, acquirer_financials, acquirer_bs, acquirer_profile.market),
            (col2, target_ticker, target_financials, target_bs, target_profile.market),
        ):
            z = altman_z_score(financials, bs, market)
            if z:
                col.metric(f"{ticker} Z-score", f"{z.z_score:.2f}", z.zone)
            else:
                col.info(f"Not enough balance-sheet data to compute {ticker}'s Z-score.")
        st.caption("Z > 2.99 \"safe\", 1.81-2.99 \"grey\", Z < 1.81 \"distress\" (Altman's public-company model).")

        st.subheader("Working capital cycle")
        wc_rows = [
            {
                "Ticker": ticker,
                "DSO (days)": diag.days_sales_outstanding,
                "DIO (days)": diag.days_inventory_outstanding,
                "DPO (days)": diag.days_payable_outstanding,
                "Cash conversion cycle (days)": diag.cash_conversion_cycle,
            }
            for ticker, diag in (
                (acquirer_ticker, working_capital_diagnostics(acquirer_financials, acquirer_bs)),
                (target_ticker, working_capital_diagnostics(target_financials, target_bs)),
            )
        ]
        st.dataframe(wc_rows, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not fetch balance-sheet data: {exc}")

    st.caption(
        "Quality-of-earnings on transaction-level GL data, cohort/churn/survival analysis, concentration "
        "metrics, price-volume-mix decomposition, the Beneish M-score, and contract/cyber diligence require "
        "transaction-level or alternative data this free-data pipeline doesn't have."
    )

# ---------------------------------------------------------------------------
# 4. Post-Deal
# ---------------------------------------------------------------------------
with tab_post_deal:
    if matching_deal:
        st.subheader("Re-underwriting: estimate vs. disclosed")
        result = score_deal(estimate, matching_deal)
        col1, col2, col3 = st.columns(3)
        col1.metric("Disclosed synergy target", f"${result.disclosed_synergies:,.0f}")
        col2.metric("This model's estimate", f"${result.estimated_synergies:,.0f}")
        col3.metric("% error", f"{result.pct_error:+.1%}")
        st.caption(
            f"The estimate compared here is the one produced by the **Deal Economics** tab's "
            f"assumptions. {matching_deal.source}"
        )

    st.subheader("Event study: cumulative abnormal returns")
    if announcement_date is None:
        st.info("Set an announcement date above to run the event study.")
    else:
        event_date_str = announcement_date.isoformat()
        try:
            with st.spinner("Fetching price history from Yahoo Finance..."):
                window_start = (announcement_date - timedelta(days=400)).isoformat()
                window_end = (announcement_date + timedelta(days=30)).isoformat()
                market_points = _cached_price_history("^GSPC", window_start, window_end)

                car_results = {}
                car_errors = {}
                for label, ticker in ((first_role, acquirer_ticker), (second_role, target_ticker)):
                    try:
                        stock_points = _cached_price_history(ticker, window_start, window_end)
                        car_results[label] = event_study_from_price_history(
                            stock_points, market_points, event_date_str, event_window_days=event_window_days
                        )
                    except Exception as exc:
                        car_errors[label] = str(exc)

            if car_results:
                label_to_ticker = {first_role: acquirer_ticker, second_role: target_ticker}
                cols = st.columns(len(car_results))
                for col, (label, result) in zip(cols, car_results.items()):
                    col.metric(
                        f"{label} ({label_to_ticker[label]}) CAR",
                        f"{result.cumulative_abnormal_return:+.1%}",
                        f"beta={result.beta:.2f}",
                    )
                fig = go.Figure()
                for label, result in car_results.items():
                    fig.add_trace(
                        go.Bar(x=list(range(len(result.abnormal_returns))), y=result.abnormal_returns, name=label)
                    )
                fig.update_layout(
                    barmode="group", xaxis_title="Trading day index within event window",
                    yaxis_title="Abnormal return",
                )
                st.plotly_chart(fig, use_container_width=True)
            for label, msg in car_errors.items():
                st.warning(f"{label}: {msg}")
        except Exception as exc:
            st.warning(f"Could not run event study: {exc}")

    st.caption(
        "Impairment testing, retained-cohort tracking, buy-and-hold abnormal returns, and "
        "difference-in-differences vs. a matched peer group require disclosures and data this pipeline "
        "doesn't parse."
    )
