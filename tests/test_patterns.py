import pytest

from synergy_estimator.corpus.patterns import (
    classify,
    extract_figures,
    headline_figure,
)


def _one(text):
    figures = extract_figures(text)
    assert len(figures) == 1, [f.quote for f in figures]
    return figures[0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$450 million in annual run-rate cost synergies", 450e6),
        ("cost synergies of approximately $400 million", 400e6),
        ("$2.5 billion of expected cost synergies", 2.5e9),
        ("synergies of approximately $10.1 million", 10.1e6),
        ("$1,250 million of annual cost synergies", 1.25e9),
        ("$40 million of identified cost synergies", 40e6),
    ],
)
def test_parses_real_disclosure_shapes(text, expected):
    assert _one(text).amount_usd == pytest.approx(expected)


def test_excludes_costs_to_achieve_synergies():
    """'$60 million in connection with achieving anticipated synergies' is money
    spent, not saved -- and it has the same sentence shape as a real target."""
    assert extract_figures("expects to incur $60 million in connection with achieving anticipated synergies") == []


@pytest.mark.parametrize(
    "text",
    [
        "one-time costs of $75 million to achieve the anticipated synergies",
        "integration cost of approximately $30 million related to synergies",
        "restructuring charges of $20 million to achieve these synergies",
    ],
)
def test_excludes_other_cost_to_achieve_phrasings(text):
    assert extract_figures(text) == []


def test_classifies_on_the_matched_span_not_the_paragraph():
    """A filing discussing both kinds must not label every figure in it
    cost_and_revenue -- the probe mislabelled exactly this case."""
    text = (
        "The transaction is expected to deliver $200 million of annual cost synergies. "
        "Separately, management identified $5 million in potential annual revenue synergies."
    )

    figures = extract_figures(text)

    by_amount = {f.amount_usd: f.synergy_type for f in figures}
    assert by_amount[200e6] == "cost"
    assert by_amount[5e6] == "revenue"


def test_classify_detects_both_kinds_in_one_span():
    assert classify("$350 million of cost and revenue synergies") == "cost_and_revenue"


def test_classify_falls_back_to_unspecified():
    assert classify("$15 million in run-rate synergies") == "unspecified"


def test_rejects_implausible_magnitudes():
    assert extract_figures("$0.2 million of cost synergies") == []
    assert extract_figures("$900 billion of cost synergies") == []


def test_overlapping_patterns_yield_one_figure():
    """Several patterns match the same sentence; it must be counted once."""
    figures = extract_figures("annual run-rate cost synergies of approximately $450 million")

    assert len(figures) == 1


def test_retains_a_verbatim_quote_for_auditing():
    figure = _one("we expect $450 million in annual run-rate cost synergies by year three")

    assert "450 million" in figure.quote
    assert figure.quote in "we expect $450 million in annual run-rate cost synergies by year three"


def test_headline_prefers_the_largest_cost_figure():
    """Announcements quantify several things; the convention is the full
    cost run-rate, not an interim milestone or the revenue number."""
    text = (
        "$120 million of annual cost synergies expected in year one, "
        "rising to $200 million of run-rate cost synergies by year three, "
        "plus $80 million in potential revenue synergies."
    )

    headline = headline_figure(extract_figures(text))

    assert headline.amount_usd == pytest.approx(200e6)
    assert headline.synergy_type == "cost"


def test_headline_falls_back_to_untyped_figures():
    headline = headline_figure(extract_figures("$15 million of run-rate synergies"))

    assert headline.amount_usd == pytest.approx(15e6)


def test_headline_of_nothing_is_none():
    assert headline_figure([]) is None


def test_ignores_prose_with_no_figure():
    assert extract_figures("We may fail to realize the anticipated synergies of the merger.") == []


def test_excludes_figures_that_merely_say_they_omit_synergies():
    """'$4.5 billion of EBITDA before synergies' is a valuation metric that
    mentions synergies only to exclude them. Found by hand-auditing the corpus:
    the sentence shape is identical to a real disclosure."""
    assert extract_figures("reported $4.5 billion of EBITDA before synergies") == []


@pytest.mark.parametrize(
    "text",
    [
        "$2.0 billion of EBITDA excluding synergies",
        "$500 million of run-rate EBITDA pre-synergies",
        "adjusted EBITDA of $1.2 billion without synergies",
    ],
)
def test_excludes_other_pre_synergy_qualifiers(text):
    assert extract_figures(text) == []


def test_still_accepts_a_genuine_ebit_synergy_disclosure():
    """The exclusion must not swallow real figures that happen to name an
    earnings measure -- 'EBIT synergies' is a normal way to disclose one."""
    (figure,) = extract_figures("$110 million in expected EBIT synergies")

    assert figure.amount_usd == pytest.approx(110e6)
