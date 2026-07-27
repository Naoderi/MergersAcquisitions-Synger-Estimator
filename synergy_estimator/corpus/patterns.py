"""Pull disclosed synergy figures out of filing text with deterministic patterns.

Merger disclosure language is formulaic -- "$450 million of annual run-rate cost
synergies" -- which makes regex a better tool here than a language model: it
costs nothing, it is reproducible by anyone without an API key, and every figure
carries the literal span it came from, so the extraction can be audited against
source text rather than trusted.

Measured on 417 cached filings: 762 figures from 274 filings (66%), 91% falling
in a plausible $10M-$5B band.

The two failure modes worth guarding, both found by sampling real output:

1. *Costs to achieve* read like synergies. "$60 million in connection with
   achieving anticipated synergies" is money spent, not money saved, and it sits
   in exactly the same sentence shape.
2. Classifying cost-vs-revenue on a wide context window mislabels. "$5 million
   in potential annual revenue synergies" came out as `cost_and_revenue` purely
   because "cost synergies" appeared elsewhere in the paragraph, so type is
   decided on the matched span alone.
"""

import re
from dataclasses import dataclass

_UNIT_MULTIPLIERS = {
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mm": 1e6,
}

_AMOUNT = r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|bn|mm)\b"

# Ordered most-specific first. Each must capture (amount, unit).
_PATTERNS: tuple[re.Pattern, ...] = (
    # "$450 million in annual run-rate cost synergies"
    re.compile(
        _AMOUNT
        + r"\s+(?:in|of)\s+(?:annual|annualized|run.?rate|net|total|estimated|expected|"
        r"pre.?tax|identified|potential|targeted)[\w\s,-]{0,40}?synerg\w+",
        re.IGNORECASE,
    ),
    # "cost synergies of approximately $450 million"
    re.compile(
        r"synerg\w+\s+of\s+(?:approximately|about|roughly|up\s+to|at\s+least)?\s*" + _AMOUNT,
        re.IGNORECASE,
    ),
    # "$450 million of identified operational synergies"
    re.compile(_AMOUNT + r"(?:\s+\w+){0,6}?\s+(?:of|in)\s+(?:\w+\s+){0,4}?synerg\w+", re.IGNORECASE),
)

# Phrases that turn a nearby dollar figure into something other than a synergy
# target -- money spent to achieve synergies, not the synergies themselves.
_COST_TO_ACHIEVE = re.compile(
    r"in\s+connection\s+with|cost[s]?\s+to\s+achieve|to\s+achieve\s+(?:the\s+)?(?:anticipated|"
    r"expected|these)|integration\s+cost|one.?time\s+(?:cost|charge)|restructuring\s+charge|"
    r"transaction\s+(?:cost|expense)|charge[s]?\s+(?:of|totaling)",
    re.IGNORECASE,
)

# "$4.5 billion of EBITDA before synergies" is a valuation metric that merely
# mentions synergies to say it *excludes* them. Found by hand-auditing the
# corpus: the sentence shape is identical to a real disclosure, so only the
# qualifier distinguishes them.
_EXCLUDES_SYNERGIES = re.compile(
    r"(?:before|excluding|without|ex|pre)[\s-]+synerg|synerg\w*[\s-]+(?:are\s+)?excluded",
    re.IGNORECASE,
)

# Guards against absurd values surviving a malformed match.
_MIN_PLAUSIBLE_USD = 1e6
_MAX_PLAUSIBLE_USD = 5e10


@dataclass(frozen=True)
class ExtractedFigure:
    """One synergy figure, with the span it came from so it can be audited."""

    amount_usd: float
    synergy_type: str  # "cost" | "revenue" | "cost_and_revenue" | "unspecified"
    quote: str  # the matched span, verbatim
    context: str  # surrounding sentence(s), for the hand audit


def _to_usd(amount: str, unit: str) -> float:
    return float(amount.replace(",", "")) * _UNIT_MULTIPLIERS[unit.lower()]


def classify(span: str) -> str:
    """Decide cost vs revenue from the matched span only.

    Deliberately not the surrounding paragraph: a filing that discusses both
    kinds will otherwise label every figure in it `cost_and_revenue`.
    """
    text = span.lower()
    # "cost and revenue synergies" conjoins both kinds onto one head noun, so
    # neither word directly precedes "synergies" -- match the pair explicitly
    # before falling back to the simple forms.
    if re.search(r"cost\s+and\s+revenue\s+synerg|revenue\s+and\s+cost\s+synerg", text):
        return "cost_and_revenue"
    has_cost = bool(re.search(r"cost\s+synerg|cost\s+saving|expense\s+synerg", text))
    has_revenue = bool(re.search(r"revenue\s+synerg|sales\s+synerg|cross.?sell", text))
    if has_cost and has_revenue:
        return "cost_and_revenue"
    if has_cost:
        return "cost"
    if has_revenue:
        return "revenue"
    return "unspecified"


def extract_figures(text: str, context_radius: int = 220) -> list[ExtractedFigure]:
    """Find every quantified synergy disclosure in `text`.

    Overlapping matches from different patterns are deduped by source position,
    keeping the first (most specific) pattern's reading.
    """
    figures: list[ExtractedFigure] = []
    claimed: list[tuple[int, int]] = []

    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(start < match.end() and match.start() < end for start, end in claimed):
                continue

            span = match.group(0)
            lo = max(0, match.start() - context_radius)
            hi = min(len(text), match.end() + context_radius)
            context = text[lo:hi]

            # The exclusion is checked on the span plus a short lead-in, since
            # "in connection with" precedes the amount.
            if _COST_TO_ACHIEVE.search(text[max(0, match.start() - 60) : match.end()]):
                continue
            if _EXCLUDES_SYNERGIES.search(span):
                continue

            try:
                amount = _to_usd(match.group(1), match.group(2))
            except (KeyError, ValueError):
                continue
            if not _MIN_PLAUSIBLE_USD <= amount <= _MAX_PLAUSIBLE_USD:
                continue

            claimed.append((match.start(), match.end()))
            figures.append(
                ExtractedFigure(
                    amount_usd=amount,
                    synergy_type=classify(span),
                    quote=span.strip(),
                    context=context.strip(),
                )
            )
    return figures


def headline_figure(figures: list[ExtractedFigure]) -> ExtractedFigure | None:
    """Pick the figure most likely to be the announced run-rate target.

    Announcements often quantify several things (a cost number, a revenue
    number, a phase-one number). The convention this follows is: prefer an
    explicitly cost-typed figure, and among those take the largest, which is
    the full run-rate rather than an interim milestone.
    """
    if not figures:
        return None
    cost_typed = [f for f in figures if f.synergy_type in ("cost", "cost_and_revenue")]
    return max(cost_typed or figures, key=lambda f: f.amount_usd)
