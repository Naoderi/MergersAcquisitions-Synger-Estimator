"""Benchmark the fitted model against the rules of thumb it has to beat.

A median absolute error of 75% means nothing on its own. The question that
decides whether any of this was worth doing is "compared to what?" -- and the
honest comparator is not zero, it is the one-line heuristics people already use
on a desk: synergies as a percentage of combined revenue, or of the target's
operating cost base.

Every baseline here has its single parameter **fitted on the training split**,
using the same log-space objective as the real model. Comparing a fitted model
against an arbitrary hand-picked constant would be rigging the test; the only
comparison that means anything is fitted-vs-fitted on the same holdout.

The log-space objective makes each fit closed-form. Minimizing
`mean((log(a) + log(x) - log(y))^2)` over `a` gives `log(a) = mean(log(y/x))`,
i.e. the geometric mean of the observed ratios -- no optimizer required.
"""

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable

from synergy_estimator.calibration.fit import Coefficients, Observation

# The hand-calibrated defaults currently shipping in models/synergies.py,
# fitted on two deals. This is the incumbent the recalibration has to improve on.
INCUMBENT = Coefficients(sga_overlap_rate=0.015, cogs_overlap_rate=0.0015, fixed_cost=15_000_000.0)


@dataclass(frozen=True)
class Baseline:
    """A named predictor with an optional single fitted scale parameter."""

    name: str
    description: str
    predict: Callable[[Observation], float]
    parameter: float | None = None


def _geometric_mean_ratio(observations: list[Observation], driver: Callable[[Observation], float | None]) -> float:
    """The scale `a` minimizing log-space squared error for `a * driver(o)`."""
    ratios = [
        math.log(o.disclosed_synergy / driver(o))
        for o in observations
        if driver(o) and driver(o) > 0 and o.disclosed_synergy > 0
    ]
    if not ratios:
        raise ValueError("no usable observations for this baseline")
    return math.exp(statistics.fmean(ratios))


def fit_baselines(train: list[Observation]) -> list[Baseline]:
    """Fit every comparator's parameter on the training split only."""
    baselines: list[Baseline] = []

    # 1. Constant. Tests whether the model's inputs carry any signal at all --
    #    if a single number for every deal scores as well, the drivers do not.
    constant = math.exp(statistics.fmean(math.log(o.disclosed_synergy) for o in train))
    baselines.append(
        Baseline(
            name="constant",
            description="Same figure for every deal (geometric mean of the training set)",
            predict=lambda o, c=constant: c,
            parameter=constant,
        )
    )

    # 2/3. Percentage-of-a-driver heuristics. A driver the corpus does not
    #      carry yields no baseline at all -- reported as unavailable rather
    #      than silently substituted with a different driver.
    for name, label, driver in (
        ("pct_of_combined_revenue", "combined revenue", lambda o: o.combined_revenue),
        ("pct_of_target_opex", "target COGS + SG&A", lambda o: o.target_opex),
    ):
        try:
            rate = _geometric_mean_ratio(train, driver)
        except ValueError:
            continue
        baselines.append(
            Baseline(
                name=name,
                description=f"{rate:.2%} of {label}",
                predict=lambda o, a=rate, d=driver: a * (d(o) or 0.0),
                parameter=rate,
            )
        )

    # 4. The incumbent two-deal defaults, unfitted -- does recalibration help?
    baselines.append(
        Baseline(
            name="incumbent_defaults",
            description="Shipping defaults: 1.5% SG&A + 0.15% COGS + $15M",
            predict=lambda o: INCUMBENT.predict(o.combined_sga, o.combined_cogs),
        )
    )

    return baselines


@dataclass
class Score:
    name: str
    n: int
    median_abs_error: float
    median_signed_error: float
    within_50pct: float  # share of deals predicted within +/-50% of disclosed

    def describe(self) -> str:
        return (
            f"{self.name:<26} n={self.n:<4} "
            f"median abs {self.median_abs_error:>6.1%}  "
            f"bias {self.median_signed_error:>+7.1%}  "
            f"within +/-50% {self.within_50pct:>5.0%}"
        )


def score(name: str, predictions: list[tuple[float, float]]) -> Score:
    """Score (predicted, actual) pairs on the metrics used throughout."""
    errors = [(p - a) / a for p, a in predictions if a > 0]
    if not errors:
        raise ValueError(f"{name}: nothing to score")
    return Score(
        name=name,
        n=len(errors),
        median_abs_error=statistics.median(abs(e) for e in errors),
        median_signed_error=statistics.median(errors),
        within_50pct=sum(1 for e in errors if abs(e) <= 0.5) / len(errors),
    )


@dataclass
class Comparison:
    """Paired bootstrap of two predictors' median absolute error."""

    label: str
    median_difference: float  # positive => the first predictor is worse
    ci_low: float
    ci_high: float
    probability_worse: float

    @property
    def is_separable(self) -> bool:
        """Whether the holdout can actually tell the two predictors apart."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def describe(self) -> str:
        verdict = "separable" if self.is_separable else "INDISTINGUISHABLE"
        return (
            f"{self.label:<38} {self.median_difference:+.1%}  "
            f"95% CI [{self.ci_low:+.1%}, {self.ci_high:+.1%}]  "
            f"P(worse)={self.probability_worse:.0%}  {verdict}"
        )


def compare(
    label: str,
    predict_a: Callable[[Observation], float],
    predict_b: Callable[[Observation], float],
    observations: list[Observation],
    trials: int = 4000,
    seed: int = 7,
) -> Comparison:
    """Is predictor A really worse than B, or is the holdout just small?

    Resamples the *same* deals for both predictors on every draw, so the
    comparison is not confounded by which deals happen to be picked. With a
    holdout of a few dozen deals a point-estimate ranking is close to
    meaningless: this is what distinguishes "A loses to B" from "this sample
    cannot separate them", and only the former belongs in a write-up.
    """
    rng = random.Random(seed)
    usable = [
        o
        for o in observations
        if o.disclosed_synergy > 0 and _safe(predict_a, o) and _safe(predict_b, o)
    ]
    if len(usable) < 2:
        raise ValueError(f"{label}: not enough comparable observations")

    differences = []
    for _ in range(trials):
        sample = [usable[rng.randrange(len(usable))] for _ in usable]
        differences.append(_median_abs_error(predict_a, sample) - _median_abs_error(predict_b, sample))

    differences.sort()
    return Comparison(
        label=label,
        median_difference=statistics.median(differences),
        ci_low=differences[int(0.025 * trials)],
        ci_high=differences[int(0.975 * trials)],
        probability_worse=sum(1 for d in differences if d > 0) / trials,
    )


def _safe(predict: Callable[[Observation], float], observation: Observation) -> float | None:
    try:
        value = predict(observation)
    except (TypeError, ZeroDivisionError):
        return None
    return value if value and value > 0 else None


def _median_abs_error(predict: Callable[[Observation], float], sample: list[Observation]) -> float:
    errors = [
        abs((predict(o) - o.disclosed_synergy) / o.disclosed_synergy)
        for o in sample
        if _safe(predict, o)
    ]
    return statistics.median(errors) if errors else float("inf")


def score_baselines(baselines: list[Baseline], observations: list[Observation]) -> list[Score]:
    scores = []
    for baseline in baselines:
        pairs = []
        for observation in observations:
            try:
                predicted = baseline.predict(observation)
            except (TypeError, ZeroDivisionError):
                continue
            if predicted and predicted > 0:
                pairs.append((predicted, observation.disclosed_synergy))
        if pairs:
            scores.append(score(baseline.name, pairs))
    return scores
