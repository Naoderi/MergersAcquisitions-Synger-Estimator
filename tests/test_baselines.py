import math
import random

import pytest

from synergy_estimator.calibration.baselines import (
    INCUMBENT,
    compare,
    fit_baselines,
    score,
    score_baselines,
)
from synergy_estimator.calibration.fit import Observation


def _obs(deal_id, sga, cogs, synergy, revenue=None, target_opex=None, year=2020):
    return Observation(
        deal_id=deal_id,
        combined_sga=sga,
        combined_cogs=cogs,
        disclosed_synergy=synergy,
        sector="software_services",
        announcement_year=year,
        combined_revenue=revenue,
        target_opex=target_opex,
    )


def _corpus(n=60, rate=0.02, seed=0, noise=0.0):
    """Deals where synergies really are `rate` x combined revenue."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        revenue = rng.uniform(1_000e6, 20_000e6)
        opex = revenue * 0.8
        rows.append(
            _obs(
                f"d{i}",
                sga=revenue * 0.15,
                cogs=revenue * 0.6,
                synergy=revenue * rate * (1 + rng.uniform(-noise, noise)),
                revenue=revenue,
                target_opex=opex,
            )
        )
    return rows


def test_recovers_the_true_percentage_of_revenue():
    baselines = fit_baselines(_corpus(rate=0.02))

    revenue_baseline = next(b for b in baselines if b.name == "pct_of_combined_revenue")
    assert revenue_baseline.parameter == pytest.approx(0.02, rel=0.02)


def test_constant_baseline_is_the_geometric_mean_of_targets():
    train = [_obs("a", 1, 1, 100.0), _obs("b", 1, 1, 400.0)]

    constant = next(b for b in fit_baselines(train) if b.name == "constant")

    assert constant.parameter == pytest.approx(math.sqrt(100.0 * 400.0))


def test_a_perfectly_specified_baseline_scores_near_zero_error():
    corpus = _corpus(rate=0.02)
    baselines = fit_baselines(corpus)

    scores = {s.name: s for s in score_baselines(baselines, corpus)}

    assert scores["pct_of_combined_revenue"].median_abs_error < 0.01
    assert scores["pct_of_combined_revenue"].within_50pct == 1.0


def test_constant_baseline_is_beaten_when_drivers_carry_signal():
    """If a single number for every deal scores as well as a driver-based
    predictor, the drivers carry nothing -- this is the sanity check for that."""
    corpus = _corpus(rate=0.02)
    baselines = fit_baselines(corpus)

    scores = {s.name: s for s in score_baselines(baselines, corpus)}

    assert scores["pct_of_combined_revenue"].median_abs_error < scores["constant"].median_abs_error


def test_baseline_parameters_are_fitted_on_train_only():
    """Fitting a comparator on the holdout would rig the comparison in its
    favour; a train-fitted parameter must not be re-derived from test data."""
    train = _corpus(n=40, rate=0.02, seed=1)
    test = _corpus(n=40, rate=0.05, seed=2)  # deliberately different regime

    fitted = next(b for b in fit_baselines(train) if b.name == "pct_of_combined_revenue")
    (test_score,) = [s for s in score_baselines([fitted], test) if s.name == "pct_of_combined_revenue"]

    assert fitted.parameter == pytest.approx(0.02, rel=0.02)
    # Applying the 2% train rate to a 5% regime must show up as large error.
    assert test_score.median_abs_error > 0.5


def test_incumbent_baseline_uses_the_shipping_defaults():
    incumbent = next(b for b in fit_baselines(_corpus()) if b.name == "incumbent_defaults")
    observation = _obs("x", sga=1_000e6, cogs=5_000e6, synergy=1.0)

    assert incumbent.predict(observation) == pytest.approx(
        INCUMBENT.sga_overlap_rate * 1_000e6 + INCUMBENT.cogs_overlap_rate * 5_000e6 + INCUMBENT.fixed_cost
    )


def test_revenue_baseline_is_omitted_when_the_corpus_lacks_revenue():
    """Reported as unavailable rather than silently faked from another driver."""
    without_revenue = [_obs(f"d{i}", 100.0, 500.0, 10.0, revenue=None, target_opex=600.0) for i in range(5)]

    names = {b.name for b in fit_baselines(without_revenue)}

    assert "pct_of_combined_revenue" not in names
    assert "pct_of_target_opex" in names


def test_score_reports_signed_bias_separately_from_magnitude():
    uniformly_low = [(50.0, 100.0), (25.0, 50.0), (10.0, 20.0)]

    result = score("low", uniformly_low)

    assert result.median_abs_error == pytest.approx(0.5)
    assert result.median_signed_error == pytest.approx(-0.5)
    assert result.within_50pct == 1.0


def test_score_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="nothing to score"):
        score("empty", [])


def test_compare_flags_a_small_holdout_as_indistinguishable():
    """A point-estimate ranking on a few dozen deals is close to meaningless;
    the write-up may only claim 'A loses to B' when the CI excludes zero."""
    corpus = _corpus(n=30, rate=0.02, seed=3, noise=0.8)
    barely_different = lambda o: 0.021 * (o.combined_revenue or 0.0)
    truth = lambda o: 0.02 * (o.combined_revenue or 0.0)

    result = compare("near-identical", barely_different, truth, corpus)

    assert not result.is_separable
    assert result.ci_low <= 0.0 <= result.ci_high


def test_compare_detects_a_genuinely_worse_predictor():
    corpus = _corpus(n=60, rate=0.02, seed=4)
    way_off = lambda o: 0.2 * (o.combined_revenue or 0.0)  # 10x too high
    truth = lambda o: 0.02 * (o.combined_revenue or 0.0)

    result = compare("10x too high", way_off, truth, corpus)

    assert result.is_separable
    assert result.median_difference > 0
    assert result.probability_worse > 0.95


def test_compare_is_paired_on_the_same_resampled_deals():
    """Comparing two predictors on independently drawn samples would add noise
    that has nothing to do with the predictors."""
    corpus = _corpus(n=40, seed=5)
    same = lambda o: 0.02 * (o.combined_revenue or 0.0)

    result = compare("identical", same, same, corpus)

    assert result.median_difference == pytest.approx(0.0, abs=1e-9)


def test_compare_needs_comparable_observations():
    with pytest.raises(ValueError, match="not enough comparable"):
        compare("empty", lambda o: 1.0, lambda o: 1.0, [])
