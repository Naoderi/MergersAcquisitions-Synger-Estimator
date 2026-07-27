import json
import random

import pytest

from synergy_estimator.calibration.fit import (
    Coefficients,
    Observation,
    fit_by_sector,
    fit_coefficients,
    percentage_errors,
    split_by_year,
    to_params_file,
)


def _synthetic(n, sga_rate, cogs_rate, fixed, sector="software_services", year=2020, seed=0, noise=0.0):
    """Deals generated from known coefficients, so the fit has a right answer."""
    rng = random.Random(seed)
    observations = []
    for i in range(n):
        sga = rng.uniform(100e6, 5_000e6)
        cogs = rng.uniform(200e6, 10_000e6)
        truth = sga_rate * sga + cogs_rate * cogs + fixed
        observations.append(
            Observation(
                deal_id=f"{sector}-{i}",
                combined_sga=sga,
                combined_cogs=cogs,
                disclosed_synergy=truth * (1 + rng.uniform(-noise, noise)),
                sector=sector,
                announcement_year=year,
            )
        )
    return observations


def test_recovers_known_coefficients_from_clean_data():
    observations = _synthetic(200, sga_rate=0.08, cogs_rate=0.01, fixed=20e6)

    fit = fit_coefficients(observations)

    assert fit.sga_overlap_rate == pytest.approx(0.08, rel=0.05)
    assert fit.cogs_overlap_rate == pytest.approx(0.01, rel=0.05)
    assert fit.fixed_cost == pytest.approx(20e6, rel=0.15)


def test_recovers_coefficients_approximately_under_noise():
    observations = _synthetic(300, sga_rate=0.06, cogs_rate=0.005, fixed=15e6, noise=0.25)

    fit = fit_coefficients(observations)

    assert fit.sga_overlap_rate == pytest.approx(0.06, rel=0.25)
    assert fit.cogs_overlap_rate == pytest.approx(0.005, rel=0.6)


def test_coefficients_are_never_negative():
    """A negative overlap rate would mean merging increases SG&A."""
    observations = _synthetic(60, sga_rate=0.0, cogs_rate=0.02, fixed=5e6, noise=0.5, seed=7)

    fit = fit_coefficients(observations)

    assert fit.sga_overlap_rate >= 0
    assert fit.cogs_overlap_rate >= 0
    assert fit.fixed_cost >= 0


def test_log_space_loss_is_not_dominated_by_megadeals():
    """One huge deal must not set the coefficients for everyone. In dollar
    space its squared residual would swamp the rest of the sample."""
    small = _synthetic(40, sga_rate=0.05, cogs_rate=0.0, fixed=0.0, seed=1)
    whale = [
        Observation("whale", combined_sga=500_000e6, combined_cogs=0.0,
                    disclosed_synergy=500_000e6 * 0.5, sector="software_services",
                    announcement_year=2020)
    ]

    fit = fit_coefficients(small + whale)

    # Still near the 5% the 40 ordinary deals imply, not dragged toward 50%.
    assert fit.sga_overlap_rate < 0.15


def test_fit_by_sector_separates_genuinely_different_sectors():
    """Grocery SG&A is store labor that survives a merger; software SG&A is
    corporate overhead that does not."""
    grocery = _synthetic(60, 0.01, 0.001, 5e6, sector="retail_grocery", seed=1)
    software = _synthetic(60, 0.15, 0.02, 25e6, sector="software_services", seed=2)

    _, fits = fit_by_sector(grocery + software)

    assert fits["retail_grocery"].shrunk.sga_overlap_rate < fits["software_services"].shrunk.sga_overlap_rate


def test_thin_sectors_are_shrunk_toward_the_global_fit():
    """A three-deal coefficient reported unshrunk repeats the two-deal
    calibration mistake this recalibration exists to correct."""
    big = _synthetic(100, 0.05, 0.005, 10e6, sector="software_services", seed=1)
    thin = _synthetic(3, 0.40, 0.05, 50e6, sector="lodging", seed=2)

    global_fit, fits = fit_by_sector(big + thin, shrinkage_k=5.0)

    thin_fit = fits["lodging"]
    assert thin_fit.shrinkage_weight == pytest.approx(3 / 8)
    # Shrunk estimate sits between its own raw fit and the pooled one.
    assert global_fit.sga_overlap_rate < thin_fit.shrunk.sga_overlap_rate < thin_fit.raw.sga_overlap_rate


def test_large_sectors_are_barely_shrunk():
    big = _synthetic(95, 0.05, 0.005, 10e6, sector="software_services", seed=3)

    _, fits = fit_by_sector(big, shrinkage_k=5.0)

    assert fits["software_services"].shrinkage_weight == pytest.approx(0.95)


def test_percentage_errors_are_near_zero_on_data_the_model_generated():
    observations = _synthetic(80, 0.07, 0.008, 12e6)
    global_fit, fits = fit_by_sector(observations)

    errors = percentage_errors(observations, global_fit, fits)

    assert max(abs(e) for e in errors) < 0.05


def test_percentage_errors_fall_back_to_the_global_fit_for_unseen_sectors():
    trained = _synthetic(50, 0.05, 0.005, 10e6, sector="software_services")
    global_fit, fits = fit_by_sector(trained)
    unseen = _synthetic(5, 0.05, 0.005, 10e6, sector="a_sector_never_trained_on", seed=9)

    errors = percentage_errors(unseen, global_fit, fits)

    assert len(errors) == 5


def test_split_by_year_is_chronological():
    observations = (
        _synthetic(5, 0.05, 0.005, 1e6, year=2021, seed=1)
        + _synthetic(5, 0.05, 0.005, 1e6, year=2024, seed=2)
    )

    train, test = split_by_year(observations, train_through=2022)

    assert {o.announcement_year for o in train} == {2021}
    assert {o.announcement_year for o in test} == {2024}


def test_fit_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        fit_coefficients([])


def test_fit_rejects_non_positive_targets():
    observations = _synthetic(5, 0.05, 0.005, 1e6)
    observations[0] = Observation("bad", 1e6, 1e6, 0.0, "software_services", 2020)

    with pytest.raises(ValueError, match="must be positive"):
        fit_coefficients(observations)


def test_params_file_contains_only_coefficients_and_counts(tmp_path):
    """Fitted numbers are publishable; the underlying corpus rows are a
    separate artifact and must not be embedded here."""
    observations = _synthetic(40, 0.05, 0.005, 10e6)
    global_fit, fits = fit_by_sector(observations)
    path = tmp_path / "fitted_params.json"

    to_params_file(global_fit, fits, path, metadata={"n_deals": 40})

    payload = json.loads(path.read_text())
    assert set(payload["sectors"]["software_services"]) == {
        "n", "shrinkage_weight", "sga_overlap_rate", "cogs_overlap_rate", "fixed_cost"
    }
    assert payload["n_deals"] == 40
    assert "deal_id" not in path.read_text()


def test_coefficients_predict_matches_the_model_formula():
    coefficients = Coefficients(sga_overlap_rate=0.1, cogs_overlap_rate=0.01, fixed_cost=5.0)

    assert coefficients.predict(100.0, 200.0) == pytest.approx(0.1 * 100 + 0.01 * 200 + 5.0)
