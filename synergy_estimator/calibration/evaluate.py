"""Fit the model on the corpus and benchmark it against the rules of thumb.

Produces the table that decides whether the recalibration was worth doing:
the fitted model and four one-line heuristics, every parameter fitted on the
same training split, all scored on the same holdout.

    python -m synergy_estimator.calibration.evaluate
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from synergy_estimator.calibration.baselines import (
    Comparison,
    Score,
    compare,
    fit_baselines,
    score,
    score_baselines,
)
from synergy_estimator.calibration.fit import (
    Coefficients,
    Observation,
    SectorFit,
    fit_by_sector,
    split_by_year,
    to_params_file,
)

DEFAULT_CORPUS = Path("data/deals_corpus.csv")
DEFAULT_PARAMS = Path("synergy_estimator/calibration/fitted_params.json")
TRAIN_THROUGH = 2022


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_observations(path: Path = DEFAULT_CORPUS, cost_only: bool = False) -> list[Observation]:
    """Read the corpus CSV into fitter inputs.

    `cost_only` keeps just the deals whose disclosure explicitly says "cost
    synergies". The rest say only "synergies", which in practice usually means
    cost but is not stated -- worth reporting both ways rather than assuming.
    """
    observations = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if cost_only and row["synergy_type"] not in ("cost", "cost_and_revenue"):
                continue
            synergy = _float(row["disclosed_synergy_usd"])
            sga, cogs = _float(row["combined_sga"]), _float(row["combined_cogs"])
            if not synergy or sga is None or cogs is None:
                continue
            target_sga, target_cogs = _float(row["target_sga"]), _float(row["target_cogs"])
            observations.append(
                Observation(
                    deal_id=row["deal_id"],
                    combined_sga=sga,
                    combined_cogs=cogs,
                    disclosed_synergy=synergy,
                    sector=row["sector"],
                    announcement_year=int(row["announcement_date"][:4]),
                    combined_revenue=_float(row.get("combined_revenue", "")),
                    target_opex=(
                        target_sga + target_cogs
                        if target_sga is not None and target_cogs is not None
                        else None
                    ),
                )
            )
    return observations


def _score_model(
    name: str,
    observations: list[Observation],
    global_fit: Coefficients,
    sector_fits: dict[str, SectorFit],
) -> Score:
    pairs = []
    for observation in observations:
        fit = sector_fits.get(observation.sector)
        coefficients = fit.shrunk if fit else global_fit
        predicted = coefficients.predict(observation.combined_sga, observation.combined_cogs)
        if predicted > 0:
            pairs.append((predicted, observation.disclosed_synergy))
    return score(name, pairs)


@dataclass
class Evaluation:
    train_n: int
    test_n: int
    train_scores: list[Score]
    test_scores: list[Score]
    global_fit: Coefficients
    sector_fits: dict[str, SectorFit]
    comparisons: list[Comparison]


def evaluate(observations: list[Observation], train_through: int = TRAIN_THROUGH) -> Evaluation:
    train, test = split_by_year(observations, train_through)
    if not train or not test:
        raise ValueError("chronological split left one side empty")

    global_fit, sector_fits = fit_by_sector(train)
    baselines = fit_baselines(train)

    return Evaluation(
        train_n=len(train),
        test_n=len(test),
        train_scores=[
            _score_model("fitted model (sector)", train, global_fit, sector_fits),
            # Same model with sector fits switched off. If the pooled version
            # generalizes better, the per-sector split is fitting noise -- many
            # sectors have n=1 or 2, which shrinkage dampens but cannot fix.
            _score_model("fitted model (global)", train, global_fit, {}),
            *score_baselines(baselines, train),
        ],
        test_scores=[
            _score_model("fitted model (sector)", test, global_fit, sector_fits),
            _score_model("fitted model (global)", test, global_fit, {}),
            *score_baselines(baselines, test),
        ],
        global_fit=global_fit,
        sector_fits=sector_fits,
        comparisons=_compare_headline(test, global_fit, sector_fits, baselines),
    )


def _compare_headline(test, global_fit, sector_fits, baselines) -> list[Comparison]:
    """Bootstrap the rankings the write-up would otherwise assert from point
    estimates alone. On a holdout this size most gaps are not separable."""
    by_name = {b.name: b for b in baselines}
    sector = lambda o: (sector_fits[o.sector].shrunk if o.sector in sector_fits else global_fit).predict(
        o.combined_sga, o.combined_cogs
    )
    pooled = lambda o: global_fit.predict(o.combined_sga, o.combined_cogs)

    pairs = [("model(sector) vs model(global)", sector, pooled)]
    if "pct_of_combined_revenue" in by_name:
        pairs.append(("model(global) vs pct_of_revenue", pooled, by_name["pct_of_combined_revenue"].predict))
    pairs.append(("model(global) vs constant", pooled, by_name["constant"].predict))
    pairs.append(("incumbent vs model(global)", by_name["incumbent_defaults"].predict, pooled))

    results = []
    for label, a, b in pairs:
        try:
            results.append(compare(label, a, b, test))
        except ValueError:
            continue
    return results


def format_report(evaluation: Evaluation, label: str) -> str:
    lines = [
        f"\n{'=' * 78}",
        f"{label}   train (<={TRAIN_THROUGH}) n={evaluation.train_n}   "
        f"test (>{TRAIN_THROUGH}) n={evaluation.test_n}",
        "=" * 78,
        "\nIN SAMPLE",
        *(f"  {s.describe()}" for s in evaluation.train_scores),
        "\nOUT OF SAMPLE  <- this is the one that counts",
        *(f"  {s.describe()}" for s in evaluation.test_scores),
        "\nFITTED RATES BY SECTOR (shrunk toward the global fit)",
        f"  {'sector':<24} {'n':>3}  {'SG&A':>7}  {'COGS':>7}  {'fixed $M':>9}  {'weight':>6}",
    ]
    for sector, fit in sorted(evaluation.sector_fits.items(), key=lambda kv: -kv[1].n):
        lines.append(
            f"  {sector:<24} {fit.n:>3}  {fit.shrunk.sga_overlap_rate:>6.2%}  "
            f"{fit.shrunk.cogs_overlap_rate:>6.2%}  {fit.shrunk.fixed_cost / 1e6:>9,.0f}  "
            f"{fit.shrinkage_weight:>6.2f}"
        )
    g = evaluation.global_fit
    lines.append(
        f"  {'GLOBAL':<24} {'':>3}  {g.sga_overlap_rate:>6.2%}  "
        f"{g.cogs_overlap_rate:>6.2%}  {g.fixed_cost / 1e6:>9,.0f}"
    )
    if evaluation.comparisons:
        lines.append("\nIS THE RANKING REAL? (paired bootstrap on the holdout)")
        lines.extend(f"  {c.describe()}" for c in evaluation.comparisons)
    return "\n".join(lines)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--params-out", type=Path, default=DEFAULT_PARAMS)
    args = parser.parse_args()

    for label, cost_only in (("ALL DISCLOSURES", False), ("EXPLICIT COST SYNERGIES ONLY", True)):
        observations = load_observations(args.corpus, cost_only=cost_only)
        if len(observations) < 20:
            print(f"\n{label}: only {len(observations)} rows -- skipping")
            continue
        evaluation = evaluate(observations)
        print(format_report(evaluation, label))

        if not cost_only:
            to_params_file(
                evaluation.global_fit,
                evaluation.sector_fits,
                args.params_out,
                metadata={"n_deals": len(observations), "train_through": TRAIN_THROUGH},
            )
            print(f"\nfitted params -> {args.params_out}")


if __name__ == "__main__":
    _cli()
