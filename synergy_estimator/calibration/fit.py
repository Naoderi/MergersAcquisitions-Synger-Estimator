"""Fit the cost-synergy overlap rates against a corpus of disclosed targets.

The model being fitted is the one in `models/synergies.py`:

    synergies = sga_rate * combined_SGA + cogs_rate * combined_COGS + fixed

Three choices matter more than the optimizer:

**Log space.** Disclosed targets span $10M to $3B+. Minimizing squared *dollar*
error would let a handful of megadeals set the coefficients for everyone. The
metric that is actually reported is percentage error, so the loss minimizes
squared log-ratio, which is symmetric in relative terms: predicting 2x and
predicting half are penalized equally.

**Non-negativity.** A negative overlap rate would fit some samples better while
meaning "merging increases SG&A", which is not a claim this model should make.

**Shrinkage.** Per-sector fits are the whole point -- grocery and software
genuinely differ -- but some sectors have three deals. An unshrunk three-deal
coefficient repeats the exact mistake this recalibration exists to correct, so
each sector is blended toward the global fit with weight n/(n+k).
"""

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Weight on the global fit: a sector needs ~k deals before its own estimate
# carries as much weight as the pooled one.
DEFAULT_SHRINKAGE_K = 5.0

# Floor for predictions inside the log, so a degenerate all-zero parameter
# vector cannot produce log(0) during the search. Applied to *scaled*
# predictions (which are O(1)), so it must stay far below any real value.
_EPSILON = 1e-12


@dataclass(frozen=True)
class Observation:
    """One deal: the model's inputs plus the target it should reproduce."""

    deal_id: str
    combined_sga: float
    combined_cogs: float
    disclosed_synergy: float
    sector: str
    announcement_year: int
    # Not used by the synergy model, which is a function of SG&A and COGS only.
    # Carried so the naive baselines in `baselines.py` can be scored on exactly
    # the same rows rather than a differently-filtered sample.
    combined_revenue: float | None = None
    target_opex: float | None = None


@dataclass
class Coefficients:
    sga_overlap_rate: float
    cogs_overlap_rate: float
    fixed_cost: float

    def predict(self, sga: float, cogs: float) -> float:
        return self.sga_overlap_rate * sga + self.cogs_overlap_rate * cogs + self.fixed_cost


@dataclass
class SectorFit:
    sector: str
    n: int
    raw: Coefficients
    shrunk: Coefficients
    shrinkage_weight: float  # n / (n + k); 0 = fully pooled, 1 = fully own-sector


def _loss(params: np.ndarray, sga: np.ndarray, cogs: np.ndarray, log_actual: np.ndarray) -> float:
    predicted = params[0] * sga + params[1] * cogs + params[2]
    return float(np.mean((np.log(np.maximum(predicted, _EPSILON)) - log_actual) ** 2))


def fit_coefficients(observations: list[Observation]) -> Coefficients:
    """Least-squares fit in log space, constrained to non-negative coefficients.

    The optimization runs on scaled inputs. Unscaled, the two overlap rates are
    O(0.01) while the fixed cost is O(1e7), and L-BFGS-B -- which uses a single
    convergence tolerance across all coordinates -- terminates on the rates
    long before the fixed cost has moved. Dividing every dollar quantity by the
    median target puts all three parameters within a couple of orders of
    magnitude of each other. Scaling cancels out of the rates (they are
    dimensionless ratios); only the fixed cost is rescaled on the way out.
    """
    if not observations:
        raise ValueError("cannot fit coefficients on an empty sample")

    sga = np.array([o.combined_sga for o in observations], dtype=float)
    cogs = np.array([o.combined_cogs for o in observations], dtype=float)
    actual = np.array([o.disclosed_synergy for o in observations], dtype=float)
    if np.any(actual <= 0):
        raise ValueError("disclosed synergies must be positive to fit in log space")

    scale = float(np.median(actual))
    # Start from the current hand-calibrated defaults so the search begins
    # somewhere plausible rather than at the origin.
    start = np.array([0.015, 0.0015, 15_000_000.0 / scale])
    result = minimize(
        _loss,
        start,
        args=(sga / scale, cogs / scale, np.log(actual / scale)),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None), (0.0, None)],
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20_000},
    )
    sga_rate, cogs_rate, fixed_scaled = (float(x) for x in result.x)
    return Coefficients(sga_rate, cogs_rate, fixed_scaled * scale)


def _blend(sector: Coefficients, glob: Coefficients, weight: float) -> Coefficients:
    return Coefficients(
        sga_overlap_rate=weight * sector.sga_overlap_rate + (1 - weight) * glob.sga_overlap_rate,
        cogs_overlap_rate=weight * sector.cogs_overlap_rate + (1 - weight) * glob.cogs_overlap_rate,
        fixed_cost=weight * sector.fixed_cost + (1 - weight) * glob.fixed_cost,
    )


def fit_by_sector(
    observations: list[Observation], shrinkage_k: float = DEFAULT_SHRINKAGE_K
) -> tuple[Coefficients, dict[str, SectorFit]]:
    """Fit a global model plus one shrunk model per sector."""
    global_fit = fit_coefficients(observations)

    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.sector, []).append(observation)

    fits: dict[str, SectorFit] = {}
    for sector, sample in grouped.items():
        raw = fit_coefficients(sample)
        weight = len(sample) / (len(sample) + shrinkage_k)
        fits[sector] = SectorFit(
            sector=sector,
            n=len(sample),
            raw=raw,
            shrunk=_blend(raw, global_fit, weight),
            shrinkage_weight=weight,
        )
    return global_fit, fits


def percentage_errors(
    observations: list[Observation], global_fit: Coefficients, sector_fits: dict[str, SectorFit]
) -> list[float]:
    """Signed (predicted - actual) / actual, using each deal's sector fit."""
    errors = []
    for observation in observations:
        coefficients = (
            sector_fits[observation.sector].shrunk
            if observation.sector in sector_fits
            else global_fit
        )
        predicted = coefficients.predict(observation.combined_sga, observation.combined_cogs)
        errors.append((predicted - observation.disclosed_synergy) / observation.disclosed_synergy)
    return errors


def split_by_year(
    observations: list[Observation], train_through: int
) -> tuple[list[Observation], list[Observation]]:
    """Time-based holdout.

    Chronological rather than random: the question a reader actually has is
    "does this generalize forward", and random k-fold would let a 2024 deal
    inform the coefficients used to score its own 2023 comparable.
    """
    train = [o for o in observations if o.announcement_year <= train_through]
    test = [o for o in observations if o.announcement_year > train_through]
    return train, test


def to_params_file(
    global_fit: Coefficients,
    sector_fits: dict[str, SectorFit],
    path: Path,
    metadata: dict | None = None,
) -> None:
    """Write the fitted coefficients -- numbers only, no source data."""
    payload = {
        "fitted_at": date.today().isoformat(),
        "shrinkage_k": DEFAULT_SHRINKAGE_K,
        "global": asdict(global_fit),
        "sectors": {
            sector: {"n": fit.n, "shrinkage_weight": fit.shrinkage_weight, **asdict(fit.shrunk)}
            for sector, fit in sorted(sector_fits.items())
        },
        **(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
