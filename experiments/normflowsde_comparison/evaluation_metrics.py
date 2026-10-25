"""Sample-based accuracy and uncertainty scores for the frozen comparison.

Inputs contain future states only: ``samples[E,M,H,D]`` and ``truth[E,H,D]``.
The empirical marginal CRPS is exact.  The multivariate energy score applies to
the D-dimensional state at each time, so it also has a valid interpretation for
NF-DPF's marginal draws, which are not coherent sampled trajectories.

Only one trajectory is promoted to float64 at a time.  This keeps the evaluation
of several hundred posterior draws practical without duplicating every run's
complete sample tensor.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.distance import pdist


DEFAULT_CALIBRATION_LEVELS = tuple(np.arange(1, 10) / 10) + (0.95,)


def transform_metric_space(samples, truth, *, metric_space="normalized",
                           state_mean=None, state_std=None):
    """Return arrays in the documented score space; normalized input is expected.

    This helper is convenient for plotting a selected trajectory.  The scoring
    function calls it on individual trajectories rather than an entire run.
    """
    samples, truth = np.asarray(samples), np.asarray(truth)
    if metric_space == "normalized":
        return samples, truth
    if metric_space != "physical":
        raise ValueError("metric_space must be 'physical' or 'normalized'")
    if state_mean is None or state_std is None:
        raise ValueError("Physical metrics require state_mean and state_std")
    mean, std = np.asarray(state_mean), np.asarray(state_std)
    expected = (truth.shape[-1],)
    if mean.shape != expected or std.shape != expected:
        raise ValueError("State normalization vectors must match the state dimension")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("State normalization must be finite with strictly positive scales")
    return samples * std + mean, truth * std + mean


def _quantiles_sorted(sorted_samples, probabilities):
    """Linear empirical quantiles along the leading, already-sorted draw axis."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    location = probabilities * (len(sorted_samples) - 1)
    lower = np.floor(location).astype(int)
    upper = np.ceil(location).astype(int)
    fraction = (location - lower).reshape((-1,) + (1,) * (sorted_samples.ndim - 1))
    a, b = sorted_samples[lower], sorted_samples[upper]
    # Weighted sums a*(1-f)+b*f can differ by an ulp even when a==b.
    # Lerp from the closer endpoint preserves collapsed intervals exactly.
    difference = b - a
    return np.where(fraction < .5, a + difference * fraction,
                    b - difference * (1 - fraction))


def empirical_crps(samples, truth, *, axis=0):
    """Exact CRPS of equally weighted samples at every target coordinate.

    ``CRPS = mean|X-y| - sum_i(2*i-M-1)*X_(i)/M**2``, with one-based ranks.
    This is the empirical-distribution score, including diagonal sample pairs,
    rather than the distinct-pair (fair/unbiased-population) variant.
    """
    samples = np.moveaxis(np.asarray(samples, dtype=np.float64), axis, 0)
    truth = np.asarray(truth, dtype=np.float64)
    if not len(samples) or samples.shape[1:] != truth.shape:
        raise ValueError("Draw and truth shapes do not agree, or there are no draws")
    if not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise ValueError("Scoring requires finite samples and truth")
    ordered = np.sort(samples, axis=0)
    ranks = (2 * np.arange(1, len(samples) + 1) - len(samples) - 1).reshape(
        (-1,) + (1,) * truth.ndim)
    return np.abs(samples - truth).mean(axis=0) - np.sum(ranks * ordered, axis=0) / len(samples) ** 2


def interval_score(lower, upper, truth, *, coverage=0.9):
    """Central interval score: width plus alpha-scaled misses (lower is better)."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    lower, upper, truth = np.broadcast_arrays(lower, upper, truth)
    if not all(np.isfinite(value).all() for value in (lower, upper, truth)):
        raise ValueError("Interval scoring requires finite bounds and truth")
    if np.any(upper < lower):
        raise ValueError("Interval upper bounds must not be below lower bounds")
    alpha = 1 - coverage
    return (upper - lower + 2 / alpha * np.maximum(lower - truth, 0)
            + 2 / alpha * np.maximum(truth - upper, 0))


def _reliability_diagnostics(predictive_std, squared_error, bins):
    """Bin predicted scales and compare RMS std with RMSE in the same units."""
    std = np.asarray(predictive_std).ravel()
    error = np.asarray(squared_error).ravel()
    order = np.argsort(std, kind="stable")
    groups = np.array_split(order, min(bins, len(order)))
    return [dict(count=int(len(group)),
                 predicted_rms_std=float(np.sqrt(np.mean(std[group] ** 2))),
                 empirical_rmse=float(np.sqrt(np.mean(error[group]))),
                 minimum_predicted_std=float(std[group].min()),
                 maximum_predicted_std=float(std[group].max()))
            for group in groups]


def metrics_from_samples(samples, truth, *, observed_indices: Sequence[int] | None = None,
                         metric_space="normalized", state_mean=None, state_std=None,
                         energy_method="exact", energy_pairs=256, energy_seed=2026,
                         calibration_levels=DEFAULT_CALIBRATION_LEVELS,
                         pit_bins=10, reliability_bins=10):
    """Evaluate normalized posterior draws and return entirely JSON-safe values.

    ``rmse`` matches the training runner: compute each trajectory's RMSE over
    time and state, then average trajectories. ``pooled_rmse`` additionally
    reports the square root of the pooled squared error.  Every uncertainty
    score uses the same metric space as accuracy; dimensions with mixed physical
    units therefore remain normalized according to the frozen data protocol.

    The energy score averages Euclidean state-vector scores over windows/time.
    Its default pair term is also exact, using SciPy's condensed pair-distance
    vector separately for each future time. Optional ``energy_method='monte_carlo'``
    uses ``energy_pairs`` independent
    uniformly drawn distinct-index pairs, multiplied by (M-1)/M so it estimates
    the empirical-distribution pair expectation, including diagonal zeros.
    The reported MC standard error is for that run's mean energy score. It
    describes pair-integration error, not uncertainty across test trajectories.
    Identical pair indices across trajectories account for their covariance.
    """
    samples, truth = np.asarray(samples), np.asarray(truth)
    if samples.ndim != 4 or truth.ndim != 3:
        raise ValueError("Expected samples[E,M,H,D] and truth[E,H,D]")
    episodes, draws, horizon, dimensions = samples.shape
    if min(samples.shape) < 1 or truth.shape != (episodes, horizon, dimensions):
        raise ValueError("Samples/truth shapes must agree and have no empty dimensions")
    if not isinstance(energy_pairs, (int, np.integer)) or energy_pairs < 1:
        raise ValueError("energy_pairs must be a positive integer")
    if energy_method not in ("exact", "monte_carlo"):
        raise ValueError("energy_method must be 'exact' or 'monte_carlo'")
    if not isinstance(pit_bins, (int, np.integer)) or pit_bins < 1:
        raise ValueError("pit_bins must be a positive integer")
    if not isinstance(reliability_bins, (int, np.integer)) or reliability_bins < 1:
        raise ValueError("reliability_bins must be a positive integer")
    levels = np.asarray(calibration_levels, dtype=np.float64)
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or np.any(levels <= 0) or np.any(levels >= 1)
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("Calibration levels must be increasing and strictly between zero and one")
    if observed_indices is None:
        observed = np.arange(dimensions, dtype=int)
    else:
        raw_observed = np.asarray(observed_indices)
        if raw_observed.ndim != 1 or (raw_observed.size and raw_observed.dtype.kind not in "iu"):
            raise ValueError("observed_indices must be a one-dimensional integer sequence")
        observed = raw_observed.astype(int)
        if (np.any(observed < 0) or np.any(observed >= dimensions)
                or len(np.unique(observed)) != len(observed)):
            raise ValueError("Observed indices must be unique valid state dimensions")
    unobserved = np.setdiff1d(np.arange(dimensions), observed)

    rng = np.random.default_rng(energy_seed)
    if draws > 1:
        first_indices = rng.integers(draws, size=energy_pairs)
        second_indices = (first_indices + rng.integers(1, draws, size=energy_pairs)) % draws
    else:
        first_indices = second_indices = np.zeros(energy_pairs, dtype=int)
    pair_distance_sums = np.zeros(energy_pairs, dtype=np.float64)
    squared_error = np.empty(truth.shape, dtype=np.float64)
    predictive_std = np.empty(truth.shape, dtype=np.float64)
    pits = np.empty(truth.shape, dtype=np.float64)
    coverage_counts = np.zeros(len(levels), dtype=np.int64)
    dimension_coverage = np.zeros(dimensions, dtype=np.int64)
    dimension_width = np.zeros(dimensions)
    dimension_abs_error = np.zeros(dimensions)
    dimension_crps = np.zeros(dimensions)
    dimension_interval = np.zeros(dimensions)
    time_coverage = np.zeros(horizon, dtype=np.int64)
    time_width = np.zeros(horizon)
    residual_edges = np.linspace(-5, 5, 41)
    residual_counts = np.zeros(len(residual_edges) - 1, dtype=np.int64)
    residual_finite_count = residual_underflow = residual_overflow = 0
    residual_sum = residual_square_sum = 0.
    zero_variance_count = zero_variance_miss_count = 0
    trajectory_metrics = []
    maximum_absolute_posterior = maximum_absolute_truth = maximum_absolute_posterior_mean = 0.
    maximum_absolute_posterior_normalized = maximum_absolute_truth_normalized = 0.
    crps_total = interval_total = absolute_total = width_total = energy_first_total = 0.
    energy_pair_term_total = 0.
    inside_total = 0
    ranks = (2 * np.arange(1, draws + 1) - draws - 1).reshape((-1, 1, 1))
    probabilities = np.concatenate(([0.05, 0.95, 0.5], (1 - levels) / 2, (1 + levels) / 2))
    for index in range(episodes):
        maximum_absolute_posterior_normalized = max(
            maximum_absolute_posterior_normalized, float(np.max(np.abs(samples[index]))))
        maximum_absolute_truth_normalized = max(
            maximum_absolute_truth_normalized, float(np.max(np.abs(truth[index]))))
        current, target = transform_metric_space(
            np.asarray(samples[index], dtype=np.float64),
            np.asarray(truth[index], dtype=np.float64), metric_space=metric_space,
            state_mean=state_mean, state_std=state_std)
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError(f"Nonfinite samples or truth in trajectory {index}")
        mean = current.mean(axis=0)
        maximum_absolute_posterior = max(maximum_absolute_posterior, float(np.max(np.abs(current))))
        maximum_absolute_truth = max(maximum_absolute_truth, float(np.max(np.abs(target))))
        maximum_absolute_posterior_mean = max(maximum_absolute_posterior_mean, float(np.max(np.abs(mean))))
        error = mean - target
        square = error ** 2
        absolute = np.abs(error)
        squared_error[index] = square
        predictive_std[index] = current.std(axis=0, ddof=0)
        positive_scale = predictive_std[index] > 0
        standardized = error[positive_scale] / predictive_std[index][positive_scale]
        residual_counts += np.histogram(standardized, bins=residual_edges)[0]
        residual_finite_count += int(len(standardized))
        residual_underflow += int(np.sum(standardized < residual_edges[0]))
        residual_overflow += int(np.sum(standardized > residual_edges[-1]))
        residual_sum += standardized.sum()
        residual_square_sum += np.square(standardized).sum()
        zero_variance_count += int(np.sum(~positive_scale))
        zero_variance_miss_count += int(np.sum((~positive_scale) & (error != 0)))
        ordered = np.sort(current, axis=0)
        quantiles = _quantiles_sorted(ordered, probabilities)
        lower, upper = quantiles[:2]
        median_error = quantiles[2] - target
        inside = (target >= lower) & (target <= upper)
        widths = upper - lower
        intervals = interval_score(lower, upper, target)
        crps = np.abs(current - target).mean(axis=0) - np.sum(ranks * ordered, axis=0) / draws ** 2
        calibration_lower = quantiles[3:3 + len(levels)]
        calibration_upper = quantiles[3 + len(levels):]
        coverage_counts += np.sum((target >= calibration_lower) & (target <= calibration_upper), axis=(1, 2))
        pits[index] = (np.sum(current < target, axis=0) + 0.5 * np.sum(current == target, axis=0)) / draws
        first = np.linalg.norm(current - target, axis=-1).mean(axis=0)
        if energy_method == "exact":
            # pdist contains each distinct unordered pair once. Dividing its
            # sum by M**2 equals half of the full empirical pair expectation.
            energy_pair_term_total += sum(
                float(pdist(current[:, time, :], metric="euclidean").sum())
                for time in range(horizon)) / draws ** 2
        else:
            pairs = np.linalg.norm(current[first_indices] - current[second_indices], axis=-1)
            pair_distance_sums += pairs.sum(axis=1)
        energy_first_total += first.sum()
        crps_total += crps.sum()
        interval_total += intervals.sum()
        width_total += widths.sum()
        absolute_total += absolute.sum()
        inside_total += int(inside.sum())
        dimension_coverage += inside.sum(axis=0)
        dimension_width += widths.sum(axis=0)
        dimension_abs_error += absolute.sum(axis=0)
        dimension_crps += crps.sum(axis=0)
        dimension_interval += intervals.sum(axis=0)
        time_coverage += inside.sum(axis=1)
        time_width += widths.sum(axis=1)
        trajectory_metrics.append(dict(
            index=index, rmse=float(np.sqrt(square.mean())),
            posterior_median_rmse=float(np.sqrt(np.square(median_error).mean())),
            mae=float(absolute.mean()), coverage_90=float(inside.mean()),
            mean_interval_width=float(widths.mean()), crps=float(crps.mean()),
            interval_score_90=float(intervals.mean()),
            rmse_observed=float(np.sqrt(square[:, observed].mean())) if len(observed) else None,
            rmse_unobserved=float(np.sqrt(square[:, unobserved].mean())) if len(unobserved) else None))

    coordinate_count = episodes * horizon * dimensions
    vector_count = episodes * horizon
    coverage = coverage_counts / coordinate_count
    pair_estimates = pair_distance_sums / vector_count * (draws - 1) / draws
    if energy_method == "exact":
        energy = (energy_first_total - energy_pair_term_total) / vector_count
        energy_se = 0.
    else:
        energy = energy_first_total / vector_count - 0.5 * pair_estimates.mean()
        energy_se = (0.5 * pair_estimates.std(ddof=1) / np.sqrt(energy_pairs)
                     if energy_pairs > 1 and draws > 1 else 0. if draws == 1 else None)
    rms_std = float(np.sqrt(np.mean(predictive_std ** 2)))
    pooled_rmse = float(np.sqrt(squared_error.mean()))
    pit_counts, pit_edges = np.histogram(pits, bins=np.linspace(0, 1, pit_bins + 1))
    scalar_metrics = dict(
        rmse=float(np.mean([row["rmse"] for row in trajectory_metrics])),
        posterior_median_rmse=float(np.mean([row["posterior_median_rmse"] for row in trajectory_metrics])),
        pooled_rmse=pooled_rmse, mae=float(absolute_total / coordinate_count),
        rmse_observed=(float(np.mean([row["rmse_observed"] for row in trajectory_metrics]))
                       if len(observed) else None),
        rmse_unobserved=(float(np.mean([row["rmse_unobserved"] for row in trajectory_metrics]))
                         if len(unobserved) else None),
        crps=float(crps_total / coordinate_count), energy_score=float(energy),
        energy_score_mc_se=float(energy_se) if energy_se is not None else None,
        interval_score_90=float(interval_total / coordinate_count),
        coverage_90=float(inside_total / coordinate_count),
        coverage_90_absolute_error=float(abs(inside_total / coordinate_count - .9)),
        mean_interval_width=float(width_total / coordinate_count),
        calibration_mae=float(np.mean(np.abs(coverage - levels))),
        predictive_rms_std=rms_std, mean_predictive_std=float(predictive_std.mean()),
        rmse_to_predictive_rms_std=pooled_rmse / rms_std if rms_std > 0 else None,
        maximum_absolute_posterior=float(maximum_absolute_posterior),
        maximum_absolute_truth=float(maximum_absolute_truth),
        maximum_absolute_posterior_mean=float(maximum_absolute_posterior_mean),
        maximum_absolute_posterior_normalized=float(maximum_absolute_posterior_normalized),
        maximum_absolute_truth_normalized=float(maximum_absolute_truth_normalized),
        eval_windows=int(episodes), horizon=int(horizon), posterior_samples=int(draws),
        state_dimensions=int(dimensions))
    per_dimension = [dict(
        dimension=dimension, observed=bool(dimension in observed),
        rmse=float(np.sqrt(squared_error[..., dimension].mean())),
        mae=float(dimension_abs_error[dimension] / vector_count),
        coverage_90=float(dimension_coverage[dimension] / vector_count),
        mean_interval_width=float(dimension_width[dimension] / vector_count),
        crps=float(dimension_crps[dimension] / vector_count),
        interval_score_90=float(dimension_interval[dimension] / vector_count))
        for dimension in range(dimensions)]
    return dict(metrics=scalar_metrics, per_dimension=per_dimension,
                diagnostics=dict(
                    calibration=dict(nominal=levels.tolist(), empirical=coverage.tolist()),
                    pit=dict(bin_edges=pit_edges.tolist(), counts=pit_counts.tolist(),
                             frequencies=(pit_counts / coordinate_count).tolist(),
                             convention="midrank empirical CDF; ties receive half mass"),
                    reliability=_reliability_diagnostics(predictive_std, squared_error, reliability_bins),
                    reliability_by_group={
                        "observed": _reliability_diagnostics(predictive_std[..., observed], squared_error[..., observed], reliability_bins) if len(observed) else [],
                        "unobserved": _reliability_diagnostics(predictive_std[..., unobserved], squared_error[..., unobserved], reliability_bins) if len(unobserved) else []},
                    standardized_residual=dict(
                        bin_edges=residual_edges.tolist(), counts=residual_counts.tolist(),
                        count_with_positive_variance=int(residual_finite_count),
                        underflow=int(residual_underflow), overflow=int(residual_overflow),
                        zero_variance_count=int(zero_variance_count),
                        zero_variance_miss_count=int(zero_variance_miss_count),
                        mean=float(residual_sum / residual_finite_count) if residual_finite_count else None,
                        rms=float(np.sqrt(residual_square_sum / residual_finite_count)) if residual_finite_count else None,
                        convention="(posterior mean - truth) / empirical posterior std; zero-variance coordinates are counted separately; Gaussian reference is descriptive only"),
                    time=dict(rmse=np.sqrt(squared_error.mean(axis=(0, 2))).tolist(),
                              coverage_90=(time_coverage / (episodes * dimensions)).tolist(),
                              mean_interval_width=(time_width / (episodes * dimensions)).tolist()),
                    per_trajectory=trajectory_metrics,
                    energy_integration=dict(method=energy_method,
                                            seed=int(energy_seed) if energy_method == "monte_carlo" else None,
                                            pairs=int(energy_pairs) if energy_method == "monte_carlo" else int(draws * (draws - 1) // 2),
                                            total_exact_pair_distances=int(vector_count * draws * (draws - 1) // 2) if energy_method == "exact" else None,
                                            draws=int(draws), vector_count=int(vector_count),
                                            distinct_pair_correction=float((draws - 1) / draws))),
                definitions=dict(
                    metric_space=metric_space, initial_state_included=False,
                    rmse="Mean over test windows of sqrt(mean squared error over future times and states)",
                    posterior_median_rmse="Same mean-window RMSE convention using the coordinatewise posterior median, provided as an outlier-sensitivity diagnostic",
                    maximum_absolute_posterior="Maximum absolute draw coordinate in metric space, before any plot scaling or clipping",
                    maximum_absolute_posterior_normalized="Maximum absolute draw coordinate in the original normalized archive space",
                    pooled_rmse="sqrt(mean squared posterior-mean error over all windows, future times and states)",
                    crps="Exact empirical marginal CRPS, averaged over windows, future times and states",
                    energy_score="Empirical energy score of each D-dimensional future state; exact first term and " + ("exact condensed all-pairs term" if energy_method == "exact" else "deterministic Monte Carlo pair term"),
                    energy_score_mc_se="Zero for exact pair integration; otherwise Monte Carlo pair-integration standard error of the run mean, not test-set or training-seed variability",
                    coverage_90="Inclusive central empirical 5th–95th percentile interval coverage",
                    interval_score_90="Width + 20*(lower-truth) for lower misses + 20*(truth-upper) for upper misses",
                    calibration_mae="Mean absolute central-interval coverage error over the reported nominal levels",
                    predictive_std="Population standard deviation of the empirical posterior (ddof=0)",
                    reliability="Equal-count bins by predicted std; compare sqrt(mean predictive variance) against sqrt(mean posterior-mean squared error)",
                    per_dimension_rmse="Pooled RMSE within each state dimension",
                    uncertainty_scope="Marginal future-state uncertainty; no score assumes coherent posterior trajectories"))
