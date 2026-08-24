from __future__ import annotations

import numpy as np

from .evaluation import PerUserEvaluation


def _bootstrap_draws(
    evaluation: PerUserEvaluation,
    replicates: int,
    seed: int,
    mask: np.ndarray | None = None,
    chunk_size: int = 50,
) -> dict[str, np.ndarray]:
    if mask is None:
        mask = np.ones(evaluation.recommendations.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (evaluation.recommendations.shape[0],):
        raise ValueError("bootstrap mask has the wrong shape")
    users = int(np.count_nonzero(mask))
    if users == 0 or replicates <= 0:
        raise ValueError("bootstrap needs users and positive replicates")
    rng = np.random.default_rng(seed)
    metric_names = list(evaluation.values)
    draws: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
    probabilities = np.full(users, 1.0 / users, dtype=np.float64)

    remaining = replicates
    while remaining:
        size = min(chunk_size, remaining)
        weights = rng.multinomial(users, probabilities, size=size).astype(np.float64)
        for name in metric_names:
            draws[name].append(weights @ evaluation.values[name][mask] / users)
        remaining -= size
    return {name: np.concatenate(parts) for name, parts in draws.items()}


def confidence_intervals(
    evaluation: PerUserEvaluation,
    cutoffs: list[int],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    del cutoffs
    draws = _bootstrap_draws(evaluation, replicates, seed)
    alpha = (1.0 - confidence_level) / 2.0
    result = {}
    for name, values in draws.items():
        low, high = np.quantile(values, [alpha, 1.0 - alpha])
        result[name] = {
            "estimate": float(evaluation.point_metrics[name]),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return result


def paired_differences(
    evaluation: PerUserEvaluation,
    baseline: PerUserEvaluation,
    cutoffs: list[int],
    replicates: int,
    seed: int,
    confidence_level: float,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if evaluation.recommendations.shape[0] != baseline.recommendations.shape[0]:
        raise ValueError("paired bootstrap evaluations have different user cohorts")
    del cutoffs
    if mask is None:
        mask = np.ones(evaluation.recommendations.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    model_draws = _bootstrap_draws(evaluation, replicates, seed, mask=mask)
    baseline_draws = _bootstrap_draws(baseline, replicates, seed, mask=mask)
    alpha = (1.0 - confidence_level) / 2.0
    result = {}
    for name in model_draws:
        difference = model_draws[name] - baseline_draws[name]
        low, high = np.quantile(difference, [alpha, 1.0 - alpha])
        result[name] = {
            "estimate": float(
                evaluation.point_metrics[name] - baseline.point_metrics[name]
            )
            if np.all(mask)
            else float(
                evaluation.values[name][mask].mean()
                - baseline.values[name][mask].mean()
            ),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return result
