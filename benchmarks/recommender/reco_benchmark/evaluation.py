from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .dataset import EvaluationData
from .models import RecommenderModel


@dataclass(frozen=True)
class PerUserEvaluation:
    recommendations: np.ndarray
    values: dict[str, np.ndarray]
    point_metrics: dict[str, float]
    warm_targets: np.ndarray
    cold_users: np.ndarray
    catalog_size: int


def _stable_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k > scores.size:
        raise ValueError("top_k must be between one and the catalog size")
    threshold = np.partition(scores, scores.size - top_k)[scores.size - top_k]
    better = np.flatnonzero(scores > threshold)
    # flatnonzero is index-ordered, so a score tie at the cutoff retains the
    # lowest catalog indices (the catalog itself is sorted by movie_id).
    tied = np.flatnonzero(scores == threshold)
    candidate = np.concatenate((better, tied[: top_k - better.size]))
    order = np.lexsort((candidate, -scores[candidate]))
    return candidate[order]


def recommend_full_catalog(
    model: RecommenderModel,
    data: EvaluationData,
    top_k: int,
    batch_users: int,
) -> np.ndarray:
    if batch_users <= 0:
        raise ValueError("batch_users must be positive")
    recommendations = np.empty((data.targets.size, top_k), dtype=np.int32)
    evaluation_users = data.evaluation_user_indices.astype(np.int64, copy=False)
    for start in range(0, evaluation_users.size, batch_users):
        users = evaluation_users[start : start + batch_users]
        scores = np.asarray(model.score_users(users), dtype=np.float32)
        expected = (users.size, data.catalog.size)
        if scores.shape != expected:
            raise ValueError(
                f"{model.name} returned score shape {scores.shape}, expected {expected}"
            )
        scores[~np.isfinite(scores)] = -np.inf
        for local, user in enumerate(users):
            row_start, row_end = data.seen_matrix.indptr[user : user + 2]
            seen = data.seen_matrix.indices[row_start:row_end]
            scores[local, seen] = -np.inf
            ranking = _stable_top_k(scores[local], top_k)
            if np.any(~np.isfinite(scores[local, ranking])):
                raise ValueError(f"{model.name} could not produce {top_k} unseen items")
            recommendations[start + local] = ranking
    return recommendations


def _metric_arrays(
    recommendations: np.ndarray, targets: np.ndarray, cutoffs: list[int]
) -> dict[str, np.ndarray]:
    matches = recommendations == targets[:, None]
    positions = np.argmax(matches, axis=1) + 1
    has_match = matches.any(axis=1)
    positions = np.where(has_match, positions, 0)
    values: dict[str, np.ndarray] = {}
    for cutoff in cutoffs:
        hit = (positions > 0) & (positions <= cutoff)
        values[f"recall_at_{cutoff}"] = hit.astype(np.float64)
        ndcg = np.zeros(targets.size, dtype=np.float64)
        reciprocal_rank = np.zeros(targets.size, dtype=np.float64)
        ndcg[hit] = 1.0 / np.log2(positions[hit] + 1)
        reciprocal_rank[hit] = 1.0 / positions[hit]
        values[f"ndcg_at_{cutoff}"] = ndcg
        values[f"mrr_at_{cutoff}"] = reciprocal_rank
        values[f"user_coverage_at_{cutoff}"] = np.ones(
            targets.size, dtype=np.float64
        )
    return values


def _coverage(recommendations: np.ndarray, cutoff: int, catalog_size: int) -> float:
    return float(np.unique(recommendations[:, :cutoff]).size / catalog_size)


def evaluate_recommendations(
    recommendations: np.ndarray,
    data: EvaluationData,
    cutoffs: list[int],
) -> PerUserEvaluation:
    if recommendations.shape != (data.targets.size, max(cutoffs)):
        raise ValueError("recommendation matrix has the wrong shape")
    values = _metric_arrays(recommendations, data.targets, cutoffs)
    point = {name: float(metric.mean()) for name, metric in values.items()}
    for cutoff in cutoffs:
        point[f"catalog_coverage_at_{cutoff}"] = _coverage(
            recommendations, cutoff, data.catalog.size
        )
    item_counts = np.asarray(data.fit_matrix.sum(axis=0)).ravel()
    warm = item_counts[data.targets] > 0
    return PerUserEvaluation(
        recommendations=recommendations,
        values=values,
        point_metrics=point,
        warm_targets=warm,
        cold_users=data.cold_users.copy(),
        catalog_size=data.catalog.size,
    )


def subgroup_metrics(evaluation: PerUserEvaluation, mask: np.ndarray) -> dict:
    result: dict[str, float | int | None] = {"users": int(np.count_nonzero(mask))}
    for name, values in evaluation.values.items():
        result[name] = float(values[mask].mean()) if np.any(mask) else None
    if np.any(mask):
        cutoffs = sorted(
            int(name.rsplit("_", 1)[1])
            for name in evaluation.values
            if name.startswith("recall_at_")
        )
        for cutoff in cutoffs:
            result[f"catalog_coverage_at_{cutoff}"] = _coverage(
                evaluation.recommendations[mask], cutoff, evaluation.catalog_size
            )
    else:
        for name in evaluation.point_metrics:
            if name.startswith("catalog_coverage_at_"):
                result[name] = None
    return result


def recommendation_incidence(
    recommendations: np.ndarray, cutoff: int, catalog_size: int
) -> sparse.csr_matrix:
    user_count = recommendations.shape[0]
    rows = np.repeat(np.arange(user_count, dtype=np.int32), cutoff)
    columns = recommendations[:, :cutoff].reshape(-1)
    values = np.ones(rows.size, dtype=np.int8)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(user_count, catalog_size), dtype=np.int8
    )
    matrix.data[:] = 1
    return matrix
