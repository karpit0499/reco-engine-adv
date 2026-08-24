from __future__ import annotations

import numpy as np
from scipy import sparse

from reco_benchmark.bootstrap import confidence_intervals, paired_differences
from reco_benchmark.dataset import Catalog, EvaluationData
from reco_benchmark.evaluation import evaluate_recommendations, recommend_full_catalog
from reco_benchmark.models.base import RecommenderModel


class FixedModel(RecommenderModel):
    name = "fixed"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def fit(self, interactions, seed):
        return self

    def score_users(self, user_indices):
        return self.scores[user_indices].copy()


def _data() -> EvaluationData:
    catalog = Catalog(
        movie_ids=np.arange(1, 8, dtype=np.int32),
        titles=tuple(str(value) for value in range(7)),
        genres=("g",) * 7,
        movie_to_index={value + 1: value for value in range(7)},
    )
    fit = sparse.csr_matrix(
        ([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 7), dtype=np.float32
    )
    # User 0 also rated item 1 below the relevance threshold. It must be masked even
    # though it is absent from the positive fitting matrix.
    seen = sparse.csr_matrix(
        ([1, 1, 1], ([0, 0, 1], [0, 1, 1])), shape=(2, 7), dtype=np.int8
    )
    return EvaluationData(
        stage="validation",
        catalog=catalog,
        training_user_ids=np.asarray([10, 20], dtype=np.int32),
        evaluation_user_indices=np.asarray([0, 1], dtype=np.int32),
        fit_matrix=fit,
        seen_matrix=seen,
        targets=np.asarray([2, 5], dtype=np.int32),
        target_timestamps=np.asarray([1, 2], dtype=np.int64),
        cold_users=np.asarray([False, True]),
        archive_sha256="x",
        protocol_hash="y",
        split_artifact_sha256="z",
        split_manifest_sha256="m",
        protocol={},
    )


def test_full_catalog_ranking_masks_all_seen_and_ties_by_movie_id() -> None:
    data = _data()
    model = FixedModel(
        [
            [100, 90, 5, 5, 4, 3, 2],
            [5, 100, 5, 4, 3, 2, 1],
        ]
    )
    recommendations = recommend_full_catalog(model, data, top_k=5, batch_users=2)
    assert recommendations[0].tolist() == [2, 3, 4, 5, 6]
    assert recommendations[1].tolist() == [0, 2, 3, 4, 5]
    assert 0 not in recommendations[0]
    assert 1 not in recommendations[0]
    assert 1 not in recommendations[1]


def test_score_tie_at_cutoff_uses_lowest_movie_ids() -> None:
    data = _data()
    model = FixedModel(np.ones((2, 7), dtype=np.float32))
    recommendations = recommend_full_catalog(model, data, top_k=2, batch_users=2)
    assert recommendations[0].tolist() == [2, 3]
    assert recommendations[1].tolist() == [0, 2]


def test_metrics_and_user_bootstrap_are_deterministic() -> None:
    data = _data()
    recommendations = np.asarray(
        [[2, 3, 4, 5, 6], [0, 2, 3, 4, 5]], dtype=np.int32
    )
    evaluation = evaluate_recommendations(recommendations, data, [1, 5])
    assert evaluation.point_metrics["recall_at_1"] == 0.5
    assert evaluation.point_metrics["recall_at_5"] == 1.0
    assert np.isclose(evaluation.point_metrics["mrr_at_5"], 0.6)
    assert np.isclose(evaluation.point_metrics["ndcg_at_5"], (1 + 1 / np.log2(6)) / 2)
    assert evaluation.point_metrics["catalog_coverage_at_5"] == 6 / 7

    first = confidence_intervals(evaluation, [1, 5], 100, 7, 0.95)
    second = confidence_intervals(evaluation, [1, 5], 100, 7, 0.95)
    assert first == second
    assert not any(name.startswith("catalog_coverage") for name in first)
    paired = paired_differences(evaluation, evaluation, [1, 5], 100, 7, 0.95)
    assert all(
        value == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        for value in paired.values()
    )


def test_paired_bootstrap_supports_declared_cold_cohort() -> None:
    data = _data()
    recommendations = np.asarray(
        [[2, 3, 4, 5, 6], [0, 2, 3, 4, 5]], dtype=np.int32
    )
    evaluation = evaluate_recommendations(recommendations, data, [1, 5])
    paired = paired_differences(
        evaluation,
        evaluation,
        [1, 5],
        100,
        7,
        0.95,
        mask=evaluation.cold_users,
    )
    assert paired["ndcg_at_5"] == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
