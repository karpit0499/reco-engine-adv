from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from reco_benchmark.models import (
    ALSModel,
    BPRModel,
    ItemKNNModel,
    PopularityModel,
    RetrievalRerankerModel,
    TwoTowerModel,
)
from reco_benchmark.models.baselines import _prune_top_k


@pytest.fixture()
def interactions() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.ones(12, dtype=np.float32),
            (
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5],
            ),
        ),
        shape=(5, 6),
    )


def test_all_models_score_the_full_catalog(interactions) -> None:
    popularity = PopularityModel().fit(interactions, 11)
    knn = ItemKNNModel(neighbors=3).fit(interactions, 11)
    als = ALSModel(factors=3, iterations=2).fit(interactions, 11)
    bpr = BPRModel(factors=4, epochs=1, batch_size=4).fit(interactions, 11)
    tower = TwoTowerModel(
        factors=4, epochs=1, batch_size=4, negatives=2
    ).fit(interactions, 11)
    reranker = RetrievalRerankerModel(
        retrieval=als,
        item_knn=knn,
        retrieval_k=3,
        positive_samples_per_user=2,
        negative_ratio=1,
    ).fit(interactions, 11)
    for model in (popularity, knn, als, bpr, tower):
        scores = model.score_users(np.asarray([0, 2]))
        assert scores.shape == (2, 6)
        assert np.all(np.isfinite(scores))
    reranked = reranker.score_users(np.asarray([0, 2]))
    assert reranked.shape == (2, 6)
    assert np.all(np.isfinite(reranked).sum(axis=1) == 3)
    assert not np.any(np.isfinite(reranked[0, [0, 1, 2]]))
    targets = np.asarray(
        [reranker._last_retrieval[0][0], reranker._last_retrieval[2][0]]
    )
    assert reranker.retrieval_target_recall(np.asarray([0, 2]), targets) == 1.0


def test_torch_models_are_deterministic(interactions) -> None:
    first = BPRModel(factors=4, epochs=1, batch_size=4).fit(interactions, 22)
    second = BPRModel(factors=4, epochs=1, batch_size=4).fit(interactions, 22)
    assert np.array_equal(first.user_factors, second.user_factors)
    assert np.array_equal(first.item_factors, second.item_factors)


def test_cold_users_use_popularity_without_random_embeddings(interactions) -> None:
    expected = np.asarray(interactions.sum(axis=0)).ravel().astype(np.float32)
    knn = ItemKNNModel(neighbors=3).fit(interactions, 11)
    als = ALSModel(factors=3, iterations=2).fit(interactions, 11)
    bpr = BPRModel(factors=4, epochs=1, batch_size=4).fit(interactions, 11)
    tower = TwoTowerModel(
        factors=4, epochs=1, batch_size=4, negatives=2
    ).fit(interactions, 11)
    for model in (knn, als, bpr, tower):
        assert np.array_equal(model.score_users(np.asarray([4]))[0], expected)
    assert np.count_nonzero(als.user_factors[4]) == 0
    assert np.count_nonzero(bpr.user_factors[4]) == 0
    assert np.count_nonzero(tower.user_factors[4]) == 0
    reranker = RetrievalRerankerModel(
        retrieval=als,
        item_knn=knn,
        retrieval_k=3,
        positive_samples_per_user=2,
        negative_ratio=1,
    ).fit(interactions, 11)
    assert np.array_equal(reranker.score_users(np.asarray([4]))[0], expected)


def test_item_neighbor_pruning_breaks_similarity_ties_by_item_id() -> None:
    similarities = sparse.csr_matrix(
        (np.ones(6, dtype=np.float32), ([0] * 6, np.arange(6))), shape=(1, 6)
    )
    pruned = _prune_top_k(similarities, 3)
    assert pruned.indices.tolist() == [0, 1, 2]
