from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import RecommenderModel


class PopularityModel(RecommenderModel):
    name = "popularity"

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "PopularityModel":
        del seed
        self.interactions = interactions.tocsr()
        self.scores = np.asarray(self.interactions.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.scores, (len(user_indices), self.scores.size)).copy()


def _prune_top_k(matrix: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        indices = matrix.indices[start:end]
        data = matrix.data[start:end]
        order = np.lexsort((indices, -data))
        if order.size > top_k:
            order = order[:top_k]
        rows.extend([row] * len(order))
        columns.extend(indices[order].tolist())
        values.extend(data[order].astype(float).tolist())
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=matrix.shape,
        dtype=np.float32,
    )


class ItemKNNModel(RecommenderModel):
    name = "item_knn"

    def __init__(self, neighbors: int = 100):
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        self.neighbors = int(neighbors)

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "ItemKNNModel":
        del seed
        self.interactions = interactions.tocsr().astype(np.float32)
        self.item_popularity = np.asarray(self.interactions.sum(axis=0)).ravel().astype(
            np.float32
        )
        item_user = self.interactions.T.tocsr()
        norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
        inverse = np.divide(
            1.0,
            norms,
            out=np.zeros_like(norms, dtype=np.float32),
            where=norms > 0,
        )
        normalized = sparse.diags(inverse) @ item_user
        similarity = (normalized @ normalized.T).tocsr().astype(np.float32)
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity = _prune_top_k(similarity, self.neighbors)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        scores = (self.interactions[user_indices] @ self.similarity).toarray().astype(
            np.float32, copy=False
        )
        cold = np.asarray(self.interactions[user_indices].sum(axis=1)).ravel() == 0
        scores[cold] = self.item_popularity
        return scores
