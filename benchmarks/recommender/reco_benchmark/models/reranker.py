from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import RecommenderModel
from .baselines import ItemKNNModel
from .factorization import ALSModel, _interaction_keys, sample_unseen_items


class RetrievalRerankerModel(RecommenderModel):
    name = "retrieval_reranker"

    def __init__(
        self,
        retrieval: ALSModel,
        item_knn: ItemKNNModel,
        retrieval_k: int = 200,
        positive_samples_per_user: int = 20,
        negative_ratio: int = 2,
        logistic_c: float = 1.0,
    ):
        self.retrieval = retrieval
        self.item_knn = item_knn
        self.retrieval_k = int(retrieval_k)
        self.positive_samples_per_user = int(positive_samples_per_user)
        self.negative_ratio = int(negative_ratio)
        self.logistic_c = float(logistic_c)
        if min(self.retrieval_k, self.positive_samples_per_user, self.negative_ratio) <= 0:
            raise ValueError("reranker sampling parameters must be positive")

    def _pair_features(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        als = np.sum(
            self.retrieval.user_factors[users] * self.retrieval.item_factors[items],
            axis=1,
        )
        knn = np.empty(users.size, dtype=np.float32)
        for user in np.unique(users):
            mask = users == user
            row_scores = (
                self.interactions[int(user)] @ self.item_knn.similarity
            ).toarray()[0]
            knn[mask] = row_scores[items[mask]]
        return np.column_stack(
            [
                als,
                knn,
                np.log1p(self.item_activity[items]),
                np.log1p(self.user_activity[users]),
            ]
        ).astype(np.float32)

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "RetrievalRerankerModel":
        self.interactions = interactions.tocsr().astype(np.float32)
        if self.retrieval_k > self.interactions.shape[1]:
            raise ValueError("retrieval_k exceeds catalog size")
        if not hasattr(self, "candidate_seen_matrix"):
            self.candidate_seen_matrix = self.interactions
        if not hasattr(self.retrieval, "user_factors"):
            self.retrieval.fit(self.interactions, seed)
        if not hasattr(self.item_knn, "similarity"):
            self.item_knn.fit(self.interactions, seed)
        self.item_activity = np.asarray(self.interactions.sum(axis=0)).ravel()
        self.user_activity = np.asarray(self.interactions.sum(axis=1)).ravel()
        seen_keys = _interaction_keys(self.interactions)
        rng = np.random.default_rng(seed)

        positive_users: list[int] = []
        positive_items: list[int] = []
        for user in range(self.interactions.shape[0]):
            start, end = self.interactions.indptr[user : user + 2]
            items = self.interactions.indices[start:end]
            if items.size > self.positive_samples_per_user:
                selected = rng.choice(
                    items.size, size=self.positive_samples_per_user, replace=False
                )
                items = items[np.sort(selected)]
            positive_users.extend([user] * items.size)
            positive_items.extend(items.tolist())

        users = np.asarray(positive_users, dtype=np.int64)
        positives = np.asarray(positive_items, dtype=np.int64)
        negative_users = np.repeat(users, self.negative_ratio)
        negatives = sample_unseen_items(
            negative_users, self.interactions.shape[1], seen_keys, rng
        )
        train_users = np.concatenate([users, negative_users])
        train_items = np.concatenate([positives, negatives])
        labels = np.concatenate(
            [
                np.ones(users.size, dtype=np.int8),
                np.zeros(negative_users.size, dtype=np.int8),
            ]
        )
        features = self._pair_features(train_users, train_items)
        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.logistic_c,
                max_iter=500,
                random_state=seed,
                solver="lbfgs",
            ),
        )
        self.classifier.fit(features, labels)
        self._last_retrieval: dict[int, np.ndarray] = {}
        return self

    @staticmethod
    def _stable_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
        threshold = np.partition(scores, scores.size - top_k)[scores.size - top_k]
        better = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        candidates = np.concatenate((better, tied[: top_k - better.size]))
        order = np.lexsort((candidates, -scores[candidates]))
        return candidates[order]

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        als_scores = self.retrieval.score_users(user_indices).astype(np.float32)
        knn_scores = self.item_knn.score_users(user_indices).astype(np.float32)
        user_feature = np.log1p(self.user_activity[user_indices]).astype(np.float32)
        item_feature = np.log1p(self.item_activity).astype(np.float32)
        output = np.full_like(als_scores, -np.inf, dtype=np.float32)
        for local, user in enumerate(user_indices):
            retrieval_scores = als_scores[local].copy()
            row_start, row_end = self.candidate_seen_matrix.indptr[user : user + 2]
            retrieval_scores[
                self.candidate_seen_matrix.indices[row_start:row_end]
            ] = -np.inf
            candidates = self._stable_top_k(retrieval_scores, self.retrieval_k)
            if np.any(~np.isfinite(retrieval_scores[candidates])):
                raise ValueError("retrieval stage has fewer unseen items than retrieval_k")
            self._last_retrieval[int(user)] = candidates.copy()
            if self.user_activity[int(user)] == 0:
                # Cold users have no learned user representation. Honor the protocol's
                # fallback exactly instead of letting the learned reranker happen to
                # approximate a popularity ordering.
                output[local] = self.item_activity
                continue
            features = np.column_stack(
                [
                    als_scores[local, candidates],
                    knn_scores[local, candidates],
                    item_feature[candidates],
                    np.full(candidates.size, user_feature[local], dtype=np.float32),
                ]
            )
            output[local, candidates] = self.classifier.decision_function(features).astype(
                np.float32
            )
        return output

    def retrieval_target_recall(
        self, user_indices: np.ndarray, targets: np.ndarray
    ) -> float:
        hits = [
            int(target) in self._last_retrieval[int(user)]
            for user, target in zip(user_indices, targets, strict=True)
        ]
        return float(np.mean(hits))
