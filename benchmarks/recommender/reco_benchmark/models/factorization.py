from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import sparse

from .base import RecommenderModel


def _interaction_keys(interactions: sparse.csr_matrix) -> np.ndarray:
    rows, columns = interactions.nonzero()
    keys = rows.astype(np.int64) * interactions.shape[1] + columns.astype(np.int64)
    keys.sort()
    return keys


def _contains_sorted(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, values)
    valid = positions < sorted_values.size
    result = np.zeros(values.shape, dtype=bool)
    result[valid] = sorted_values[positions[valid]] == values[valid]
    return result


def sample_unseen_items(
    users: np.ndarray,
    item_count: int,
    seen_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, item_count, size=users.size, dtype=np.int64)
    encoded = users.astype(np.int64) * item_count + negatives
    invalid = _contains_sorted(seen_keys, encoded)
    attempts = 0
    while np.any(invalid):
        attempts += 1
        if attempts > 100:
            raise RuntimeError("could not sample an unseen item")
        negatives[invalid] = rng.integers(
            0, item_count, size=int(np.count_nonzero(invalid)), dtype=np.int64
        )
        encoded[invalid] = users[invalid].astype(np.int64) * item_count + negatives[invalid]
        invalid = _contains_sorted(seen_keys, encoded)
    return negatives


class ALSModel(RecommenderModel):
    name = "als"

    def __init__(
        self,
        factors: int = 24,
        iterations: int = 4,
        alpha: float = 20.0,
        regularization: float = 0.05,
    ):
        self.factors = int(factors)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        if min(self.factors, self.iterations) <= 0:
            raise ValueError("ALS factors and iterations must be positive")

    def _least_squares(
        self, rows: sparse.csr_matrix, fixed: np.ndarray
    ) -> np.ndarray:
        output = np.zeros((rows.shape[0], self.factors), dtype=np.float32)
        gram = fixed.T @ fixed
        identity = np.eye(self.factors, dtype=np.float32) * self.regularization
        for row_index in range(rows.shape[0]):
            start, end = rows.indptr[row_index : row_index + 2]
            indices = rows.indices[start:end]
            if not indices.size:
                continue
            selected = fixed[indices]
            system = gram + self.alpha * (selected.T @ selected) + identity
            target = (1.0 + self.alpha) * selected.sum(axis=0)
            output[row_index] = np.linalg.solve(system, target).astype(np.float32)
        return output

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "ALSModel":
        self.interactions = interactions.tocsr().astype(np.float32)
        self.item_popularity = np.asarray(self.interactions.sum(axis=0)).ravel().astype(
            np.float32
        )
        item_rows = self.interactions.T.tocsr()
        rng = np.random.default_rng(seed)
        self.item_factors = rng.normal(
            0.0, 0.01, size=(self.interactions.shape[1], self.factors)
        ).astype(np.float32)
        self.user_factors = np.zeros(
            (self.interactions.shape[0], self.factors), dtype=np.float32
        )
        for _ in range(self.iterations):
            self.user_factors = self._least_squares(
                self.interactions, self.item_factors
            )
            self.item_factors = self._least_squares(item_rows, self.user_factors)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        scores = self.user_factors[user_indices] @ self.item_factors.T
        cold = np.asarray(self.interactions[user_indices].sum(axis=1)).ravel() == 0
        scores[cold] = self.item_popularity
        return scores


class _EmbeddingPair(torch.nn.Module):
    def __init__(self, users: int, items: int, factors: int):
        super().__init__()
        self.user = torch.nn.Embedding(users, factors)
        self.item = torch.nn.Embedding(items, factors)
        torch.nn.init.normal_(self.user.weight, std=0.05)
        torch.nn.init.normal_(self.item.weight, std=0.05)

    def dot(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(dim=1)


class _TorchEmbeddingModel(RecommenderModel):
    def __init__(
        self,
        factors: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        regularization: float,
    ):
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        if min(self.factors, self.epochs, self.batch_size) <= 0:
            raise ValueError("embedding factors, epochs, and batch size must be positive")

    def _setup(
        self, interactions: sparse.csr_matrix, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.random.Generator, torch.Generator]:
        self.interactions = interactions.tocsr().astype(np.float32)
        self.item_popularity = np.asarray(self.interactions.sum(axis=0)).ravel().astype(
            np.float32
        )
        positive_users, positive_items = self.interactions.nonzero()
        positive_users = positive_users.astype(np.int64)
        positive_items = positive_items.astype(np.int64)
        seen_keys = _interaction_keys(self.interactions)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.network = _EmbeddingPair(
            self.interactions.shape[0], self.interactions.shape[1], self.factors
        )
        return positive_users, positive_items, seen_keys, np_rng, generator

    def _finalize(self) -> None:
        self.user_factors = self.network.user.weight.detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        self.item_factors = self.network.item.weight.detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        # ID-only towers cannot learn a defensible vector for an item absent from
        # the fitting interactions. Remove random initialization residue while
        # keeping those items in the full-catalog candidate set at a neutral score.
        cold_items = np.asarray(self.interactions.sum(axis=0)).ravel() == 0
        cold_users = np.asarray(self.interactions.sum(axis=1)).ravel() == 0
        self.item_factors[cold_items] = 0.0
        self.user_factors[cold_users] = 0.0

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        scores = self.user_factors[user_indices] @ self.item_factors.T
        cold = np.asarray(self.interactions[user_indices].sum(axis=1)).ravel() == 0
        scores[cold] = self.item_popularity
        return scores


class BPRModel(_TorchEmbeddingModel):
    name = "bpr"

    def __init__(
        self,
        factors: int = 32,
        epochs: int = 5,
        batch_size: int = 8192,
        learning_rate: float = 0.03,
        regularization: float = 0.0001,
    ):
        super().__init__(factors, epochs, batch_size, learning_rate, regularization)

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "BPRModel":
        users, positives, seen_keys, rng, generator = self._setup(interactions, seed)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        count = users.size
        for _ in range(self.epochs):
            order = torch.randperm(count, generator=generator).numpy()
            for start in range(0, count, self.batch_size):
                selected = order[start : start + self.batch_size]
                batch_users = users[selected]
                batch_positives = positives[selected]
                batch_negatives = sample_unseen_items(
                    batch_users, self.interactions.shape[1], seen_keys, rng
                )
                u = torch.from_numpy(batch_users)
                p = torch.from_numpy(batch_positives)
                n = torch.from_numpy(batch_negatives)
                positive_scores = self.network.dot(u, p)
                negative_scores = self.network.dot(u, n)
                regularizer = (
                    self.network.user(u).pow(2).mean()
                    + self.network.item(p).pow(2).mean()
                    + self.network.item(n).pow(2).mean()
                )
                loss = functional.softplus(-(positive_scores - negative_scores)).mean()
                loss = loss + self.regularization * regularizer
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self._finalize()
        return self


class TwoTowerModel(_TorchEmbeddingModel):
    name = "two_tower"

    def __init__(
        self,
        factors: int = 32,
        epochs: int = 5,
        batch_size: int = 8192,
        negatives: int = 3,
        learning_rate: float = 0.01,
        regularization: float = 0.0001,
    ):
        super().__init__(factors, epochs, batch_size, learning_rate, regularization)
        self.negatives = int(negatives)
        if self.negatives <= 0:
            raise ValueError("two-tower negative count must be positive")

    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "TwoTowerModel":
        users, positives, seen_keys, rng, generator = self._setup(interactions, seed)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        count = users.size
        for _ in range(self.epochs):
            order = torch.randperm(count, generator=generator).numpy()
            for start in range(0, count, self.batch_size):
                selected = order[start : start + self.batch_size]
                positive_users = users[selected]
                positive_items = positives[selected]
                negative_users = np.repeat(positive_users, self.negatives)
                negative_items = sample_unseen_items(
                    negative_users, self.interactions.shape[1], seen_keys, rng
                )
                p_users = torch.from_numpy(positive_users)
                p_items = torch.from_numpy(positive_items)
                n_users = torch.from_numpy(negative_users)
                n_items = torch.from_numpy(negative_items)
                positive_logits = self.network.dot(p_users, p_items)
                negative_logits = self.network.dot(n_users, n_items)
                loss = functional.binary_cross_entropy_with_logits(
                    positive_logits, torch.ones_like(positive_logits)
                ) + functional.binary_cross_entropy_with_logits(
                    negative_logits, torch.zeros_like(negative_logits)
                )
                regularizer = (
                    self.network.user(p_users).pow(2).mean()
                    + self.network.item(p_items).pow(2).mean()
                    + self.network.item(n_items).pow(2).mean()
                )
                loss = loss + self.regularization * regularizer
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self._finalize()
        return self
