from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class RecommenderModel(ABC):
    name: str

    @abstractmethod
    def fit(self, interactions: sparse.csr_matrix, seed: int) -> "RecommenderModel":
        raise NotImplementedError

    @abstractmethod
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a dense users-by-full-catalog score matrix."""

        raise NotImplementedError
