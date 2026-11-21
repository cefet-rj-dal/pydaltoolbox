"""k-Nearest Neighbors classifier (``cla_knn.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.neighbors import KNeighborsClassifier

from .base import SklearnClassification

__all__ = ["ClaKnn"]


class ClaKnn(SklearnClassification):
    """Instance-based classifier (wraps scikit-learn's KNeighborsClassifier).

    k-Nearest Neighbors is a simple, non-parametric method that classifies each point by
    majority vote among its ``k`` closest training examples. The choice of ``k`` trades
    off bias and variance; the distance metric is Euclidean by default, so features
    should be on comparable scales.

    References:
        Cover, T., and Hart, P. (1967). Nearest neighbor pattern classification. IEEE
        Transactions on Information Theory, 13(1):21-27.
    """

    def __init__(self, attribute: str, slevels: Sequence | None = None, k: int = 5) -> None:
        super().__init__(attribute, slevels)
        self.k = k

    def _make_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=self.k)
