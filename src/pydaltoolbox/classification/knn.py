"""k-Nearest Neighbors classifier (``cla_knn.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.neighbors import KNeighborsClassifier

from .base import SklearnClassification

__all__ = ["ClaKnn"]


class ClaKnn(SklearnClassification):
    """Instance-based classifier (wraps scikit-learn's KNeighborsClassifier)."""

    def __init__(self, attribute: str, slevels: Sequence | None = None, k: int = 5) -> None:
        super().__init__(attribute, slevels)
        self.k = k

    def _make_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=self.k)
