"""k-Nearest Neighbors regression (``reg_knn.R`` analog)."""
from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from .base import SklearnRegression

__all__ = ["RegKnn"]


class RegKnn(SklearnRegression):
    """Instance-based regressor (wraps scikit-learn's KNeighborsRegressor)."""

    def __init__(self, attribute: str, k: int = 5) -> None:
        super().__init__(attribute)
        self.k = k

    def _make_model(self) -> KNeighborsRegressor:
        return KNeighborsRegressor(n_neighbors=self.k)
