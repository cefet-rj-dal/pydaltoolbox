"""k-Nearest Neighbors regression (``reg_knn.R`` analog)."""
from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from .base import SklearnRegression

__all__ = ["RegKnn"]


class RegKnn(SklearnRegression):
    """Instance-based regressor (wraps scikit-learn's KNeighborsRegressor).

    Predicts the target of a point by averaging the targets of its ``k`` nearest
    training examples -- a non-parametric method well suited to local smoothing. It is
    sensitive to feature scaling, so normalization beforehand is advisable.

    References:
        Altman, N. S. (1992). An Introduction to Kernel and Nearest-Neighbor
        Nonparametric Regression. The American Statistician, 46(3):175-185.
    """

    def __init__(self, attribute: str, k: int = 5) -> None:
        super().__init__(attribute)
        self.k = k

    def _make_model(self) -> KNeighborsRegressor:
        return KNeighborsRegressor(n_neighbors=self.k)
