"""Support Vector Machine regression (``reg_svm.R`` analog)."""
from __future__ import annotations

from sklearn.svm import SVR

from .base import SklearnRegression

__all__ = ["RegSvm"]


class RegSvm(SklearnRegression):
    """Support vector regressor (wraps scikit-learn's SVR)."""

    def __init__(
        self,
        attribute: str,
        kernel: str = "rbf",
        cost: float = 1.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(attribute)
        self.kernel = kernel
        self.cost = cost
        self.gamma = gamma
        self.epsilon = epsilon

    def _make_model(self) -> SVR:
        return SVR(kernel=self.kernel, C=self.cost, gamma=self.gamma, epsilon=self.epsilon)
