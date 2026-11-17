"""Support Vector Machine regression (``reg_svm.R`` analog)."""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .base import SklearnRegression

__all__ = ["RegSvm"]


class RegSvm(SklearnRegression):
    """Support vector regressor (wraps scikit-learn's SVR).

    Both the features and the target are standardized inside the model, replicating
    R ``e1071::svm``'s ``scale = TRUE`` default (which scales x and y and reverses it
    on prediction). Without this, SVR badly underfits targets with large magnitude.
    The default ``cost`` matches the daltoolbox default.
    """

    def __init__(
        self,
        attribute: str,
        kernel: str = "rbf",
        cost: float = 10.0,
        gamma: str | float = "scale",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(attribute)
        self.kernel = kernel
        self.cost = cost
        self.gamma = gamma
        self.epsilon = epsilon

    def _make_model(self) -> TransformedTargetRegressor:
        regressor = make_pipeline(
            StandardScaler(),
            SVR(kernel=self.kernel, C=self.cost, gamma=self.gamma, epsilon=self.epsilon),
        )
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
