"""Multi-layer perceptron regression (``reg_mlp.R`` analog)."""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor

from .base import SklearnRegression

__all__ = ["RegMlp"]


class RegMlp(SklearnRegression):
    """Single-hidden-layer MLP regressor (wraps scikit-learn).

    ``size`` -> hidden layer width, ``decay`` -> L2 regularization (``alpha``).
    """

    def __init__(
        self,
        attribute: str,
        size: int = 5,
        decay: float = 0.0,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute)
        self.size = size
        self.decay = decay
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=(self.size,),
            alpha=self.decay,
            max_iter=self.maxit,
            random_state=self.random_state,
        )
