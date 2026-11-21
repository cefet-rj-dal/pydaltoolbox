"""Multi-layer perceptron regression (``reg_mlp.R`` analog)."""
from __future__ import annotations

import math

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import SklearnRegression

__all__ = ["RegMlp"]


class RegMlp(SklearnRegression):
    """Single-hidden-layer MLP regressor (wraps scikit-learn).

    A feedforward neural network with one hidden layer of ``size`` units, trained by
    back-propagation with L2 regularization controlled by ``decay``. It can model smooth
    non-linear relationships; data should be scaled for stable training.

    Features and target are standardized inside the model (neural networks are
    scale-sensitive and the R ``nnet`` documentation recommends scaling). ``size`` ->
    hidden layer width (if ``None``, a ``ceil(n_features / 3)`` heuristic like the R
    package), ``decay`` -> L2 regularization (``alpha``), matching the R defaults.

    References:
        Rumelhart, D., Hinton, G., and Williams, R. (1986). Learning representations by
        back-propagating errors. Nature, 323:533-536.
        Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford University
        Press.
    """

    def __init__(
        self,
        attribute: str,
        size: int | None = None,
        decay: float = 0.05,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute)
        self.size = size
        self.decay = decay
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> TransformedTargetRegressor:
        size = self.size if self.size is not None else max(1, math.ceil(len(self.x) / 3))
        regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(size,),
                alpha=self.decay,
                max_iter=self.maxit,
                random_state=self.random_state,
            ),
        )
        return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
