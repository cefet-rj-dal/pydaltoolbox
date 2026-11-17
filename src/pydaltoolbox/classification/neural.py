"""Multi-layer perceptron classifier (``cla_mlp.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.neural_network import MLPClassifier

from .base import SklearnClassification

__all__ = ["ClaMlp"]


class ClaMlp(SklearnClassification):
    """Single-hidden-layer MLP classifier (wraps scikit-learn).

    ``size`` -> hidden layer width, ``decay`` -> L2 regularization (``alpha``),
    matching the ``nnet`` hyperparameter names used by the R package's tuning.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        size: int = 5,
        decay: float = 0.0,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.size = size
        self.decay = decay
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(self.size,),
            alpha=self.decay,
            max_iter=self.maxit,
            random_state=self.random_state,
        )
