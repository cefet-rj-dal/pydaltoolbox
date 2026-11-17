"""Multi-layer perceptron classifier (``cla_mlp.R`` analog)."""
from __future__ import annotations

import math
from collections.abc import Sequence

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import SklearnClassification

__all__ = ["ClaMlp"]


class ClaMlp(SklearnClassification):
    """Single-hidden-layer MLP classifier (wraps scikit-learn).

    Features are standardized inside the model: neural networks are scale-sensitive
    and the R ``nnet`` documentation explicitly recommends scaling. ``size`` -> hidden
    layer width (if ``None``, a ``ceil(n_features / 3)`` heuristic like the R package),
    ``decay`` -> L2 regularization (``alpha``).
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        size: int | None = None,
        decay: float = 0.05,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.size = size
        self.decay = decay
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> Pipeline:
        size = self.size if self.size is not None else max(1, math.ceil(len(self.x) / 3))
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(size,),
                alpha=self.decay,
                max_iter=self.maxit,
                random_state=self.random_state,
            ),
        )
