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

    A multi-layer perceptron is a feedforward neural network trained by
    back-propagation. This single-hidden-layer model uses a softmax output over one-hot
    targets; ``size`` controls the number of hidden units and ``decay`` applies L2
    regularization. Features should be scaled for stable training.

    Features are standardized inside the model: neural networks are scale-sensitive
    and the R ``nnet`` documentation explicitly recommends scaling. ``size`` -> hidden
    layer width (if ``None``, a ``ceil(n_features / 3)`` heuristic like the R package),
    ``decay`` -> L2 regularization (``alpha``).

    Porting:
        Mirrors ``cla_mlp.R``. Fidelity: adapted.
        - scikit-learn ``MLPClassifier`` backend instead of R ``nnet``.
        - Wraps ``StandardScaler`` (R ``nnet`` recommends scaling; the net is
          scale-sensitive); ``decay=0.05`` and a ``ceil(n_features/3)`` hidden-size
          heuristic match the R defaults.

    References:
        Rumelhart, D., Hinton, G., and Williams, R. (1986). Learning representations by
        back-propagating errors. Nature, 323:533-536.
        Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford University
        Press.
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
