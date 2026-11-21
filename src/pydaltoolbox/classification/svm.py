"""Support Vector Machine classifier (``cla_svm.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import SklearnClassification

__all__ = ["ClaSvm"]


class ClaSvm(SklearnClassification):
    """Support vector classifier (wraps scikit-learn's SVC).

    Support Vector Machines find a maximum-margin hyperplane separating the classes in a
    feature space defined by a kernel (linear, radial/RBF, polynomial, sigmoid). The
    ``cost`` controls the trade-off between margin width and training error, and
    ``kernel`` sets the feature map.

    The features are standardized inside the model, replicating R ``e1071::svm``'s
    ``scale = TRUE`` default (SVMs are scale-sensitive). ``probability=True`` is
    required so ``predict`` can return class scores. The default ``cost`` matches
    the daltoolbox default.

    References:
        Cortes, C., and Vapnik, V. (1995). Support-Vector Networks. Machine Learning,
        20(3):273-297.
        Chang, C.-C., and Lin, C.-J. (2011). LIBSVM: A library for support vector
        machines.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        kernel: str = "rbf",
        cost: float = 10.0,
        gamma: str | float = "scale",
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.kernel = kernel
        self.cost = cost
        self.gamma = gamma
        self.random_state = random_state

    def _make_model(self) -> Pipeline:
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel=self.kernel,
                C=self.cost,
                gamma=self.gamma,
                probability=True,
                random_state=self.random_state,
            ),
        )
