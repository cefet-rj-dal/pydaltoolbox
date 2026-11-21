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

    Support Vector Regression fits a function within an epsilon-insensitive tube around
    the data, ignoring errors smaller than ``epsilon`` and penalizing larger ones. The
    ``cost`` controls regularization strength, ``epsilon`` sets the width of the
    insensitive tube, and ``kernel`` defines the feature map (linear, radial/RBF,
    polynomial, sigmoid).

    Both the features and the target are standardized inside the model, replicating
    R ``e1071::svm``'s ``scale = TRUE`` default (which scales x and y and reverses it
    on prediction). Without this, SVR badly underfits targets with large magnitude.
    The default ``cost`` matches the daltoolbox default.

    Porting:
        Mirrors ``reg_svm.R``. Fidelity: adapted.
        - scikit-learn ``SVR`` backend instead of R ``e1071::svm``.
        - Wraps ``StandardScaler`` on the features and, via ``TransformedTargetRegressor``,
          on the target to replicate ``e1071::svm``'s ``scale=TRUE`` default (which scales
          x and y and reverses it on prediction); without it SVR underfits large targets.
        - Default ``cost=10`` matches daltoolbox.

    References:
        Drucker, H., Burges, C., Kaufman, L., Smola, A., and Vapnik, V. (1997). Support
        Vector Regression Machines. Advances in Neural Information Processing Systems 9.
        Chang, C.-C., and Lin, C.-J. (2011). LIBSVM: A library for support vector
        machines.
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
