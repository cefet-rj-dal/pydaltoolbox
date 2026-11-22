"""Decision tree regression (``reg_dtree.R`` analog)."""
from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from .base import SklearnRegression

__all__ = ["RegDtree"]


class RegDtree(SklearnRegression):
    """Decision tree for regression (wraps scikit-learn).

    A regression tree recursively partitions the feature space, choosing splits that
    reduce the within-node squared error; each leaf predicts a constant (the mean of its
    training targets), so the model is a piecewise-constant, interpretable function.

    Pre-pruning mirrors R ``tree::tree``: besides the node-size floors
    (``min_samples_split=10``, ``min_samples_leaf=5``), a node is split only when it
    reduces impurity by at least ``mindev`` (default ``0.01``) times the target variance,
    reproducing ``tree::tree``'s ``mindev`` complexity stop. Without this relative stop a
    single tree grows deep and overfits (negative test R2 on near-linear targets such as
    diabetes), while with it the tree generalizes and matches the R backend.

    Porting:
        Mirrors ``reg_dtree.R``. Fidelity: adapted.
        - scikit-learn ``DecisionTreeRegressor`` backend instead of R ``tree::tree``.
        - ``mindev`` maps to ``min_impurity_decrease = mindev * var(y)`` (computed at fit),
          the relative-deviance split stop of ``tree::tree``; node-size floors
          (``min_samples_split=10``, ``min_samples_leaf=5``) match ``minsize``/``mincut``.

    References:
        Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification and
        Regression Trees. Wadsworth.
    """

    def __init__(
        self, attribute: str, mindev: float = 0.01, random_state: int | None = None
    ) -> None:
        super().__init__(attribute)
        self.mindev = mindev
        self.random_state = random_state
        self._y_var = 0.0

    def fit(self, data, **kwargs) -> RegDtree:
        self._y_var = float(pd.to_numeric(pd.DataFrame(data)[self.attribute]).var())
        return super().fit(data, **kwargs)

    def _make_model(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            min_samples_split=10,
            min_samples_leaf=5,
            min_impurity_decrease=self.mindev * self._y_var,
            random_state=self.random_state,
        )
