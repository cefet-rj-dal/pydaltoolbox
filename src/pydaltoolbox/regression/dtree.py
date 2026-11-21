"""Decision tree regression (``reg_dtree.R`` analog)."""
from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from .base import SklearnRegression

__all__ = ["RegDtree"]


class RegDtree(SklearnRegression):
    """Decision tree for regression (wraps scikit-learn).

    A regression tree recursively partitions the feature space, choosing splits that
    reduce the within-node squared error; each leaf predicts a constant (the mean of its
    training targets), so the model is a piecewise-constant, interpretable function.

    Pre-pruning defaults (``min_samples_split=10``, ``min_samples_leaf=5``) approximate
    R ``tree::tree`` (``minsize=10``, ``mincut=5``). A fully grown tree overfits and
    generalizes poorly, which is why the pre-pruning matters for quality.

    Porting:
        Mirrors ``reg_dtree.R``. Fidelity: adapted.
        - scikit-learn ``DecisionTreeRegressor`` backend instead of R ``tree::tree``.
        - Pre-pruning defaults (``min_samples_split=10``, ``min_samples_leaf=5``)
          approximate R ``tree::tree`` (``minsize=10``, ``mincut=5``).

    References:
        Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification and
        Regression Trees. Wadsworth.
    """

    def __init__(self, attribute: str, random_state: int | None = None) -> None:
        super().__init__(attribute)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
        )
