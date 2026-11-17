"""Decision tree regression (``reg_dtree.R`` analog)."""
from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from .base import SklearnRegression

__all__ = ["RegDtree"]


class RegDtree(SklearnRegression):
    """Decision tree for regression (wraps scikit-learn).

    Pre-pruning defaults (``min_samples_split=10``, ``min_samples_leaf=5``) approximate
    R ``tree::tree`` (``minsize=10``, ``mincut=5``). A fully grown tree overfits and
    generalizes poorly, which is why the pre-pruning matters for quality.
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
