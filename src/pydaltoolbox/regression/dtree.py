"""Decision tree regression (``reg_dtree.R`` analog)."""
from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from .base import SklearnRegression

__all__ = ["RegDtree"]


class RegDtree(SklearnRegression):
    """Decision tree for regression (wraps scikit-learn)."""

    def __init__(self, attribute: str, random_state: int | None = None) -> None:
        super().__init__(attribute)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(random_state=self.random_state)
