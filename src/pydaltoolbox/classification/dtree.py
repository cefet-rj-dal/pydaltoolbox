"""Decision tree classifiers.

Thin adapters over scikit-learn trees. ``ClaDtree`` mirrors ``cla_dtree.R`` and
``ClaRpart`` mirrors ``cla_rpart.R`` (scikit-learn's DecisionTree is a CART, which
is exactly what ``rpart`` builds).
"""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.tree import DecisionTreeClassifier

from .base import SklearnClassification

__all__ = ["ClaDtree", "ClaRpart"]


class ClaDtree(SklearnClassification):
    """Decision tree for classification (wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=self.random_state)


class ClaRpart(SklearnClassification):
    """CART decision tree (the ``rpart`` analog; scikit-learn trees are CART)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(criterion="gini", random_state=self.random_state)
