"""Decision tree classifiers.

Thin adapters over scikit-learn trees. ``ClaDtree`` mirrors ``cla_dtree.R`` and
``ClaRpart`` mirrors ``cla_rpart.R``. scikit-learn trees grow to full depth by
default (which overfits), whereas R's ``tree`` and ``rpart`` pre-prune; we set
pre-pruning defaults that approximate those packages so quality matches.
"""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.tree import DecisionTreeClassifier

from .base import SklearnClassification

__all__ = ["ClaDtree", "ClaRpart"]


class ClaDtree(SklearnClassification):
    """Decision tree for classification (wraps scikit-learn).

    Decision trees recursively split the feature space to maximize node purity (e.g.
    Gini impurity or entropy), yielding a human-readable set of if-then rules. They are
    fast and interpretable, and often serve as base learners in ensembles.

    Pre-pruning defaults (``min_samples_split=10``, ``min_samples_leaf=5``)
    approximate R ``tree::tree`` (``minsize=10``, ``mincut=5``).

    References:
        Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification
        and Regression Trees. Wadsworth.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
        )


class ClaRpart(SklearnClassification):
    """CART decision tree (the ``rpart`` analog, wraps scikit-learn).

    A CART (Classification And Regression Trees) decision tree, splitting on Gini
    impurity like R's ``rpart``. Same interpretable tree model as ``ClaDtree`` with the
    pruning conventions of ``rpart`` rather than ``tree``.

    Pre-pruning defaults (``min_samples_split=20``, ``min_samples_leaf=7``)
    approximate R ``rpart`` (``minsplit=20``, ``minbucket=7``).

    References:
        Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification
        and Regression Trees. Wadsworth.
        Therneau, T. M., & Atkinson, E. J. (1997). An Introduction to Recursive
        Partitioning Using the rpart Routines.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.random_state = random_state

    def _make_model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion="gini",
            min_samples_split=20,
            min_samples_leaf=7,
            random_state=self.random_state,
        )
