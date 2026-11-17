"""Decision tree classifier.

Python counterpart of ``cla_dtree.R``. This is a *thin adapter*: the algorithm
is ``sklearn.tree.DecisionTreeClassifier``; our contribution is wiring it into the
Experiment Line contract (``fit`` / ``predict`` returning a class-score table).
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..adjust import adjust_dataframe
from .base import Classification

__all__ = ["ClaDtree"]


class ClaDtree(Classification):
    """Decision tree for classification (wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.random_state = random_state
        self.model: DecisionTreeClassifier | None = None

    def fit(self, data, **kwargs) -> ClaDtree:
        data = self._prepare_classification_data(data)
        x = data[self.x]
        y = data[self.attribute].astype(object)
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.model.fit(x, y)
        return self

    def predict(self, x, **kwargs) -> pd.DataFrame:
        """Return a class-score table (one column per class level).

        Only the feature columns (``self.x``) are used, so ``x`` may be either the
        feature frame or the full dataset including the target column; the target,
        if present, is ignored. Passing only the features is the recommended habit.
        """
        x = adjust_dataframe(x)
        proba = self.model.predict_proba(x[self.x])
        result = pd.DataFrame(proba, columns=[str(c) for c in self.model.classes_])
        # reorder/complete columns to match the declared class levels
        return result.reindex(columns=[str(s) for s in self.slevels], fill_value=0.0)
