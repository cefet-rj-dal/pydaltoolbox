"""Ensemble classifiers.

Analogs of ``cla_rf.R`` (random forest), ``cla_bagging.R`` and ``cla_boosting.R``.
All are thin adapters over scikit-learn ensembles.
"""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)

from .base import SklearnClassification

__all__ = ["ClaBagging", "ClaBoosting", "ClaRf"]


class ClaRf(SklearnClassification):
    """Random forest classifier (wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        ntree: int = 100,
        mtry: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.ntree = ntree
        self.mtry = mtry
        self.random_state = random_state

    def _make_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.ntree,
            max_features=self.mtry,
            random_state=self.random_state,
        )


class ClaBagging(SklearnClassification):
    """Bagging classifier (wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        nbagg: int = 25,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.nbagg = nbagg
        self.random_state = random_state

    def _make_model(self) -> BaggingClassifier:
        return BaggingClassifier(n_estimators=self.nbagg, random_state=self.random_state)


class ClaBoosting(SklearnClassification):
    """AdaBoost classifier (the ``adabag`` boosting analog, wraps scikit-learn)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        mfinal: int = 50,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.mfinal = mfinal
        self.random_state = random_state

    def _make_model(self) -> AdaBoostClassifier:
        return AdaBoostClassifier(n_estimators=self.mfinal, random_state=self.random_state)
