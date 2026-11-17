"""Baseline classifier (``cla_majority.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.dummy import DummyClassifier

from .base import SklearnClassification

__all__ = ["ClaMajority"]


class ClaMajority(SklearnClassification):
    """Majority-class baseline (wraps scikit-learn's DummyClassifier).

    Always predicts the most frequent class. Useful as a reference floor against
    which real learners are compared.
    """

    def __init__(self, attribute: str, slevels: Sequence | None = None) -> None:
        super().__init__(attribute, slevels)

    def _make_model(self) -> DummyClassifier:
        return DummyClassifier(strategy="most_frequent")
