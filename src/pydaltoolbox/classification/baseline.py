"""Baseline classifier (``cla_majority.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.dummy import DummyClassifier

from .base import SklearnClassification

__all__ = ["ClaMajority"]


class ClaMajority(SklearnClassification):
    """Majority-class baseline (wraps scikit-learn's DummyClassifier).

    Always predicts the most frequent class, ignoring the predictors. It is a minimum
    reference point: if a real classifier does not outperform this baseline, it is often
    worth revisiting the predictors, the sampling strategy, or the evaluation protocol
    before tuning models.

    Porting:
        Mirrors ``cla_majority.R``. Fidelity: exact.
        - scikit-learn ``DummyClassifier(strategy="most_frequent")``; deterministic.

    References:
        Witten, I. H., Frank, E., Hall, M. A., and Pal, C. J. (2016). Data Mining:
        Practical Machine Learning Tools and Techniques (4th ed.). Morgan Kaufmann.
    """

    def __init__(self, attribute: str, slevels: Sequence | None = None) -> None:
        super().__init__(attribute, slevels)

    def _make_model(self) -> DummyClassifier:
        return DummyClassifier(strategy="most_frequent")
