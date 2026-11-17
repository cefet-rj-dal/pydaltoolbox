"""Naive Bayes classifier (``cla_nb.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.naive_bayes import GaussianNB

from .base import SklearnClassification

__all__ = ["ClaNb"]


class ClaNb(SklearnClassification):
    """Gaussian Naive Bayes classifier (wraps scikit-learn)."""

    def __init__(self, attribute: str, slevels: Sequence | None = None) -> None:
        super().__init__(attribute, slevels)

    def _make_model(self) -> GaussianNB:
        return GaussianNB()
