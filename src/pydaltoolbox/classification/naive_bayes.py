"""Naive Bayes classifier (``cla_nb.R`` analog)."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.naive_bayes import GaussianNB

from .base import SklearnClassification

__all__ = ["ClaNb"]


class ClaNb(SklearnClassification):
    """Gaussian Naive Bayes classifier (wraps scikit-learn).

    Naive Bayes applies Bayes' theorem under the "naive" assumption that features are
    conditionally independent given the class label, which makes probabilistic
    classification very fast. The Gaussian variant models each continuous feature's
    class-conditional density as a normal distribution.

    Porting:
        Mirrors ``cla_nb.R``. Fidelity: approximate.
        - scikit-learn ``GaussianNB`` backend instead of R ``e1071::naiveBayes``.

    References:
        Mitchell, T. (1997). Machine Learning. McGraw-Hill.
    """

    def __init__(self, attribute: str, slevels: Sequence | None = None) -> None:
        super().__init__(attribute, slevels)

    def _make_model(self) -> GaussianNB:
        return GaussianNB()
