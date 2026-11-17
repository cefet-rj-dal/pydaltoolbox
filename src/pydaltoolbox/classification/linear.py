"""Linear (logistic-regression based) classifiers.

Analogs of ``cla_glm.R`` (binomial/logistic), ``cla_multinom.R`` (multinomial) and
``cla_glmnet.R`` (elastic-net regularized). In Python all three are expressed with
scikit-learn's ``LogisticRegression`` under different configurations, but we keep
three classes to preserve the API surface the downstream packages expect.
"""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.linear_model import LogisticRegression

from .base import SklearnClassification

__all__ = ["ClaGlm", "ClaGlmnet", "ClaMultinom"]


class ClaGlm(SklearnClassification):
    """Logistic-regression classifier (generalized linear model)."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=self.maxit, random_state=self.random_state)


class ClaMultinom(SklearnClassification):
    """Multinomial logistic-regression classifier."""

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        maxit: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=self.maxit, random_state=self.random_state)


class ClaGlmnet(SklearnClassification):
    """Elastic-net regularized logistic regression (``glmnet`` analog).

    ``alpha`` is the elastic-net mixing ratio (0 = ridge, 1 = lasso).
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        alpha: float = 0.5,
        maxit: int = 10000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.alpha = alpha
        self.maxit = maxit
        self.random_state = random_state

    def _make_model(self) -> LogisticRegression:
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.alpha,
            max_iter=self.maxit,
            random_state=self.random_state,
        )
