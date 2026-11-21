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
    """Logistic-regression classifier (generalized linear model).

    A generalized linear model with a logit link: it models the log-odds of the class as
    a linear combination of the predictors, fitted by maximum likelihood. Simple,
    interpretable, and a strong baseline for (mainly binary) classification.

    References:
        Nelder, J. A., and Wedderburn, R. W. M. (1972). Generalized Linear Models.
        Journal of the Royal Statistical Society, Series A, 135(3):370-384.
        Hosmer, D. W., and Lemeshow, S. (2000). Applied Logistic Regression (2nd ed.).
        Wiley.
    """

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
    """Multinomial logistic-regression classifier.

    Extends logistic regression to more than two classes via the multinomial (softmax)
    model, estimating one set of coefficients per class and normalizing their
    exponentials into class probabilities. The analog of R ``nnet::multinom``.

    References:
        Agresti, A. (2002). Categorical Data Analysis (2nd ed.). Wiley.
        Venables, W. N., and Ripley, B. D. (2002). Modern Applied Statistics with S
        (4th ed.). Springer.
    """

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

    Logistic regression with an elastic-net penalty, a convex combination of the L1
    (lasso) and L2 (ridge) penalties. The L1 part performs feature selection by driving
    coefficients to zero while the L2 part handles correlated predictors, so the model is
    well suited to high-dimensional data. ``alpha`` is the elastic-net mixing ratio
    (0 = ridge, 1 = lasso).

    References:
        Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization Paths for
        Generalized Linear Models via Coordinate Descent. Journal of Statistical
        Software, 33(1):1-22.
        Zou, H., and Hastie, T. (2005). Regularization and variable selection via the
        elastic net. Journal of the Royal Statistical Society, Series B, 67(2):301-320.
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
