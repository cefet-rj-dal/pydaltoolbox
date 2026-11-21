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
    """Random forest classifier (wraps scikit-learn).

    A random forest combines many decorrelated decision trees -- each grown on a
    bootstrap sample using a random subset of features at every split -- and aggregates
    their votes, which reduces variance and overfitting. Key hyperparameters are the
    number of trees (``ntree``) and the number of variables tried per split (``mtry``).

    Porting:
        Mirrors ``cla_rf.R``. Fidelity: approximate.
        - scikit-learn ``RandomForestClassifier`` backend instead of R ``randomForest``;
          results differ numerically with the RNG/implementation.

    References:
        Breiman, L. (2001). Random Forests. Machine Learning, 45(1):5-32.
        Liaw, A., and Wiener, M. (2002). Classification and Regression by randomForest.
        R News, 2(3):18-22.
    """

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
    """Bagging classifier (wraps scikit-learn).

    Bootstrap aggregating (bagging) trains an ensemble of base classifiers, each on a
    bootstrap resample of the training data, and combines them by majority vote. By
    averaging over resamples it reduces the variance of unstable learners such as
    decision trees.

    Porting:
        Mirrors ``cla_bagging.R``. Fidelity: approximate.
        - scikit-learn ``BaggingClassifier`` backend instead of R ``ipred::bagging``;
          results differ numerically with the RNG/implementation.

    References:
        Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2):123-140.
    """

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
    """AdaBoost classifier (the ``adabag`` boosting analog, wraps scikit-learn).

    Boosting builds an ensemble sequentially: each new weak learner focuses on the
    examples the previous ones misclassified (by reweighting the training set), and the
    learners are combined into a weighted vote. AdaBoost (adaptive boosting) is the
    classic instance of this scheme and often yields strong accuracy from simple trees.

    Porting:
        Mirrors ``cla_boosting.R``. Fidelity: adapted.
        - scikit-learn ``AdaBoostClassifier`` backend instead of R ``adabag::boosting``
          (a related SAMME-style boosting rather than the exact adabag algorithm).

    References:
        Freund, Y., and Schapire, R. E. (1997). A Decision-Theoretic Generalization of
        On-Line Learning and an Application to Boosting. Journal of Computer and System
        Sciences, 55(1):119-139.
    """

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
