"""Gradient boosting classifier (``cla_xgboost.R`` analog).

The ``xgboost`` backend is an optional dependency, imported lazily so the core
package stays light; install it with ``pip install pydaltoolbox[xgboost]``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..adjust import adjust_dataframe
from .base import SklearnClassification

__all__ = ["ClaXgboost"]


def _require_xgboost():
    """Import xgboost lazily, with a clear install hint when it is missing."""
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ClaXgboost requires the optional 'xgboost' dependency. "
            "Install it with: pip install pydaltoolbox[xgboost]"
        ) from exc
    return xgboost


class ClaXgboost(SklearnClassification):
    """Gradient boosting classifier (wraps xgboost's scikit-learn API).

    Gradient boosting builds an additive ensemble of regression trees in a
    forward stage-wise fashion: each new tree is fit to the gradient of a
    differentiable loss with respect to the current model's output, so it
    corrects the residual errors of the trees already in the ensemble. XGBoost is
    a scalable, regularized realization of this scheme that adds second-order
    (Newton) boosting, shrinkage, column subsampling and sparsity-aware split
    finding, which is why it is a strong default for tabular classification.
    ``nrounds`` sets the number of boosting rounds (trees); the probabilistic
    objective is chosen by the number of classes -- a logistic objective for two
    classes and a softmax (softprob) objective for the multiclass case.

    Porting:
        Mirrors ``cla_xgboost.R``. Fidelity: adapted.
        - xgboost ``XGBClassifier`` (scikit-learn API) backend instead of the
          lower-level ``xgboost::xgb.train``; it selects the same default
          objective (logistic for binary, softmax/softprob for multiclass) from
          the class count, matching the R default behavior, and ``nrounds`` maps
          to ``n_estimators``.
        - ``params`` is an optional dict of extra arguments forwarded to
          ``XGBClassifier`` (counterpart of the R ``params`` list).

    References:
        Friedman, J. H. (2001). Greedy Function Approximation: A Gradient
        Boosting Machine. Annals of Statistics, 29(5):1189-1232.
        Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
        System. Proceedings of the 22nd ACM SIGKDD International Conference on
        Knowledge Discovery and Data Mining, 785-794.
    """

    def __init__(
        self,
        attribute: str,
        slevels: Sequence | None = None,
        params: dict[str, Any] | None = None,
        nrounds: int = 20,
        random_state: int | None = None,
    ) -> None:
        super().__init__(attribute, slevels)
        self.params = params
        self.nrounds = nrounds
        self.random_state = random_state

    def _make_model(self):
        xgboost = _require_xgboost()
        kwargs: dict[str, Any] = {
            "n_estimators": self.nrounds,
            "random_state": self.random_state,
        }
        if self.params:
            kwargs.update(self.params)
        return xgboost.XGBClassifier(**kwargs)

    def fit(self, data, **kwargs) -> ClaXgboost:
        # XGBClassifier requires integer-encoded labels (0..n-1); encode them in
        # the ``slevels`` order, mirroring the R ``as.numeric(factor) - 1``.
        data = self._prepare_classification_data(data)
        self.model = self._make_model()
        code = {label: i for i, label in enumerate(self.slevels)}
        y = data[self.attribute].astype(object).map(code)
        self.model.fit(data[self.x], y)
        return self

    def predict(self, x, **kwargs) -> pd.DataFrame:
        """Return a class-score table (one column per class level)."""
        x = adjust_dataframe(x)
        proba = self.model.predict_proba(x[self.x])
        # ``model.classes_`` are integer codes into the ``slevels`` order.
        cols = [str(self.slevels[int(c)]) for c in self.model.classes_]
        result = pd.DataFrame(proba, columns=cols)
        return result.reindex(columns=[str(s) for s in self.slevels], fill_value=0.0)
