"""Predictive (model-based) imputation.

Python counterparts of ``trans_imputation_predictive.R`` (base) and
``trans_imputation_tree.R``. A predictive imputer learns **one** target column from a set
of source columns: it fits a model on the rows where the target is observed and uses it to
fill the missing ones. Source columns are first completed with :class:`ImputationSimple`
so the model can always be trained and applied.
"""
from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..adjust import adjust_dataframe
from ..base import DalTransform
from .clean import ImputationSimple

__all__ = ["ImputationPredictive", "ImputationTree"]


class ImputationPredictive(DalTransform):
    """Base class for supervised imputers (learn ``target`` from ``sources``).

    Parameters
    ----------
    target:
        Column to impute.
    sources:
        Predictor columns (default: all other columns).
    method:
        Initial imputation for numeric source columns: ``"median"`` or ``"mean"``.

    This base only resolves target/sources; concrete subclasses (e.g.
    :class:`ImputationTree`) implement ``fit``/``transform``.

    Predictive imputation treats each incomplete column as a supervised learning problem:
    a model is trained on the rows where it is observed and predicts the missing entries
    from the other columns. This exploits relationships between variables, giving more
    plausible fills than constant/mean imputation -- the idea behind chained-equations
    imputation.

    Porting:
        Mirrors ``trans_imputation_predictive.R``. Fidelity: adapted.
        - Base class: resolves target/sources and pre-fills source columns with
          :class:`ImputationSimple`; concrete subclasses supply the model.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann.
        van Buuren, S., and Groothuis-Oudshoorn, K. (2011). mice: Multivariate
        Imputation by Chained Equations in R. Journal of Statistical Software, 45(3):1-67.
    """

    def __init__(self, target: str, sources: list | None = None, method: str = "median") -> None:
        if method not in ("median", "mean"):
            raise ValueError("imputation_predictive: method must be 'median' or 'mean'.")
        self.target = target
        self.sources = sources
        self.method = method

    def _resolve(self, data) -> tuple:
        data = adjust_dataframe(data)
        if self.target not in data.columns:
            raise ValueError(f"imputation_predictive: target '{self.target}' not in data.")
        sources = self.sources if self.sources is not None else [
            c for c in data.columns if c != self.target
        ]
        sources = [c for c in sources if c in data.columns and c != self.target]
        if not sources:
            raise ValueError("imputation_predictive: at least one source column is required.")
        if data[self.target].notna().sum() == 0:
            raise ValueError(f"imputation_predictive: target '{self.target}' is all missing.")
        return data, sources


class ImputationTree(ImputationPredictive):
    """Tree-based predictive imputation of a single target column.

    Fits a decision tree (regressor for a numeric target, classifier otherwise) on the
    observed-target rows, using the source columns as predictors, then predicts the
    missing values. A tree captures non-linear dependencies and interactions among the
    sources without requiring them to be scaled.

    Porting:
        Mirrors ``trans_imputation_tree.R``. Fidelity: adapted.
        - scikit-learn decision tree backend instead of R ``tree``; pre-pruning defaults
          match R's ``tree`` package.
        - Categorical source columns are one-hot encoded before fitting the tree.

    References:
        Breiman, L., Friedman, J., Olshen, R., and Stone, C. (1984). Classification and
        Regression Trees. Wadsworth.
        van Buuren, S., and Groothuis-Oudshoorn, K. (2011). mice: Multivariate
        Imputation by Chained Equations in R. Journal of Statistical Software, 45(3):1-67.
    """

    def __init__(
        self,
        target: str,
        sources: list | None = None,
        method: str = "median",
        random_state: int | None = None,
    ) -> None:
        super().__init__(target, sources, method)
        self.random_state = random_state
        self.sources_: list | None = None
        self.initial_: ImputationSimple | None = None
        self.model_ = None
        self.feature_names_: list | None = None
        self.target_numeric_ = True

    def _encode(self, x: pd.DataFrame) -> pd.DataFrame:
        cat = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
        if cat:
            return pd.get_dummies(x, columns=cat, dtype=float)
        return x.astype(float)

    def fit(self, data, **kwargs) -> ImputationTree:
        data, sources = self._resolve(data)
        self.sources_ = sources
        self.initial_ = ImputationSimple(method=self.method, cols=sources).fit(data)
        work = self.initial_.transform(data)

        observed = data[self.target].notna()
        x = self._encode(work.loc[observed, sources])
        y = data.loc[observed, self.target]
        self.feature_names_ = list(x.columns)
        self.target_numeric_ = pd.api.types.is_numeric_dtype(y)

        kw = {"min_samples_split": 10, "min_samples_leaf": 5, "random_state": self.random_state}
        if self.target_numeric_:
            self.model_ = DecisionTreeRegressor(**kw).fit(x, y.astype(float))
        else:
            self.model_ = DecisionTreeClassifier(**kw).fit(x, y.astype(object))
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        missing = data[self.target].isna()
        if not missing.any():
            return data
        work = self.initial_.transform(data)
        x = self._encode(work.loc[missing, self.sources_]).reindex(
            columns=self.feature_names_, fill_value=0.0
        )
        data.loc[missing, self.target] = self.model_.predict(x)
        return data
