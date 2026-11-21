"""Feature-selection transforms.

Python counterparts of the R ``trans_feature_selection_*`` wrappers:

* ``FeatureSelectionCorr``    - drop highly correlated numeric features (caret::findCorrelation)
* ``FeatureSelectionLasso``   - keep predictors with non-zero L1 coefficients (glmnet lasso)
* ``FeatureSelectionInfoGain``- rank by entropy-based information gain (quantile-binned)
* ``FeatureSelectionStepwise``- forward subset search maximizing adjusted R-squared (leaps)

Each learns ``selected`` on ``fit`` and keeps those columns (plus the target / kept
columns) on ``transform``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = [
    "FeatureSelectionCorr",
    "FeatureSelectionInfoGain",
    "FeatureSelectionLasso",
    "FeatureSelectionStepwise",
]


def _numeric(data: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]


def _find_correlation(corr: pd.DataFrame, cutoff: float) -> list:
    """Approximate caret::findCorrelation: drop the member of each highly correlated
    pair that has the larger mean absolute correlation with the others."""
    corr = corr.abs()
    cols = list(corr.columns)
    mean_abs = corr.mean()
    remove: set = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in remove or b in remove:
                continue
            if corr.loc[a, b] > cutoff:
                remove.add(a if mean_abs[a] >= mean_abs[b] else b)
    return [c for c in cols if c in remove]


class FeatureSelectionCorr(DalTransform):
    """Remove highly correlated numeric features (|corr| > cutoff).

    A filter method that discards redundant predictors: for each pair whose absolute
    correlation exceeds the cutoff, it drops the one with the larger mean correlation to
    the rest. Reducing collinearity stabilizes linear models and shrinks the feature set
    without consulting the target.

    Porting:
        Mirrors ``trans_feature_selection_corr.R``. Fidelity: approximate.
        - Approximates R ``caret::findCorrelation``: for each highly correlated pair it
          drops the member with the larger mean absolute correlation to the rest.

    References:
        Hall, M. A. (1999). Correlation-based Feature Selection for Machine Learning.
        PhD thesis, University of Waikato.
        Kuhn, M. (2008). Building Predictive Models in R Using the caret Package. Journal
        of Statistical Software, 28(5):1-26.
    """

    def __init__(self, cutoff: float = 0.9, features: list | None = None, keep: list | None = None) -> None:
        self.cutoff = cutoff
        self.features = features
        self.keep = keep
        self.selected: list | None = None
        self.remove: list | None = None

    def fit(self, data, **kwargs) -> FeatureSelectionCorr:
        data = adjust_dataframe(data)
        nums = _numeric(data, self.features if self.features is not None else list(data.columns))
        self.features = nums
        if len(nums) <= 1:
            self.remove, self.selected = [], list(nums)
            return self
        corr = data[nums].corr().fillna(0.0)
        self.remove = _find_correlation(corr, self.cutoff)
        self.selected = [c for c in nums if c not in self.remove]
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        keep = (
            [c for c in data.columns if c not in self.features]
            if self.keep is None
            else [c for c in self.keep if c in data.columns]
        )
        wanted = set(keep) | {c for c in self.selected if c in data.columns}
        return data[[c for c in data.columns if c in wanted]]


class FeatureSelectionLasso(DalTransform):
    """Select predictors with non-zero L1 (lasso) coefficients. Target must be numeric.

    An embedded method: the lasso adds an L1 penalty to least squares, which shrinks some
    coefficients exactly to zero and thereby performs selection while fitting. The
    surviving (non-zero) predictors are kept. The penalty strength is chosen by
    cross-validation. Features are standardized before the lasso fit, matching glmnet's
    internal scaling.

    Porting:
        Mirrors ``trans_feature_selection_lasso.R``. Fidelity: approximate.
        - scikit-learn ``LassoCV`` backend instead of R ``glmnet``; the cross-validated
          penalty and the surviving set may differ from ``cv.glmnet``.

    References:
        Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal
        of the Royal Statistical Society, Series B, 58(1):267-288.
        Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization Paths for
        Generalized Linear Models via Coordinate Descent. Journal of Statistical
        Software, 33(1):1-22.
    """

    def __init__(self, attribute: str, features: list | None = None) -> None:
        self.attribute = attribute
        self.features = features
        self.selected: list | None = None
        self.ranking: pd.DataFrame | None = None

    def fit(self, data, **kwargs) -> FeatureSelectionLasso:
        data = adjust_dataframe(data)
        cols = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = _numeric(data, cols)
        self.features = feats
        if not feats:
            self.selected = []
            return self
        x = StandardScaler().fit_transform(data[feats].to_numpy(dtype=float))
        y = data[self.attribute].to_numpy(dtype=float)
        model = LassoCV(cv=5, max_iter=100000).fit(x, y)
        coefs = model.coef_
        pairs = [(f, abs(c)) for f, c in zip(feats, coefs) if c != 0]
        pairs.sort(key=lambda t: t[1], reverse=True)
        self.selected = [f for f, _ in pairs]
        self.ranking = pd.DataFrame(pairs, columns=["feature", "score"])
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]


class FeatureSelectionInfoGain(DalTransform):
    """Rank features by entropy-based information gain; numeric ones are quantile-binned.

    A filter method that scores each feature by its information gain -- the reduction in
    the target's entropy once the feature is known (mutual information). Higher gain means
    the feature is more informative about the class. Numeric features are quantile-binned
    first so entropy can be computed. This is the splitting criterion behind ID3/C4.5
    decision trees, used here as a univariate ranker.

    Porting:
        Mirrors ``trans_feature_selection_info_gain.R``. Fidelity: adapted.
        - Native entropy computation instead of R ``FSelector``; numeric features are
          quantile-binned (``bins``) before the gain is computed.

    References:
        Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1):81-106.
        Cover, T. M., and Thomas, J. A. (2006). Elements of Information Theory (2nd ed.).
        Wiley.
    """

    def __init__(
        self,
        attribute: str,
        features: list | None = None,
        top: int | None = None,
        cutoff: float = 0.0,
        bins: int = 3,
    ) -> None:
        self.attribute = attribute
        self.features = features
        self.top = top
        self.cutoff = cutoff
        self.bins = bins
        self.selected: list | None = None
        self.ranking: pd.DataFrame | None = None

    @staticmethod
    def _entropy(y: pd.Series) -> float:
        p = y.value_counts(normalize=True)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def _discretize(self, x: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(x):
            return x.astype(object)
        edges = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, self.bins + 1)))
        if len(edges) < 2:
            return pd.Series(["all"] * len(x), index=x.index)
        return pd.cut(x, bins=edges, include_lowest=True, labels=False).astype(object)

    def _info_gain(self, x: pd.Series, y: pd.Series) -> float:
        xb = self._discretize(x)
        total = self._entropy(y)
        cond = 0.0
        n = len(y)
        for level in pd.unique(xb.dropna()):
            idx = xb == level
            cond += (idx.sum() / n) * self._entropy(y[idx])
        return total - cond

    def fit(self, data, **kwargs) -> FeatureSelectionInfoGain:
        data = adjust_dataframe(data)
        y = data[self.attribute].astype(object)
        feats = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = [c for c in feats if c in data.columns]
        self.features = feats
        scores = []
        for f in feats:
            score = self._info_gain(data[f], y)
            scores.append((f, score if np.isfinite(score) else 0.0))
        scores.sort(key=lambda t: t[1], reverse=True)
        self.ranking = pd.DataFrame(scores, columns=["feature", "score"])
        selected = [f for f, s in scores if s >= self.cutoff]
        if self.top is not None:
            selected = selected[: self.top]
        self.selected = selected
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]


class FeatureSelectionStepwise(DalTransform):
    """Forward subset search keeping the subset with the highest adjusted R-squared.

    A wrapper method: predictors are added one at a time (greedy forward selection), each
    step adding the variable that most improves the fit, and the subset with the best
    adjusted R-squared is kept. The adjusted R-squared penalizes extra predictors, guarding
    against adding variables that do not pay for their complexity. Target must be numeric
    (regression).

    Porting:
        Mirrors ``trans_feature_selection_fss.R``. Fidelity: approximate.
        - Greedy forward selection with ``LinearRegression`` keeping the best adjusted
          R-squared, instead of R ``leaps`` best-subset search. (Step 1.6 of the roadmap
          will reconcile ``fss`` vs ``stepwise``.)

    References:
        Hocking, R. R. (1976). The Analysis and Selection of Variables in Linear
        Regression. Biometrics, 32(1):1-49.
        Miller, A. (2002). Subset Selection in Regression (2nd ed.). Chapman & Hall/CRC.
    """

    def __init__(self, attribute: str, features: list | None = None) -> None:
        self.attribute = attribute
        self.features = features
        self.selected: list | None = None

    @staticmethod
    def _adj_r2(x: pd.DataFrame, y: np.ndarray) -> float:
        n, p = x.shape
        if n - p - 1 <= 0:
            return -np.inf
        r2 = LinearRegression().fit(x, y).score(x, y)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def fit(self, data, **kwargs) -> FeatureSelectionStepwise:
        data = adjust_dataframe(data)
        cols = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = _numeric(data, cols)
        self.features = feats
        if not feats:
            self.selected = []
            return self
        y = data[self.attribute].to_numpy(dtype=float)
        remaining = list(feats)
        selected: list = []
        history: list = []
        while remaining:
            best_f, best_score = None, -np.inf
            for f in remaining:
                score = self._adj_r2(data[[*selected, f]], y)
                if score > best_score:
                    best_score, best_f = score, f
            selected.append(best_f)
            remaining.remove(best_f)
            history.append((list(selected), best_score))
        self.selected = max(history, key=lambda t: t[1])[0]
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]
