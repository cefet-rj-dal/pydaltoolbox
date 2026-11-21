"""Feature-selection transforms.

Python counterparts of the R ``trans_feature_selection_*`` wrappers:

* ``FeatureSelectionCorr``    - drop highly correlated numeric features (caret::findCorrelation)
* ``FeatureSelectionLasso``   - keep predictors with non-zero L1 coefficients (glmnet lasso)
* ``FeatureSelectionInfoGain``- rank by entropy-based information gain (quantile-binned)
* ``FeatureSelectionFss``     - forward subset search maximizing adjusted R-squared (leaps)
* ``FeatureSelectionStepwise``- stepwise GLM selection by AIC (stats::step)
* ``FeatureSelectionRelief``  - rank features with the ReliefF weighting scheme

Each learns ``selected`` on ``fit`` and keeps those columns (plus the target / kept
columns) on ``transform``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = [
    "FeatureSelectionCorr",
    "FeatureSelectionFss",
    "FeatureSelectionInfoGain",
    "FeatureSelectionLasso",
    "FeatureSelectionRelief",
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


class FeatureSelectionFss(DalTransform):
    """Forward subset search keeping the subset with the highest adjusted R-squared.

    A wrapper method: predictors are added one at a time (greedy forward selection), each
    step adding the variable that most improves the fit, and the subset with the best
    adjusted R-squared is kept. The adjusted R-squared penalizes extra predictors, guarding
    against adding variables that do not pay for their complexity. Target must be numeric
    (regression).

    Porting:
        Mirrors ``trans_feature_selection_fss.R``. Fidelity: approximate.
        - Greedy forward selection with ``LinearRegression`` keeping the best adjusted
          R-squared, instead of R ``leaps::regsubsets`` best-subset search. Both optimize
          adjusted R-squared over nested forward models, so the selected subset usually
          agrees; it may differ when a non-greedy subset wins.

    References:
        Hocking, R. R. (1976). The Analysis and Selection of Variables in Linear
        Regression. Biometrics, 32(1):1-49.
        Miller, A. (2002). Subset Selection in Regression (2nd ed.). Chapman & Hall/CRC.
    """

    def __init__(self, attribute: str, features: list | None = None) -> None:
        self.attribute = attribute
        self.features = features
        self.selected: list | None = None
        self.ranking: pd.DataFrame | None = None

    @staticmethod
    def _adj_r2(x: pd.DataFrame, y: np.ndarray) -> float:
        n, p = x.shape
        if n - p - 1 <= 0:
            return -np.inf
        r2 = LinearRegression().fit(x, y).score(x, y)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def fit(self, data, **kwargs) -> FeatureSelectionFss:
        data = adjust_dataframe(data)
        cols = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = _numeric(data, cols)
        self.features = feats
        if not feats:
            self.selected = []
            self.ranking = pd.DataFrame({"feature": [], "score": []})
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
        self.ranking = pd.DataFrame(
            {"feature": self.selected, "score": list(range(1, len(self.selected) + 1))}
        )
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]


class FeatureSelectionStepwise(DalTransform):
    """Stepwise selection over a generalized linear model, minimizing the AIC.

    A wrapper method mirroring R ``stats::step``: starting from the null (forward/both) or
    full (backward) model, predictors are added and/or dropped one at a time, each step
    taking the move that most lowers Akaike's Information Criterion
    ``AIC = 2k - 2*logLik`` (``k`` = number of fitted parameters). AIC rewards goodness of
    fit while penalizing model size, so the search stops at a parsimonious model. With the
    default ``binomial`` family the target is a binary outcome (logistic regression); the
    ``gaussian`` family handles a numeric target (ordinary linear regression).

    Parameters
    ----------
    direction : {"forward", "backward", "both"}
        Search direction, as in ``stats::step``.
    family : {"binomial", "gaussian"}
        GLM family. ``binomial`` (default) = logistic regression for a binary target;
        ``gaussian`` = linear regression for a numeric target.

    Porting:
        Mirrors ``trans_feature_selection_stepwise.R``. Fidelity: adapted.
        - Native implementation: the GLM is fit with scikit-learn
          (``LogisticRegression(penalty=None)`` for ``binomial``, ``LinearRegression``
          for ``gaussian``) and the AIC is computed directly from the log-likelihood,
          instead of R ``stats::glm`` + ``stats::step``. Avoids a statsmodels dependency.
        - Only the ``binomial`` and ``gaussian`` families are supported (the two the R
          default and examples use).

    References:
        Akaike, H. (1974). A New Look at the Statistical Model Identification. IEEE
        Transactions on Automatic Control, 19(6):716-723.
        Hastie, T., Tibshirani, R., and Friedman, J. (2009). The Elements of Statistical
        Learning (2nd ed.). Springer.
    """

    def __init__(
        self,
        attribute: str,
        features: list | None = None,
        direction: str = "forward",
        family: str = "binomial",
    ) -> None:
        if direction not in ("forward", "backward", "both"):
            raise ValueError("direction must be 'forward', 'backward', or 'both'.")
        if family not in ("binomial", "gaussian"):
            raise ValueError("family must be 'binomial' or 'gaussian'.")
        self.attribute = attribute
        self.features = features
        self.direction = direction
        self.family = family
        self.selected: list | None = None
        self.ranking: pd.DataFrame | None = None

    def _aic(self, data: pd.DataFrame, feats: list, y: np.ndarray) -> float:
        """AIC of the GLM regressing ``y`` on ``feats`` (intercept always included)."""
        n = len(y)
        if feats:
            x = data[feats].to_numpy(dtype=float)
        else:
            x = np.empty((n, 0), dtype=float)
        if self.family == "binomial":
            # k = intercept + one coefficient per feature
            k = len(feats) + 1
            if feats:
                model = LogisticRegression(penalty=None, max_iter=1000)
                model.fit(x, y)
                p = model.predict_proba(x)[:, 1]
            else:
                p = np.full(n, float(np.mean(y)))
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)
            loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:  # gaussian
            # k = intercept + coefficients + residual variance
            k = len(feats) + 2
            if feats:
                model = LinearRegression()
                model.fit(x, y)
                resid = y - model.predict(x)
            else:
                resid = y - float(np.mean(y))
            rss = float(np.sum(resid**2))
            rss = max(rss, 1e-12)
            loglik = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
        return 2 * k - 2 * loglik

    def fit(self, data, **kwargs) -> FeatureSelectionStepwise:
        data = adjust_dataframe(data)
        cols = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = _numeric(data, cols)
        self.features = feats

        raw_y = data[self.attribute]
        if self.family == "binomial":
            # encode the target as 0/1 (factor-like, first level = 0)
            levels = pd.unique(raw_y)
            if len(levels) != 2:
                raise ValueError("binomial family requires a binary target.")
            order = sorted(levels, key=lambda v: str(v))
            y = raw_y.map({order[0]: 0, order[1]: 1}).to_numpy(dtype=float)
        else:
            y = raw_y.to_numpy(dtype=float)

        if not feats:
            self.selected = []
            self.ranking = pd.DataFrame({"feature": [], "score": []})
            return self

        if self.direction == "backward":
            current = list(feats)
        else:
            current = []
        best_aic = self._aic(data, current, y)

        improved = True
        while improved:
            improved = False
            candidates: list = []
            if self.direction in ("forward", "both"):
                candidates += [("add", f) for f in feats if f not in current]
            if self.direction in ("backward", "both"):
                candidates += [("drop", f) for f in current]
            best_move, best_move_aic = None, best_aic
            for move, f in candidates:
                trial = current + [f] if move == "add" else [c for c in current if c != f]
                aic = self._aic(data, trial, y)
                if aic < best_move_aic - 1e-8:
                    best_move_aic, best_move = aic, (move, f)
            if best_move is not None:
                move, f = best_move
                current = current + [f] if move == "add" else [c for c in current if c != f]
                best_aic = best_move_aic
                improved = True

        self.selected = [f for f in feats if f in current]
        self.ranking = pd.DataFrame(
            {"feature": self.selected, "score": list(range(1, len(self.selected) + 1))}
        )
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]


class FeatureSelectionRelief(DalTransform):
    """Rank and select features with the ReliefF weighting scheme.

    An instance-based filter method. For each of ``m`` randomly sampled instances, the
    algorithm finds its nearest neighbour of the same class (the *nearest hit*) and its
    nearest neighbour of a different class (the *nearest miss*) in the normalized feature
    space, then updates each feature's weight by ``-|x_i - nearHit| + |x_i - nearMiss|``.
    A feature scores high when it keeps same-class instances close and pushes
    different-class instances apart, so Relief captures feature relevance including local
    feature interactions that univariate filters miss. Features are min-max normalized and
    categorical ones are integer-encoded before distances are computed. The target is
    treated as a class factor (two or more classes).

    Parameters
    ----------
    top : int, optional
        Keep only the ``top`` highest-weighted features.
    cutoff : float, optional
        Keep only features whose weight is ``>= cutoff``.
    m : int
        Number of sampled instances used for the weight updates (default 50).

    Porting:
        Mirrors ``trans_feature_selection_relief.R``. Fidelity: adapted.
        - Native numpy reimplementation of the R simplified Relief loop (no ``skrebate``
          / CORElearn dependency).
        - Random instance sampling uses numpy's RNG, so the exact weights differ run to
          run (and from R) unless a seed is fixed; the feature ranking is stable.

    References:
        Kira, K., and Rendell, L. A. (1992). A Practical Approach to Feature Selection.
        In Proceedings of the Ninth International Workshop on Machine Learning, 249-256.
        Kononenko, I. (1994). Estimating Attributes: Analysis and Extensions of RELIEF.
        In Proceedings of the European Conference on Machine Learning, 171-182.
    """

    def __init__(
        self,
        attribute: str,
        features: list | None = None,
        top: int | None = None,
        cutoff: float | None = None,
        m: int = 50,
    ) -> None:
        self.attribute = attribute
        self.features = features
        self.top = top
        self.cutoff = cutoff
        self.m = m
        self.selected: list | None = None
        self.ranking: pd.DataFrame | None = None

    def fit(self, data, **kwargs) -> FeatureSelectionRelief:
        data = adjust_dataframe(data)
        if self.attribute not in data.columns:
            raise ValueError("FeatureSelectionRelief: attribute not found in data.")
        y = data[self.attribute].astype("category")
        if len(y.cat.categories) < 2:
            raise ValueError("FeatureSelectionRelief: target must have at least two classes.")
        y_codes = y.cat.codes.to_numpy()

        feats = self.features if self.features is not None else [c for c in data.columns if c != self.attribute]
        feats = [c for c in feats if c in data.columns]
        self.features = feats
        if not feats:
            self.ranking = pd.DataFrame({"feature": [], "score": []})
            self.selected = []
            return self

        # integer-encode categorical features, keep numeric as is
        cols = []
        for f in feats:
            col = data[f]
            if pd.api.types.is_numeric_dtype(col):
                cols.append(col.to_numpy(dtype=float))
            else:
                cols.append(col.astype("category").cat.codes.to_numpy(dtype=float))
        x = np.column_stack(cols)

        # min-max normalize each feature (constant columns -> denominator 1)
        mins = np.nanmin(x, axis=0)
        maxs = np.nanmax(x, axis=0)
        den = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        x = (x - mins) / den

        n = x.shape[0]
        m = min(self.m, n)
        rng = np.random.default_rng()
        idxs = rng.choice(n, size=m, replace=False)
        w = np.zeros(x.shape[1])

        all_idx = np.arange(n)
        for i in idxs:
            xi = x[i]
            yi = y_codes[i]
            d = np.sum((x - xi) ** 2, axis=1)
            same = all_idx[(y_codes == yi) & (all_idx != i)]
            diff = all_idx[y_codes != yi]
            if len(same) == 0 or len(diff) == 0:
                continue
            nh = same[np.argmin(d[same])]
            nm = diff[np.argmin(d[diff])]
            w += -np.abs(x[i] - x[nh]) + np.abs(x[i] - x[nm])

        w = w / m
        order = np.argsort(-w, kind="stable")
        ranking = pd.DataFrame(
            {"feature": [feats[j] for j in order], "score": [float(w[j]) for j in order]}
        )

        selected = list(ranking["feature"])
        if self.cutoff is not None:
            selected = [f for f, s in zip(ranking["feature"], ranking["score"]) if s >= self.cutoff]
        if self.top is not None:
            selected = selected[: self.top]

        self.ranking = ranking
        self.selected = selected
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        if self.selected is None:
            raise ValueError("FeatureSelectionRelief: call fit() before transform().")
        wanted = {self.attribute, *self.selected}
        return data[[c for c in data.columns if c in wanted]]
