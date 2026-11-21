"""Smoothing by binning / quantization.

Python counterparts of ``trans_smoothing*.R``. A smoothing reduces noise by splitting a
numeric vector into ``n`` bins and replacing each value with its bin mean. Strategies
differ in how the bin boundaries are chosen: equal interval (robust boxplot bounds),
equal frequency (quantiles), k-means quantization, or class-aware clustering.

When ``n`` is a collection of candidates, ``fit`` tunes it: the non-supervised strategies
pick the elbow (max curvature) of the MSE curve; the class-aware one minimizes entropy.
(We pick the chosen ``n`` and refit with it — the R base keeps the last candidate's
boundaries, which looks like a bug.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..adjust import adjust_dataframe
from ..base import DalTransform
from .curvature import FitCurvatureMax

__all__ = [
    "Smoothing",
    "SmoothingCluster",
    "SmoothingFreq",
    "SmoothingInter",
    "SmoothingQuantization",
]

_HUGE = float(np.finfo(float).max)


def _is_candidates(n) -> bool:
    return isinstance(n, (list, tuple, range, np.ndarray))


def _entropy_eval(bins, attribute) -> dict:
    """Weighted entropy of a reference label across bins (log base 2)."""
    df = pd.DataFrame({"x": pd.Series(np.asarray(bins)).astype(str),
                       "y": pd.Series(np.asarray(attribute)).astype(str)})
    n = len(df)
    rows, total = [], 0.0
    for xk, group in df.groupby("x", observed=True):
        p = group["y"].value_counts(normalize=True)
        p = p[p > 0]
        ce = float(-(p * np.log2(p)).sum())
        rows.append({"x": xk, "ce": ce, "qtd": len(group)})
        total += ce * len(group) / n
    return {"entropy": total, "entropy_clusters": pd.DataFrame(rows)}


class Smoothing(DalTransform):
    """Base smoothing: bin a numeric vector and replace values by bin means.

    Smoothing by binning is a discretization technique that reduces noise and minor
    fluctuations: the value range is partitioned into ``n`` bins and every value is
    replaced by its bin's mean. Subclasses differ only in how the bin boundaries are
    placed. It is also a simple way to turn a continuous attribute into a small number of
    representative levels.

    Subclasses implement :meth:`_compute_interval`. ``n`` may be an int or a collection
    of candidate ints (tuned at fit).

    Porting:
        Mirrors ``trans_smoothing.R``. Fidelity: adapted.
        - R-bug fix: when ``n`` is tuned, this refits with the chosen ``n``; the R base
          keeps the last candidate's boundaries instead, which looks like a bug.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Discretization)
    """

    def __init__(self, n) -> None:
        self.n = n
        self.interval: np.ndarray | None = None
        self.interval_adj: np.ndarray | None = None

    @staticmethod
    def _values(data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data.select_dtypes("number").iloc[:, 0].to_numpy(dtype=float)
        if isinstance(data, pd.Series):
            return data.to_numpy(dtype=float)
        return np.asarray(data, dtype=float).ravel()

    def _compute_interval(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _fit_single(self, values: np.ndarray) -> Smoothing:
        interval = np.asarray(self._compute_interval(values), dtype=float)
        interval[0] = float(values.min())
        interval[-1] = float(values.max())
        adj = interval.copy()
        adj[0] = -_HUGE
        adj[-1] = _HUGE
        self.interval = interval
        self.interval_adj = adj
        return self

    def _tune(self, values: np.ndarray) -> Smoothing:
        candidates = list(self.n)
        mses = []
        for i in candidates:
            self.n = i
            self._fit_single(values)
            vm = self.transform(values)
            mses.append(float(np.mean((values - vm) ** 2)))
        pos = int(FitCurvatureMax().transform(np.array(mses))["x"].iloc[0]) - 1
        self.n = candidates[pos]
        self._fit_single(values)
        return self

    def fit(self, data, **kwargs) -> Smoothing:
        values = self._values(data)
        if _is_candidates(self.n):
            return self._tune(values)
        return self._fit_single(values)

    def transform(self, data, **kwargs) -> np.ndarray:
        v = pd.Series(self._values(data))
        edges = np.unique(self.interval_adj)
        if len(edges) < 2:
            return v.to_numpy()
        codes = pd.cut(v, bins=edges, include_lowest=True, labels=False)
        return v.groupby(codes).transform("mean").to_numpy()

    def evaluate(self, data, attribute, **kwargs) -> dict:
        return _entropy_eval(data, attribute)


class SmoothingInter(Smoothing):
    """Equal-interval bins with robust (boxplot-whisker) bounds. Mirrors ``smoothing_inter``.

    Equal-width binning: the range is split into ``n`` intervals of equal size. Here the
    range is taken from robust boxplot whiskers rather than the raw min/max, so extreme
    outliers do not stretch the bins.

    Porting:
        Mirrors ``trans_smoothing_inter.R``. Fidelity: exact.
        - Inherits the tuning-refit adaptation from :class:`Smoothing`.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Discretization)
    """

    def _compute_interval(self, values: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo = values[values >= q1 - 1.5 * iqr]
        hi = values[values <= q3 + 1.5 * iqr]
        bimin = float(lo.min()) if lo.size else float(values.min())
        bimax = float(hi.max()) if hi.size else float(values.max())
        if bimin == bimax:
            bimin, bimax = float(values.min()), float(values.max())
        return np.linspace(bimin, bimax, self.n + 1)


class SmoothingFreq(Smoothing):
    """Equal-frequency (quantile) bins. Mirrors ``smoothing_freq``.

    Equal-frequency (equal-depth) binning: boundaries are placed at quantiles so each bin
    holds roughly the same number of observations. This adapts to skewed distributions
    better than equal-width bins.

    Porting:
        Mirrors ``trans_smoothing_freq.R``. Fidelity: exact.
        - Inherits the tuning-refit adaptation from :class:`Smoothing`.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Discretization)
    """

    def _compute_interval(self, values: np.ndarray) -> np.ndarray:
        return np.quantile(values, np.linspace(0, 1, self.n + 1))


class SmoothingQuantization(Smoothing):
    """k-means quantization of the values. Mirrors ``smoothing_quantization``.

    Clusters the one-dimensional values with k-means and derives bin boundaries from the
    cluster centers (their midpoints). Bins therefore follow the natural grouping of the
    data rather than a fixed width or frequency.

    Porting:
        Mirrors ``trans_smoothing_quantization.R``. Fidelity: approximate.
        - scikit-learn ``KMeans`` backend (RNG/init differ from R); inherits the
          tuning-refit adaptation from :class:`Smoothing`.

    References:
        MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate
        Observations.
    """

    def __init__(self, n, seed: int | None = None) -> None:
        super().__init__(n)
        self.seed = seed

    def _compute_interval(self, values: np.ndarray) -> np.ndarray:
        km = KMeans(n_clusters=self.n, n_init=10, random_state=self.seed)
        km.fit(values.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        mids = (centers[:-1] + centers[1:]) / 2 if self.n > 1 else np.array([])
        return np.concatenate([[values.min()], mids, [values.max()]])


class SmoothingCluster(Smoothing):
    """Class-aware clustering of value + one-hot label. Mirrors ``smoothing_cluster``.

    A supervised discretization: the value is clustered together with the one-hot encoded
    class label, so bins tend to group together values that share a class. This yields
    class-coherent intervals, which is why tuning minimizes label entropy across bins.

    ``fit`` takes a DataFrame with one numeric column and the class column named
    ``class_label``. Tuning (``n`` as candidates) minimizes entropy.

    Porting:
        Mirrors ``trans_smoothing_cluster.R``. Fidelity: approximate.
        - scikit-learn ``KMeans`` backend (RNG/init differ from R); tuning minimizes
          label entropy across bins.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Discretization)
    """

    def __init__(self, class_label: str, n, seed: int | None = None) -> None:
        super().__init__(n)
        self.class_label = class_label
        self.seed = seed

    def _prepare(self, data):
        data = adjust_dataframe(data)
        if self.class_label not in data.columns:
            raise ValueError(f"smoothing_cluster: '{self.class_label}' not in data.")
        features = [c for c in data.columns if c != self.class_label]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
        if len(numeric) != 1:
            raise ValueError("smoothing_cluster: need exactly one numeric attribute.")
        return data[numeric[0]].to_numpy(dtype=float), data[self.class_label]

    def _fit_values_labels(self, values: np.ndarray, labels) -> SmoothingCluster:
        std = values.std()
        scaled = (values - values.mean()) / std if std > 0 else np.zeros_like(values)
        onehot = pd.get_dummies(pd.Series(labels).astype(str)).to_numpy(dtype=float)
        x = np.column_stack([scaled, onehot])
        km = KMeans(n_clusters=self.n, n_init=10, random_state=self.seed).fit(x)
        centers = np.sort(pd.Series(values).groupby(km.labels_).mean().to_numpy())
        cuts = (centers[:-1] + centers[1:]) / 2 if len(centers) > 1 else np.array([])
        self.interval = np.concatenate([[values.min()], cuts, [values.max()]])
        self.interval[0], self.interval[-1] = float(values.min()), float(values.max())
        adj = self.interval.copy()
        adj[0], adj[-1] = -_HUGE, _HUGE
        self.interval_adj = adj
        return self

    def fit(self, data, **kwargs) -> SmoothingCluster:
        values, labels = self._prepare(data)
        if _is_candidates(self.n):
            candidates = list(self.n)
            entropies = []
            for i in candidates:
                self.n = i
                self._fit_values_labels(values, labels)
                bins = pd.cut(pd.Series(values), bins=np.unique(self.interval_adj),
                              include_lowest=True, labels=False)
                entropies.append(_entropy_eval(bins, labels)["entropy"])
            self.n = candidates[int(np.argmin(entropies))]
        return self._fit_values_labels(values, labels)
