"""Clustering metrics and selectors (``clu_utils.R`` analog).

Each metric is a callable returning a dict ``{metric, value, goal, type}`` so the
``Clusterer.evaluate`` output has a uniform shape. Internal metrics score a partition
from the data alone; external metrics compare it against a reference label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

__all__ = [
    "metric_adjusted_rand_index",
    "metric_calinski_harabasz",
    "metric_davies_bouldin",
    "metric_entropy",
    "metric_purity",
    "metric_silhouette",
    "metric_wcss",
    "selector_best",
    "selector_elbow",
]


def _result(metric, value, goal, type_):
    return {"metric": metric, "value": float(value), "goal": goal, "type": type_}


def _xy(data, cluster):
    x = data.to_numpy(dtype=float) if isinstance(data, (pd.DataFrame, pd.Series)) else np.asarray(data, dtype=float)
    labels = np.asarray(cluster)
    return x, labels


def _valid_internal(x, labels) -> bool:
    n_labels = len(np.unique(labels))
    return 2 <= n_labels <= len(x) - 1


def metric_wcss(data=None, cluster=None, **_):
    """Total within-cluster sum of squares (lower is better)."""
    x, labels = _xy(data, cluster)
    total = 0.0
    for c in np.unique(labels):
        pts = x[labels == c]
        if len(pts):
            total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return _result("wcss", total, "minimize", "internal")


def metric_silhouette(data=None, cluster=None, **_):
    """Mean silhouette coefficient (higher is better)."""
    x, labels = _xy(data, cluster)
    value = silhouette_score(x, labels) if _valid_internal(x, labels) else np.nan
    return _result("silhouette", value, "maximize", "internal")


def metric_davies_bouldin(data=None, cluster=None, **_):
    """Davies-Bouldin index (lower is better)."""
    x, labels = _xy(data, cluster)
    value = davies_bouldin_score(x, labels) if _valid_internal(x, labels) else np.nan
    return _result("davies_bouldin", value, "minimize", "internal")


def metric_calinski_harabasz(data=None, cluster=None, **_):
    """Calinski-Harabasz score (higher is better)."""
    x, labels = _xy(data, cluster)
    value = calinski_harabasz_score(x, labels) if _valid_internal(x, labels) else np.nan
    return _result("calinski_harabasz", value, "maximize", "internal")


def _entropy(labels) -> float:
    p = pd.Series(labels).value_counts(normalize=True)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def metric_entropy(cluster=None, attribute=None, **_):
    """External clustering entropy against a reference label (lower is better)."""
    labels = np.asarray(cluster)
    y = pd.Series(np.asarray(attribute))
    n = len(y)
    total = 0.0
    for c in np.unique(labels):
        yc = y[labels == c]
        total += len(yc) / n * _entropy(yc)
    return _result("entropy", total, "minimize", "external")


def metric_purity(cluster=None, attribute=None, **_):
    """Cluster purity against a reference label (higher is better)."""
    labels = np.asarray(cluster)
    y = pd.Series(np.asarray(attribute))
    n = len(y)
    total = sum(y[labels == c].value_counts().iloc[0] for c in np.unique(labels))
    return _result("purity", total / n, "maximize", "external")


def metric_adjusted_rand_index(cluster=None, attribute=None, **_):
    """Adjusted Rand index against a reference label (higher is better)."""
    value = adjusted_rand_score(np.asarray(attribute), np.asarray(cluster))
    return _result("adjusted_rand_index", value, "maximize", "external")


def selector_best(values, goal: str = "maximize") -> int:
    """Index of the best value (max if maximizing, min if minimizing); NaNs ignored."""
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return 0
    return int(np.nanargmax(arr) if goal == "maximize" else np.nanargmin(arr))


def selector_elbow(values, goal: str = "maximize") -> int:
    """Index of the elbow of the value curve (maximum/minimum curvature)."""
    from ..transform.curvature import FitCurvatureMax, FitCurvatureMin

    arr = np.asarray(values, dtype=float)
    picker = FitCurvatureMax() if goal == "maximize" else FitCurvatureMin()
    return int(picker.transform(arr)["x"].iloc[0]) - 1
