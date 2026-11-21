"""PAM -- Partitioning Around Medoids (``clu_pam.R`` analog).

PAM clusters around representative data points (medoids) instead of centroids, which
makes it more robust to outliers than k-means. The R package wraps ``cluster::pam``;
this is a native numpy implementation of the classic BUILD + SWAP algorithm, so no
extra dependency is needed.
"""
from __future__ import annotations

import numpy as np

from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_calinski_harabasz,
    metric_davies_bouldin,
    metric_entropy,
    metric_purity,
    metric_silhouette,
)

__all__ = ["ClusterPam"]


def _euclidean_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between rows of ``a`` and rows of ``b``."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))


def _pam(d: np.ndarray, k: int, max_iter: int) -> np.ndarray:
    """Classic PAM (BUILD then SWAP) on a precomputed distance matrix ``d``.

    Returns the indices of the ``k`` selected medoids.
    """
    n = d.shape[0]
    # BUILD: first medoid minimizes total distance, then greedily add the point
    # that most reduces the total distance to the nearest medoid.
    medoids = [int(np.argmin(d.sum(axis=1)))]
    while len(medoids) < k:
        nearest = d[:, medoids].min(axis=1)
        gains = np.maximum(nearest[None, :] - d, 0.0).sum(axis=1)
        gains[medoids] = -np.inf
        medoids.append(int(np.argmax(gains)))
    medoids = np.array(sorted(medoids))

    # SWAP: repeatedly apply the single best medoid/non-medoid swap that lowers cost.
    for _ in range(max_iter):
        current = d[:, medoids].min(axis=1).sum()
        best_delta, best_swap = 0.0, None
        non_medoids = [h for h in range(n) if h not in set(medoids.tolist())]
        for mi in range(k):
            for h in non_medoids:
                trial = medoids.copy()
                trial[mi] = h
                delta = d[:, trial].min(axis=1).sum() - current
                if delta < best_delta:
                    best_delta, best_swap = delta, (mi, h)
        if best_swap is None:
            break
        mi, h = best_swap
        medoids[mi] = h
    return medoids


class ClusterPam(Clusterer):
    """PAM (Partitioning Around Medoids) clustering (native implementation).

    PAM groups the data around ``k`` representative observations called *medoids*,
    minimizing the total distance of each point to its nearest medoid. Because a medoid
    is an actual data point rather than a mean, PAM is more robust to outliers and noise
    than k-means and works with any dissimilarity. It proceeds in two phases: BUILD, a
    greedy selection of the initial medoids, and SWAP, which iteratively exchanges a
    medoid with a non-medoid whenever that lowers the total cost. New data is assigned to
    the nearest medoid.

    Defaults follow the R package: silhouette is the main metric; internal evaluation
    uses silhouette, Davies-Bouldin and Calinski-Harabasz; external evaluation uses
    entropy, purity and the adjusted Rand index.

    References:
        Kaufman, L., & Rousseeuw, P. J. (1990). Finding Groups in Data: An Introduction
        to Cluster Analysis.
        Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and
        validation of cluster analysis.
        Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
        Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
    """

    def __init__(self, k: int = 1, max_iter: int = 100) -> None:
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.medoids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.metric = metric_silhouette
        self.metric_name = "silhouette"
        self.eval_internal = [metric_silhouette, metric_davies_bouldin, metric_calinski_harabasz]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def fit(self, data, **kwargs) -> ClusterPam:
        data = self._prepare_fit(data)
        x = data[self.xnames].to_numpy(dtype=float)
        d = _euclidean_matrix(x, x)
        medoid_idx = _pam(d, self.k, self.max_iter)
        self.medoids_ = x[medoid_idx]
        self.labels_ = np.argmin(d[:, medoid_idx], axis=1)
        self.model = {"medoids": self.medoids_, "clustering": self.labels_}
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        x = self._features(data)
        if len(x) == len(self.train_data) and np.array_equal(
            x, self.train_data[self.xnames].to_numpy(dtype=float)
        ):
            return self.labels_
        return np.argmin(_euclidean_matrix(x, self.medoids_), axis=1)
