"""Fuzzy c-means clustering (``clu_cmeans.R`` analog).

Fuzzy c-means produces a soft membership of each point to every cluster. The R package
wraps ``e1071::cmeans``; this is a native numpy implementation, so no extra dependency
is needed. The hard assignment (argmax of the membership) is returned by ``cluster``;
the full membership matrix is kept in the ``membership_`` attribute.
"""
from __future__ import annotations

import numpy as np

from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_entropy,
    metric_purity,
    metric_silhouette,
    metric_withinerror,
)

__all__ = ["ClusterCmeans"]


def _distance(x: np.ndarray, centers: np.ndarray, dist: str) -> np.ndarray:
    """Distance from each row of ``x`` to each center; Euclidean or Manhattan."""
    diff = x[:, None, :] - centers[None, :, :]
    if dist == "manhattan":
        return np.abs(diff).sum(axis=2)
    return np.sqrt((diff ** 2).sum(axis=2))


class ClusterCmeans(Clusterer):
    """Fuzzy c-means clustering (native implementation).

    Fuzzy c-means is a soft-partition method: rather than assigning each point to a
    single cluster, it gives every point a graded membership in each of the ``centers``
    clusters. It minimizes a weighted within-cluster objective in which memberships are
    raised to the fuzziness exponent ``m`` (m > 1); larger ``m`` yields softer, more
    overlapping partitions. The algorithm alternates between updating cluster centers as
    membership-weighted means and recomputing memberships from the distances to those
    centers, until convergence. The hard assignment (argmax of the membership) is
    returned by ``cluster``; the full membership matrix is kept in ``membership_``.

    Defaults follow the R package: silhouette is the main metric; internal evaluation
    uses silhouette and the within-cluster fuzzy error; external evaluation uses
    entropy, purity and the adjusted Rand index. Does not support clustering new data
    after fit.

    References:
        Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function
        Algorithms.
        Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and Its Use in
        Detecting Compact Well-Separated Clusters.
        Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and
        validation of cluster analysis.
    """

    def __init__(
        self,
        centers: int = 2,
        m: float = 2.0,
        iter: int = 100,
        dist: str = "euclidean",
        tol: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.centers = centers
        self.m = m
        self.iter = iter
        self.dist = dist
        self.tol = tol
        self.seed = seed
        self.membership_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.withinerror: float | None = None
        self.metric = metric_silhouette
        self.metric_name = "silhouette"
        self.eval_internal = [metric_silhouette, metric_withinerror]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def fit(self, data, **kwargs) -> ClusterCmeans:
        data = self._prepare_fit(data)
        x = data[self.xnames].to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed)
        u = rng.random((len(x), self.centers))
        u /= u.sum(axis=1, keepdims=True)

        power = 2.0 / (self.m - 1.0)
        d = None
        for _ in range(self.iter):
            um = u ** self.m
            centers = (um.T @ x) / um.sum(axis=0)[:, None]
            d = np.fmax(_distance(x, centers, self.dist), 1e-10)
            inv = 1.0 / (d ** power)
            u_new = inv / inv.sum(axis=1, keepdims=True)
            if np.linalg.norm(u_new - u) < self.tol:
                u = u_new
                break
            u = u_new

        self.membership_ = u
        self.labels_ = np.argmax(u, axis=1)
        self.withinerror = float(((u ** self.m) * (d ** 2)).sum())
        self.model = {"centers": centers, "membership": u, "cluster": self.labels_}
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        x = self._features(data)
        if len(x) != len(self.train_data) or not np.array_equal(
            x, self.train_data[self.xnames].to_numpy(dtype=float)
        ):
            raise ValueError(
                "ClusterCmeans does not support clustering new data after fit()."
            )
        return self.labels_
