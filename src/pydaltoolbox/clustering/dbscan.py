"""DBSCAN density-based clustering (``clu_dbscan.R`` analog)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from ..transform.curvature import FitCurvatureMax
from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_entropy,
    metric_noise_points,
    metric_purity,
)

__all__ = ["ClusterDbscan"]


class ClusterDbscan(Clusterer):
    """DBSCAN clustering (wraps scikit-learn's DBSCAN).

    Finds dense regions and labels sparse points as noise (-1). If ``eps`` is not given,
    it is estimated from the elbow of the sorted k-nearest-neighbor distance curve, as
    in the R package. Does not support clustering new data after fit.
    """

    def __init__(self, min_pts: int = 3, eps: float | None = None) -> None:
        super().__init__()
        self.min_pts = min_pts
        self.eps = eps
        self.labels_ = None
        self.metric = metric_noise_points
        self.metric_name = "noise_points"
        self.eval_internal = [metric_noise_points]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def _estimate_eps(self, x: np.ndarray) -> float:
        nn = NearestNeighbors(n_neighbors=self.min_pts).fit(x)
        dist, _ = nn.kneighbors(x)
        kdist = np.sort(dist[:, -1])
        return float(FitCurvatureMax().transform(kdist)["y"].iloc[0])

    def fit(self, data, **kwargs) -> ClusterDbscan:
        data = self._prepare_fit(data)
        x = data[self.xnames].to_numpy(dtype=float)
        if self.eps is None:
            self.eps = self._estimate_eps(x)
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_pts)
        self.labels_ = self.model.fit_predict(x)
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        return self.labels_
