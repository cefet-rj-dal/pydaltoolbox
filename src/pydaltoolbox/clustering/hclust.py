"""Hierarchical (agglomerative) clustering (``clu_hclust.R`` analog)."""
from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_calinski_harabasz,
    metric_davies_bouldin,
    metric_entropy,
    metric_purity,
    metric_silhouette,
)

__all__ = ["ClusterHclust"]


class ClusterHclust(Clusterer):
    """Agglomerative hierarchical clustering (wraps scikit-learn).

    Cuts the dendrogram at ``k`` clusters, or at a distance threshold ``h`` when given.
    Features are standardized first by default (as in the R package). Does not support
    clustering new data after fit.
    """

    def __init__(
        self,
        k: int = 2,
        h: float | None = None,
        method: str = "ward",
        dist: str = "euclidean",
        scale: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.h = h
        self.method = method
        self.dist = dist
        self.scale = scale
        self.labels_ = None
        self.metric = metric_silhouette
        self.metric_name = "silhouette"
        self.eval_internal = [metric_silhouette, metric_davies_bouldin, metric_calinski_harabasz]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def fit(self, data, **kwargs) -> ClusterHclust:
        data = self._prepare_fit(data)
        x = data[self.xnames].to_numpy(dtype=float)
        if self.scale:
            x = StandardScaler().fit_transform(x)
        metric = "euclidean" if self.method == "ward" else self.dist
        if self.h is not None:
            self.model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.h, linkage=self.method, metric=metric
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=self.k, linkage=self.method, metric=metric
            )
        self.labels_ = self.model.fit_predict(x)
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        return self.labels_
