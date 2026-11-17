"""k-means clustering (``clu_kmeans.R`` analog)."""
from __future__ import annotations

from sklearn.cluster import KMeans

from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_calinski_harabasz,
    metric_davies_bouldin,
    metric_entropy,
    metric_purity,
    metric_silhouette,
)

__all__ = ["ClusterKmeans"]


class ClusterKmeans(Clusterer):
    """k-means clustering (wraps scikit-learn's KMeans).

    Defaults follow the R package: silhouette is the main metric; internal evaluation
    uses silhouette, Davies-Bouldin and Calinski-Harabasz; external evaluation uses
    entropy, purity and the adjusted Rand index.
    """

    def __init__(self, k: int = 1, seed: int | None = None) -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.metric = metric_silhouette
        self.metric_name = "silhouette"
        self.eval_internal = [metric_silhouette, metric_davies_bouldin, metric_calinski_harabasz]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def fit(self, data, **kwargs) -> ClusterKmeans:
        data = self._prepare_fit(data)
        self.model = KMeans(n_clusters=self.k, random_state=self.seed, n_init=10)
        self.model.fit(data[self.xnames].to_numpy(dtype=float))
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        return self.model.predict(self._features(data))
