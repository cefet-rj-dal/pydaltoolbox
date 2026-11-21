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

    k-means partitions the data into ``k`` clusters by minimizing the total
    within-cluster sum of squares (WCSS) around ``k`` centroids, alternating between
    assigning each point to its nearest centroid and recomputing centroids as cluster
    means. It is fast and a natural choice for compact, roughly spherical clusters.

    Defaults follow the R package: silhouette is the main metric; internal evaluation
    uses silhouette, Davies-Bouldin and Calinski-Harabasz; external evaluation uses
    entropy, purity and the adjusted Rand index. WCSS is not used as the main metric
    because it decreases mechanically as ``k`` grows, making it a weak standalone
    criterion for comparing partitions.

    References:
        MacQueen, J. (1967). Some Methods for Classification and Analysis of
        Multivariate Observations.
        Lloyd, S. (1982). Least squares quantization in PCM.
        Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation
        and validation of cluster analysis.
        Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.
        Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis.
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
