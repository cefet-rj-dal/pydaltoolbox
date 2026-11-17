"""Gaussian mixture model clustering (``clu_gmm.R`` analog)."""
from __future__ import annotations

from sklearn.mixture import GaussianMixture

from .base import Clusterer
from .metrics import (
    metric_adjusted_rand_index,
    metric_entropy,
    metric_loglik,
    metric_purity,
)

__all__ = ["ClusterGmm"]


class ClusterGmm(Clusterer):
    """Gaussian mixture model clustering (wraps scikit-learn's GaussianMixture).

    If the number of components ``g`` is not given, it is chosen by minimizing the BIC
    over 1..10 components (the mclust-style model selection). Supports clustering new
    data (the fitted mixture can assign posteriors to unseen points).
    """

    def __init__(self, g: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.g = g
        self.seed = seed
        self.loglik = None
        self.metric = metric_loglik
        self.metric_name = "loglik"
        self.eval_internal = [metric_loglik]
        self.eval_external = [metric_entropy, metric_purity, metric_adjusted_rand_index]

    def fit(self, data, **kwargs) -> ClusterGmm:
        data = self._prepare_fit(data)
        x = data[self.xnames].to_numpy(dtype=float)
        if self.g is None:
            best = None
            for g in range(1, 11):
                model = GaussianMixture(n_components=g, random_state=self.seed).fit(x)
                bic = model.bic(x)
                if best is None or bic < best[0]:
                    best = (bic, g, model)
            self.g, self.model = best[1], best[2]
        else:
            self.model = GaussianMixture(n_components=self.g, random_state=self.seed).fit(x)
        self.loglik = float(self.model.score(x) * len(x))
        return self

    def cluster(self, data, **kwargs):
        self._require_fitted()
        return self.model.predict(self._features(data))
