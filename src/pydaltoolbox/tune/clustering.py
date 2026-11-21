"""Clustering tuning via an intrinsic metric (``clu_tune.R`` analog)."""
from __future__ import annotations

from .base import DalTune

__all__ = ["CluTune"]


class CluTune(DalTune):
    """Tune a clusterer over a parameter grid using its intrinsic metric.

    For each configuration the clusterer is fitted and its main metric is computed on
    the training data; the base model's selector (best, or elbow) picks the winning
    configuration. No cross-validation -- clustering is unsupervised.

    Porting:
        Mirrors ``clu_tune.R``. Fidelity: approximate.
        - Selects over the pydaltoolbox clusterers using the base model's intrinsic metric
          and selector; numeric results follow the underlying clusterer's backend/RNG.
    """

    def fit(self, data, **kwargs):
        grid = self._grid()
        base = self.base_model
        values = []
        goal = "maximize"
        for params in grid:
            model = self._build(params, data)
            labels = model.cluster(model.train_data)
            result = base.metric(data=model.train_data, cluster=labels, obj=model)
            values.append(result["value"])
            goal = result["goal"]
        index = base.selector(values, goal)
        return self._build(grid[index], data)
