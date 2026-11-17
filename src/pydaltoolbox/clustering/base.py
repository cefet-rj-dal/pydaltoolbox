"""Clusterer base class (``clu_clusterer.R`` analog).

Clusterers follow the Experiment Line contract: ``fit`` learns and stores the model,
``cluster`` (reached via ``action``) produces labels, and ``evaluate`` scores the
partition with internal metrics (and external ones when a reference label is given).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalLearner
from .metrics import metric_entropy, metric_wcss, selector_best

__all__ = ["Clusterer"]


class Clusterer(DalLearner):
    """Base class for clustering algorithms and their evaluation."""

    def __init__(self) -> None:
        self.model = None
        self.train_data: pd.DataFrame | None = None
        self.xnames: list | None = None
        self.fitted = False
        # main metric + selector are used by tuning
        self.metric = metric_wcss
        self.metric_name = "wcss"
        self.selector = selector_best
        self.eval_internal = [metric_wcss]
        self.eval_external = [metric_entropy]

    def _prepare_fit(self, data) -> pd.DataFrame:
        data = adjust_dataframe(data)
        self.train_data = data
        self.xnames = list(data.columns)
        self.fitted = True
        return data

    def _features(self, data) -> np.ndarray:
        data = adjust_dataframe(data)
        return data[self.xnames].to_numpy(dtype=float)

    def _require_fitted(self) -> None:
        if not self.fitted or self.model is None:
            raise ValueError(f"{type(self).__name__} must be fitted before clustering.")

    def cluster(self, data, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError

    def action(self, data, **kwargs):
        """For clusterers, ``action`` dispatches to :meth:`cluster`."""
        return self.cluster(data, **kwargs)

    def evaluate(self, cluster, attribute=None, **kwargs) -> dict:
        rows = [fn(data=self.train_data, cluster=cluster, obj=self) for fn in self.eval_internal]
        if attribute is not None:
            rows += [
                fn(cluster=cluster, attribute=attribute, obj=self) for fn in self.eval_external
            ]
        return {"metrics": pd.DataFrame(rows, columns=["metric", "value", "goal", "type"])}
