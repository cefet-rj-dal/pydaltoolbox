"""Supervised tuning via k-fold cross-validation (``cla_tune.R`` / ``reg_tune.R``)."""
from __future__ import annotations

import numpy as np

from ..sampling import SampleRandom, train_test_from_folds
from .base import DalTune

__all__ = ["ClaTune", "RegTune"]


class ClaTune(DalTune):
    """Tune a classifier by k-fold CV, maximizing a classification metric.

    Performs a grid search over the hyperparameter ``ranges``: each configuration is
    scored by k-fold cross-validation (the data is split into ``folds`` parts, each used
    once as the validation set), and the configuration with the best mean metric is
    refit on the full data. Cross-validation gives a less optimistic, lower-variance
    estimate of out-of-sample performance than a single train/test split.

    References:
        Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy
        Estimation and Model Selection. IJCAI.
    """

    def __init__(
        self,
        base_model,
        folds: int = 10,
        ranges: dict | None = None,
        metric: str = "accuracy",
        seed: int | None = None,
    ) -> None:
        super().__init__(base_model, folds, ranges, seed)
        self.metric = metric

    def fit(self, data, **kwargs):
        grid = self._grid()
        best = grid[0]
        if len(grid) > 1:
            attribute = self.base_model.attribute
            folds = SampleRandom(seed=self.seed).k_fold(data, self.folds)
            means = []
            for params in grid:
                vals = []
                for j in range(1, len(folds) + 1):
                    tt = train_test_from_folds(folds, j)
                    try:
                        model = self._build(params, tt["train"])
                        pred = model.predict(tt["test"])
                        m = model.evaluate(tt["test"][attribute], pred)["metrics"].iloc[0][self.metric]
                    except Exception:  # noqa: BLE001 - a bad configuration scores NaN
                        m = np.nan
                    vals.append(m)
                means.append(np.nanmean(vals) if np.any(~np.isnan(vals)) else -np.inf)
            best = grid[int(np.argmax(means))]
        return self._build(best, data)


class RegTune(DalTune):
    """Tune a regressor by k-fold CV, minimizing the mean squared error.

    Grid search over the hyperparameter ``ranges``, scoring each configuration by k-fold
    cross-validation and refitting the best on the full data. The selection criterion is
    the mean squared error averaged across folds, which estimates generalization error
    more reliably than a single split.

    References:
        Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy
        Estimation and Model Selection. IJCAI.
    """

    def fit(self, data, **kwargs):
        grid = self._grid()
        best = grid[0]
        if len(grid) > 1:
            attribute = self.base_model.attribute
            folds = SampleRandom(seed=self.seed).k_fold(data, self.folds)
            means = []
            for params in grid:
                vals = []
                for j in range(1, len(folds) + 1):
                    tt = train_test_from_folds(folds, j)
                    try:
                        model = self._build(params, tt["train"])
                        pred = model.predict(tt["test"])
                        m = model.evaluate(tt["test"][attribute], pred)["metrics"].iloc[0]["mse"]
                    except Exception:  # noqa: BLE001 - a bad configuration scores NaN
                        m = np.nan
                    vals.append(m)
                means.append(np.nanmean(vals) if np.any(~np.isnan(vals)) else np.inf)
            best = grid[int(np.argmin(means))]
        return self._build(best, data)
