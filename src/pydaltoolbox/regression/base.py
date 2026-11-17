"""Regression base classes.

Python counterpart of ``reg_regression.R``. ``Regression`` stores the target
attribute and provides the standard regression ``evaluate`` (MSE, sMAPE, R2).
``SklearnRegression`` captures the fit/predict contract once for scikit-learn
backed regressors, mirroring ``SklearnClassification``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe
from ..base import Predictor

__all__ = ["Regression", "SklearnRegression"]


class Regression(Predictor):
    """Ancestor class for regression models.

    Parameters
    ----------
    attribute:
        Name of the numeric target column.
    """

    def __init__(self, attribute: str) -> None:
        super().__init__()
        self.attribute = attribute

    def evaluate(self, values, prediction, **kwargs) -> dict:
        """Compute regression metrics (MSE, sMAPE, R2)."""
        values = np.asarray(values, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        if len(values) != len(prediction):
            raise ValueError("values and prediction have different lengths")

        mse = float(np.mean((values - prediction) ** 2))
        smape = float(
            np.mean(np.abs(values - prediction) / ((np.abs(values) + np.abs(prediction)) / 2))
        )
        ss_res = np.sum((prediction - values) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = float(1 - ss_res / ss_tot)

        return {
            "values": values,
            "prediction": prediction,
            "metrics": pd.DataFrame([{"mse": mse, "smape": smape, "R2": r2}]),
        }


class SklearnRegression(Regression):
    """Base for regressors that wrap a scikit-learn estimator.

    Subclasses implement :meth:`_make_model` and store tunable hyperparameters as
    attributes; fit/predict are shared.
    """

    def _make_model(self):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def fit(self, data, **kwargs) -> SklearnRegression:
        data = self._prepare_fit(data)  # records self.x (features)
        self.model = self._make_model()
        self.model.fit(data[self.x], data[self.attribute])
        return self

    def predict(self, x, **kwargs) -> np.ndarray:
        """Return numeric predictions.

        Only the feature columns (``self.x``) are used, so ``x`` may be the feature
        frame or the full dataset including the target; the target is ignored.
        """
        x = adjust_dataframe(x)
        return np.asarray(self.model.predict(x[self.x]))
