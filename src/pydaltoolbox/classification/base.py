"""Classification base class.

Python counterpart of ``cla_classification.R``. Holds the target attribute and
class levels, and provides the standard classification ``evaluate`` producing
accuracy plus a binary (one-vs-rest) confusion summary for a reference class.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe, adjust_factor
from ..base import Predictor

__all__ = ["Classification"]


def _as_labels(prediction, slevels: Sequence) -> np.ndarray:
    """Normalize a prediction (scores matrix or labels) into a label array."""
    slevels = list(slevels)
    if isinstance(prediction, pd.DataFrame):
        cols = [c for c in slevels if c in prediction.columns]
        base = cols if cols else list(prediction.columns)
        mat = prediction[base].to_numpy()
        return np.array([base[i] for i in mat.argmax(axis=1)], dtype=object)

    arr = prediction.to_numpy() if isinstance(prediction, pd.Series) else np.asarray(prediction)
    if arr.ndim == 2:
        return np.array([slevels[i] for i in arr.argmax(axis=1)], dtype=object)
    return arr.astype(object)


class Classification(Predictor):
    """Ancestor class for classification models.

    Parameters
    ----------
    attribute:
        Name of the target column.
    slevels:
        Possible class labels. If ``None`` they are inferred at fit time.
    """

    def __init__(self, attribute: str, slevels: Sequence | None = None) -> None:
        super().__init__()
        self.attribute = attribute
        self.slevels = list(slevels) if slevels is not None else None

    def _prepare_classification_data(self, data):
        data = adjust_dataframe(data)
        if self.attribute not in data.columns:
            raise ValueError(f"{type(self).__name__}: attribute '{self.attribute}' not found in data.")
        if self.slevels is None:
            y = data[self.attribute]
            if isinstance(y.dtype, pd.CategoricalDtype):
                self.slevels = list(y.cat.categories)
            else:
                self.slevels = list(pd.unique(y.astype(object)))
        data = data.copy()
        data[self.attribute] = adjust_factor(data[self.attribute], self.slevels)
        self._prepare_fit(data)
        return data

    def evaluate(self, data, prediction, ref: int = 0, **kwargs) -> dict:
        """Compute classification metrics.

        ``data`` holds the true labels (or a score matrix), ``prediction`` holds
        predicted scores (or labels). ``ref`` is the 0-based index of the positive
        class for the one-vs-rest confusion metrics.
        """
        slevels = self.slevels
        true_f = _as_labels(data, slevels)
        pred_f = _as_labels(prediction, slevels)

        metrics = {}
        metrics["accuracy"] = float(np.mean(true_f == pred_f))

        pos = slevels[ref]
        t = true_f == pos
        p = pred_f == pos
        tp = int(np.sum(t & p))
        tn = int(np.sum(~t & ~p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        metrics.update(TP=tp, TN=tn, FP=fp, FN=fn)

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else float("nan")
        )
        metrics.update(
            precision=precision,
            recall=recall,
            sensitivity=recall,
            specificity=specificity,
            f1=f1,
        )

        return {
            "data": true_f,
            "prediction": pred_f,
            "metrics": pd.DataFrame([metrics]),
        }
