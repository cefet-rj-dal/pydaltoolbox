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

__all__ = ["Classification", "SklearnClassification"]


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

    Porting:
        Mirrors ``cla_classification.R``. Fidelity: adapted.
        - Idiomatic-Python API (see :class:`~pydaltoolbox.base.DalBase`); provides the
          standard classification ``evaluate`` (accuracy and related metrics).
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


class SklearnClassification(Classification):
    """Base for classifiers that wrap a scikit-learn estimator.

    This captures the whole integration contract once: ``fit`` prepares the data,
    builds the estimator and trains it; ``predict`` returns a class-score table.
    Concrete learners only implement :meth:`_make_model` (and store their
    hyperparameters as attributes, so ``set_params`` can tune them). This is the
    Experiment Line extensibility promise in practice.

    Porting:
        No direct R analog. Fidelity: adapted.
        - Python-only helper that factors out the shared scikit-learn integration for the
          concrete ``Cla*`` learners (R repeats this wiring per ``cla_*.R`` file).
    """

    def _make_model(self):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def fit(self, data, **kwargs) -> SklearnClassification:
        data = self._prepare_classification_data(data)
        self.model = self._make_model()
        self.model.fit(data[self.x], data[self.attribute].astype(object))
        return self

    def predict(self, x, **kwargs) -> pd.DataFrame:
        """Return a class-score table (one column per class level).

        Only the feature columns (``self.x``) are used, so ``x`` may be the feature
        frame or the full dataset including the target; the target is ignored.
        """
        x = adjust_dataframe(x)
        proba = self.model.predict_proba(x[self.x])
        result = pd.DataFrame(proba, columns=[str(c) for c in self.model.classes_])
        return result.reindex(columns=[str(s) for s in self.slevels], fill_value=0.0)
