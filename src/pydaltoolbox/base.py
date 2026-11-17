"""Core Experiment Line base classes.

Python counterpart of the R ``dal_base.R``, ``dal_transform.R``,
``dal_learner.R`` and ``dal_predictor.R`` files.

The Experiment Line is a *rigid but extensible* contract. Every component is an
object that understands the same small protocol:

* :meth:`DalBase.fit` -- learn / adjust internal state from data (returns ``self``
  so calls can be chained).
* :meth:`DalBase.action` -- apply the component to data. It is **polymorphic**:
  transforms dispatch to :meth:`transform`, predictors to :meth:`predict`,
  clusterers to ``cluster`` and pattern miners to ``discover``. This uniform entry
  point is what lets heterogeneous steps be composed in a pipeline.
* :meth:`DalBase.set_params` -- inject (hyper)parameters, used by tuning.
* :meth:`DalLearner.evaluate` -- score the component's output.

Creating a new method means subclassing the right base and implementing two or
three methods; everything else (sampling, evaluation, tuning) is inherited.
"""
from __future__ import annotations

from collections.abc import Mapping

from .adjust import adjust_dataframe

__all__ = ["DalBase", "DalLearner", "DalTransform", "Predictor"]


class DalBase:
    """Minimal abstract base class for all DAL objects."""

    def fit(self, data=None, **kwargs) -> DalBase:
        """Train/adjust the object. Default is a no-op returning ``self``."""
        return self

    def action(self, data=None, **kwargs):
        """Apply the object to data. Default forwards ``data`` unchanged."""
        return data

    def set_params(self, params: Mapping | None) -> DalBase:
        """Assign matching parameters to existing attributes (best effort).

        Only keys that correspond to an existing attribute are assigned, mirroring
        the R ``set_params.dal_base`` behaviour used by the tuning machinery.
        """
        if params:
            for key, value in dict(params).items():
                if hasattr(self, key):
                    setattr(self, key, value)
        return self


class DalTransform(DalBase):
    """Base class for data transformations with optional inverse support."""

    def transform(self, data, **kwargs):
        """Apply the transformation. Default is identity."""
        return data

    def inverse_transform(self, data, **kwargs):
        """Revert the transformation. Default is identity."""
        return data

    def action(self, data, **kwargs):
        """For transforms, ``action`` dispatches to :meth:`transform`."""
        return self.transform(data, **kwargs)


class DalLearner(DalBase):
    """Base ancestor for learning tasks (classification, regression, ...)."""

    def action(self, data, **kwargs):
        """For learners, ``action`` dispatches to :meth:`predict`."""
        return self.predict(data, **kwargs)

    def predict(self, data, **kwargs):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """Evaluate learner performance. Default returns ``None``."""
        return


class Predictor(DalLearner):
    """Ancestor for supervised predictors (classification and regression).

    Provides a default :meth:`fit` that records the feature columns (everything
    except the target ``attribute``).
    """

    def __init__(self) -> None:
        self.attribute: str | None = None
        self.x: list | None = None

    def _prepare_fit(self, data):
        data = adjust_dataframe(data)
        self.x = [c for c in data.columns if c != self.attribute]
        return data

    def fit(self, data, **kwargs) -> Predictor:
        self._prepare_fit(data)
        return self
