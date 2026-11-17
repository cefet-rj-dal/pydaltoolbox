"""Hyperparameter tuning base (``dal_tune.R`` analog).

A tuner holds a base model, a fold count and a grid of hyperparameter ranges. Its
``fit`` returns the best **fitted model** (not the tuner), selected over the grid.
Subclasses define how a configuration is scored and selected.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

from ..base import DalBase

__all__ = ["DalTune"]


def _as_list(value) -> list:
    if isinstance(value, (list, tuple, range)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


class DalTune(DalBase):
    """Base class for hyperparameter optimization.

    Parameters
    ----------
    base_model:
        The model to tune (its hyperparameters are set via ``set_params``).
    folds:
        Number of cross-validation folds (used by the supervised tuners).
    ranges:
        Dict mapping hyperparameter name to the list of values to try.
    seed:
        Optional seed for reproducible fold generation.
    """

    def __init__(self, base_model, folds: int = 10, ranges: dict | None = None, seed: int | None = None) -> None:
        self.base_model = base_model
        self.folds = folds
        self.ranges = ranges or {}
        self.seed = seed

    def _grid(self) -> list:
        if not self.ranges:
            return [{}]
        keys = list(self.ranges.keys())
        values = [_as_list(self.ranges[k]) for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _build(self, params: dict, data):
        """Clone the base model, apply params and fit it on ``data``."""
        model = copy.deepcopy(self.base_model)
        model.set_params(params)
        model.fit(data)
        return model
