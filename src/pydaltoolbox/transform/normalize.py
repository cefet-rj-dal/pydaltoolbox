"""Normalization transforms.

Python counterparts of ``trans_norm_minmax.R`` and ``trans_norm_zscore.R``. Both
follow the full transform contract: ``fit`` learns per-column parameters, ``transform``
applies them, and ``inverse_transform`` reverses them. Only numeric columns are
touched; others pass through unchanged.
"""
from __future__ import annotations

import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["MinMax", "ZScore"]


def _numeric_columns(data: pd.DataFrame) -> list:
    return [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]


class MinMax(DalTransform):
    """Min-max normalization: scale numeric columns to [0, 1] per column.

    A linear rescaling that maps each column onto a common [0, 1] range, preserving the
    shape of its distribution. It puts features on equal footing for scale-sensitive
    methods (distance-based learners, neural networks) without distorting relative
    spacing.

    ``(x - min) / (max - min)``; constant columns map to 0.

    Porting:
        Mirrors ``trans_norm_minmax.R``. Fidelity: exact.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Normalization)
    """

    def __init__(self) -> None:
        self.norm_set: dict | None = None

    def fit(self, data, **kwargs) -> MinMax:
        data = adjust_dataframe(data)
        self.norm_set = {
            c: (float(data[c].min()), float(data[c].max())) for c in _numeric_columns(data)
        }
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for col, (mn, mx) in self.norm_set.items():
            if col in data.columns:
                data[col] = (data[col] - mn) / (mx - mn) if mx != mn else 0.0
        return data

    def inverse_transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for col, (mn, mx) in self.norm_set.items():
            if col in data.columns:
                data[col] = data[col] * (mx - mn) + mn if mx != mn else mx
        return data


class ZScore(DalTransform):
    """Z-score normalization: standardize numeric columns to mean 0 / sd 1.

    Centers each column on its mean and scales by its standard deviation, so the result
    is expressed in standard-deviation units. Unlike min-max it is not bounded and is
    more robust to differing spreads, which suits methods that assume roughly centered,
    unit-variance inputs.

    Optionally rescaled to a target mean (``nmean``) and sd (``nsd``):
    ``((x - mean) / sd) * nsd + nmean``. Constant columns map to ``nmean``.

    Porting:
        Mirrors ``trans_norm_zscore.R``. Fidelity: exact.
        - Uses the sample standard deviation (``ddof=1``), as R does.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann. (Standardization)
    """

    def __init__(self, nmean: float = 0.0, nsd: float = 1.0) -> None:
        self.nmean = nmean
        self.nsd = nsd
        self.norm_set: dict | None = None

    def fit(self, data, **kwargs) -> ZScore:
        data = adjust_dataframe(data)
        self.norm_set = {
            c: (float(data[c].mean()), float(data[c].std())) for c in _numeric_columns(data)
        }
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for col, (mean, sd) in self.norm_set.items():
            if col in data.columns:
                data[col] = (
                    (data[col] - mean) / sd * self.nsd + self.nmean if sd > 0 else self.nmean
                )
        return data

    def inverse_transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for col, (mean, sd) in self.norm_set.items():
            if col in data.columns:
                data[col] = (
                    (data[col] - self.nmean) / self.nsd * sd + mean if sd > 0 else self.nmean
                )
        return data
