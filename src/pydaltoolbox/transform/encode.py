"""Categorical encoding (``trans_categ_mapping.R`` analog).

Note on behavior: the R ``categ_mapping`` returns *only* the one-hot columns of the
target attribute, dropping every other column. That is error-prone in a pipeline, so
here we keep the other columns and replace the attribute with its one-hot dummies.
``fit`` records the categories so train/test produce the same columns.
"""
from __future__ import annotations

import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["CategMapping"]


class CategMapping(DalTransform):
    """One-hot encode a categorical column into indicator columns.

    One-hot (dummy) encoding turns a categorical attribute with ``m`` levels into ``m``
    binary indicator columns, so models that require numeric input can use it without
    imposing a false ordinal relationship between the categories. Levels are learned on
    ``fit`` so train and test produce the same columns.

    Porting:
        Mirrors ``trans_categ_mapping.R``. Fidelity: adapted.
        - Behavior change: R ``categ_mapping`` returns *only* the target's one-hot columns
          and drops every other column (error-prone in a pipeline); here the other columns
          are kept and the attribute is replaced by its one-hot dummies.

    References:
        Kuhn, M., and Johnson, K. (2013). Applied Predictive Modeling. Springer.
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann.

    Parameters
    ----------
    attribute:
        Name of the categorical column to encode.
    """

    def __init__(self, attribute: str) -> None:
        self.attribute = attribute
        self.levels: list | None = None

    @staticmethod
    def _levels_of(series: pd.Series) -> list:
        if isinstance(series.dtype, pd.CategoricalDtype):
            return list(series.cat.categories)
        return list(pd.unique(series.dropna().astype(object)))

    def fit(self, data, **kwargs) -> CategMapping:
        data = adjust_dataframe(data)
        self.levels = self._levels_of(data[self.attribute])
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        levels = self.levels if self.levels is not None else self._levels_of(data[self.attribute])
        cat = pd.Categorical(data[self.attribute].astype(object), categories=levels)
        dummies = pd.get_dummies(cat, prefix=self.attribute, dtype=int)
        dummies.index = data.index
        return pd.concat([data.drop(columns=[self.attribute]), dummies], axis=1)
