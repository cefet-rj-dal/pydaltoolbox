"""Feature-engineering transforms.

Python counterparts of ``trans_feature_generation.R``, ``trans_aggregation.R`` and
``trans_hierarchy_cut.R``. R uses non-standard evaluation (named expressions); the
idiomatic Python equivalent is **callables** of the DataFrame (or ``df.eval`` strings).
"""
from __future__ import annotations

import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["Aggregation", "FeatureGeneration", "HierarchyCut"]


class FeatureGeneration(DalTransform):
    """Create new columns from existing ones.

    Each keyword maps a new column name to either a callable ``df -> Series`` or a
    pandas ``eval`` expression string. Mirrors ``feature_generation``.

    Example::

        FeatureGeneration(
            sepal_area=lambda df: df["sepal_length"] * df["sepal_width"],
            ratio="sepal_length / sepal_width",
        )
    """

    def __init__(self, **features) -> None:
        self.features = features

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for name, expr in self.features.items():
            if callable(expr):
                data[name] = expr(data)
            elif isinstance(expr, str):
                data[name] = data.eval(expr)
            else:
                raise TypeError(f"feature_generation: '{name}' must be a callable or a string.")
        return data


class Aggregation(DalTransform):
    """Aggregate rows by a grouping column into one row per group.

    Each keyword maps an output column to either a callable ``group_df -> scalar`` or a
    ``(column, func)`` tuple (e.g. ``("sepal_length", "mean")``). Mirrors ``aggregation``.

    Example::

        Aggregation("Species", mean_sepal=("sepal_length", "mean"), n=("sepal_length", "size"))
    """

    def __init__(self, group: str, **aggs) -> None:
        self.group = group
        self.aggs = aggs

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        if self.group not in data.columns:
            raise ValueError(f"aggregation: '{self.group}' must be a valid column name.")
        if not self.aggs:
            raise ValueError("aggregation: no aggregation expressions provided.")
        rows = []
        for key, group in data.groupby(self.group, observed=True):
            row = {self.group: key}
            for name, spec in self.aggs.items():
                if callable(spec):
                    row[name] = spec(group)
                elif isinstance(spec, tuple) and len(spec) == 2:
                    col, func = spec
                    row[name] = group[col].agg(func)
                else:
                    raise TypeError(f"aggregation: '{name}' must be a callable or (column, func).")
            rows.append(row)
        return pd.DataFrame(rows)


class HierarchyCut(DalTransform):
    """Discretize a numeric attribute into an ordered categorical level (concept hierarchy).

    Adds a new column (default ``<attribute>_Level``) with the binned labels. Mirrors
    ``hierarchy_cut``.
    """

    def __init__(
        self,
        attribute: str,
        breaks,
        labels=None,
        new_attribute: str | None = None,
    ) -> None:
        self.attribute = attribute
        self.breaks = breaks
        self.labels = labels
        self.new_attribute = new_attribute

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        if self.attribute not in data.columns:
            raise ValueError(f"hierarchy_cut: '{self.attribute}' must be a valid column name.")
        new = self.new_attribute or f"{self.attribute}_Level"
        data[new] = pd.cut(
            data[self.attribute],
            bins=self.breaks,
            labels=self.labels,
            include_lowest=True,
        )
        return data
