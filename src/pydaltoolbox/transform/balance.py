"""Class-balancing transforms.

Python counterparts of ``trans_sample_balance.R``, ``trans_sample_oversampling.R``
and ``trans_sample_subsampling.R``. They rebalance the class distribution of a
categorical target by up/down sampling or by generating synthetic minority examples
(a local SMOTE, matching the R implementation -- no extra dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["BalOversampling", "BalSubsampling", "SampleBalance"]


def _present_counts(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    return counts[counts > 0]


def _restore_categoricals(result: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    cat_cols = {
        c: original[c].dtype
        for c in original.columns
        if isinstance(original[c].dtype, pd.CategoricalDtype)
    }
    return result.astype(cat_cols) if cat_cols else result


class SampleBalance(DalTransform):
    """Balance classes by up- or down-sampling. Mirrors R ``sample_balance()``.

    ``method="down"`` reduces every class to the minority count; ``method="up"``
    grows every class to the majority count by sampling with replacement.
    """

    def __init__(self, attribute: str, method: str = "down", seed: int | None = None) -> None:
        if method not in ("down", "up"):
            raise ValueError("sample_balance: method must be 'down' or 'up'.")
        self.attribute = attribute
        self.method = method
        self.seed = seed

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        rng = np.random.default_rng(self.seed)
        counts = _present_counts(data[self.attribute])
        target = int(counts.min() if self.method == "down" else counts.max())
        parts = []
        for _, grp in data.groupby(self.attribute, observed=True):
            replace = len(grp) < target
            idx = rng.choice(len(grp), size=target, replace=replace)
            parts.append(grp.iloc[idx])
        result = pd.concat(parts).reset_index(drop=True)
        return _restore_categoricals(result, data)


class BalSubsampling(DalTransform):
    """Undersample every class to the minority count. Mirrors R ``bal_subsampling()``."""

    def __init__(self, attribute: str, seed: int | None = None) -> None:
        self.attribute = attribute
        self.seed = seed

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        rng = np.random.default_rng(self.seed)
        target = int(_present_counts(data[self.attribute]).min())
        parts = []
        for _, grp in data.groupby(self.attribute, observed=True):
            idx = rng.choice(len(grp), size=target, replace=False)
            parts.append(grp.iloc[idx])
        result = pd.concat(parts).reset_index(drop=True)
        return _restore_categoricals(result, data)


class BalOversampling(DalTransform):
    """Oversample every class to the majority count. Mirrors R ``bal_oversampling()``.

    ``method="random"`` replicates existing minority examples; ``method="smote"``
    generates synthetic ones by interpolating between a sample and one of its ``k``
    nearest neighbors.
    """

    def __init__(
        self,
        attribute: str,
        method: str = "smote",
        k: int = 5,
        seed: int | None = None,
    ) -> None:
        if method not in ("smote", "random"):
            raise ValueError("bal_oversampling: method must be 'smote' or 'random'.")
        self.attribute = attribute
        self.method = method
        self.k = int(k)
        self.seed = seed

    def _numeric_matrix(self, grp: pd.DataFrame, features: list) -> np.ndarray:
        cols = []
        for f in features:
            col = grp[f]
            if pd.api.types.is_numeric_dtype(col):
                cols.append(col.to_numpy(dtype=float))
            else:
                cols.append(pd.factorize(col)[0].astype(float))
        return np.column_stack(cols)

    def _random_oversample(self, grp: pd.DataFrame, need: int, rng) -> pd.DataFrame:
        idx = rng.choice(len(grp), size=need, replace=True)
        return grp.iloc[idx]

    def _smote(self, grp: pd.DataFrame, need: int, rng) -> pd.DataFrame:
        features = [c for c in grp.columns if c != self.attribute]
        n = len(grp)
        if not features or n < 2:
            return self._random_oversample(grp, need, rng)
        x = self._numeric_matrix(grp, features)
        k = max(1, min(self.k, n - 1))
        rows = []
        for _ in range(need):
            b = int(rng.integers(n))
            dist = ((x - x[b]) ** 2).sum(axis=1)
            neighbors = [o for o in np.argsort(dist) if o != b][:k]
            nn = int(neighbors[int(rng.integers(len(neighbors)))])
            gap = float(rng.random())
            row = grp.iloc[b].copy()
            for f in features:
                bv, nv = grp.iloc[b][f], grp.iloc[nn][f]
                if pd.api.types.is_numeric_dtype(grp[f]):
                    row[f] = bv + gap * (nv - bv)
                else:
                    row[f] = bv if rng.random() < 0.5 else nv
            rows.append(row)
        return pd.DataFrame(rows)

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        rng = np.random.default_rng(self.seed)
        counts = _present_counts(data[self.attribute])
        target = int(counts.max())
        parts = []
        for _, grp in data.groupby(self.attribute, observed=True):
            need = target - len(grp)
            if need > 0:
                synth = (
                    self._random_oversample(grp, need, rng)
                    if self.method == "random"
                    else self._smote(grp, need, rng)
                )
                parts.append(pd.concat([grp, synth]))
            else:
                parts.append(grp)
        result = pd.concat(parts).reset_index(drop=True)
        return _restore_categoricals(result, data)
