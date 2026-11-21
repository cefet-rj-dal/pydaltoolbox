"""Missing-value transforms.

Python counterparts of ``trans_na_removal.R`` and ``trans_imputation_simple.R``.
"""
from __future__ import annotations

import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["ImputationSimple", "NaRemoval"]


class NaRemoval(DalTransform):
    """Remove rows that contain any missing value.

    Complete-case analysis (listwise deletion): any row with at least one missing value
    is dropped. It is simple and unbiased when data are missing completely at random, but
    discards information and can bias results otherwise, so it is best when missingness is
    rare.

    Porting:
        Mirrors ``trans_na_removal.R``. Fidelity: exact.

    References:
        Little, R. J. A., and Rubin, D. B. (2019). Statistical Analysis with Missing Data
        (3rd ed.). Wiley.
    """

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        return data.dropna(axis=0, how="any").reset_index(drop=True)


class ImputationSimple(DalTransform):
    """Impute missing values with simple per-column statistics.

    Univariate imputation: each missing entry is replaced by a summary of its own column
    (median/mean for numeric, mode for categorical), learned on ``fit`` and reused on
    ``transform`` so train and test are filled consistently. Fast and robust, though it
    ignores relationships between columns (see :class:`ImputationTree` for a predictive
    alternative).

    Numeric columns are filled with the median (default) or mean; non-numeric columns
    with the mode (most frequent value).

    Porting:
        Mirrors ``trans_imputation_simple.R``. Fidelity: exact.

    References:
        Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and Techniques
        (3rd ed.). Morgan Kaufmann.
        Little, R. J. A., and Rubin, D. B. (2019). Statistical Analysis with Missing Data
        (3rd ed.). Wiley.

    Parameters
    ----------
    method:
        ``"median"`` (default) or ``"mean"`` for numeric columns.
    cols:
        Optional list of columns to impute (default: all columns).
    """

    def __init__(self, method: str = "median", cols: list | None = None) -> None:
        if method not in ("median", "mean"):
            raise ValueError("imputation_simple: method must be 'median' or 'mean'.")
        self.method = method
        self.cols = cols
        self.values: dict | None = None

    def fit(self, data, **kwargs) -> ImputationSimple:
        data = adjust_dataframe(data)
        cols = self.cols if self.cols is not None else list(data.columns)
        values = {}
        for col in cols:
            series = data[col]
            if pd.api.types.is_numeric_dtype(series):
                if series.notna().sum() == 0:
                    raise ValueError(f"imputation_simple: column '{col}' is all missing.")
                values[col] = series.median() if self.method == "median" else series.mean()
            else:
                mode = series.dropna().mode()
                if len(mode) == 0:
                    raise ValueError(f"imputation_simple: column '{col}' is all missing.")
                values[col] = mode.iloc[0]
        self.values = values
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        for col, value in self.values.items():
            if col in data.columns:
                data[col] = data[col].fillna(value)
        return data
