"""Outlier-removal transforms.

Python counterparts of ``trans_outliers_boxplot.R`` (Tukey IQR rule) and
``trans_outliers_gaussian.R`` (3-sigma rule). Both learn per-numeric-column
thresholds on ``fit`` and drop rows with any out-of-range value on ``transform``.
"""
from __future__ import annotations

import pandas as pd

from ..adjust import adjust_dataframe
from ..base import DalTransform

__all__ = ["OutliersBoxplot", "OutliersGaussian"]


class _Outliers(DalTransform):
    """Shared row-removal logic given per-column lower/higher thresholds."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.lower: dict | None = None
        self.higher: dict | None = None

    def _bounds(self, series: pd.Series) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    def fit(self, data, **kwargs) -> _Outliers:
        data = adjust_dataframe(data)
        self.lower, self.higher = {}, {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                lo, hi = self._bounds(data[col])
                self.lower[col], self.higher[col] = lo, hi
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data)
        mask = pd.Series(False, index=data.index)
        for col in self.lower:
            if col in data.columns:
                series = data[col]
                mask = mask | (series.notna() & ((series < self.lower[col]) | (series > self.higher[col])))
        return data[~mask].reset_index(drop=True)


class OutliersBoxplot(_Outliers):
    """Remove outliers using Tukey's IQR rule (Q1 - alpha*IQR, Q3 + alpha*IQR).

    Tukey's boxplot rule flags values that fall beyond the quartiles by a multiple of the
    interquartile range (IQR). Being based on quantiles rather than the mean/sd, it is
    robust to the very outliers it detects and makes no distributional assumption.

    ``alpha=1.5`` are the standard whiskers; ``alpha=3`` targets extreme values.
    Mirrors R ``outliers_boxplot()``.

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.
    """

    def __init__(self, alpha: float = 1.5) -> None:
        super().__init__(alpha)

    def _bounds(self, series: pd.Series) -> tuple[float, float]:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - self.alpha * iqr, q3 + self.alpha * iqr


class OutliersGaussian(_Outliers):
    """Remove outliers using the Gaussian rule (mean +/- alpha*sd).

    Flags values lying more than ``alpha`` standard deviations from the mean. Under an
    approximately normal distribution the three-sigma band (``alpha=3``) covers about
    99.7% of the data, so points outside it are treated as outliers. Best suited to
    roughly symmetric, bell-shaped columns.

    ``alpha=3`` is the three-sigma rule. Mirrors R ``outliers_gaussian()``
    (sample standard deviation, ddof=1).

    References:
        Pukelsheim, F. (1994). The Three Sigma Rule. The American Statistician,
        48(2):88-91.
    """

    def __init__(self, alpha: float = 3.0) -> None:
        super().__init__(alpha)

    def _bounds(self, series: pd.Series) -> tuple[float, float]:
        mean = series.mean()
        sd = series.std()
        return mean - self.alpha * sd, mean + self.alpha * sd
