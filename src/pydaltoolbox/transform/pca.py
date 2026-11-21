"""PCA dimensionality reduction (``trans_dt_pca.R`` analog).

Fits PCA on the numeric predictors (excluding ``attribute`` when given), standardizing
them first (center + scale, matching R's ``prcomp(scale.=TRUE)``). The number of
components is chosen by a minimum-curvature elbow on the cumulative explained variance
unless ``components`` is given. New data are projected with the learned parameters and
the target attribute is reattached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..adjust import adjust_dataframe
from ..base import DalTransform
from .curvature import FitCurvatureMin

__all__ = ["DtPca"]


class DtPca(DalTransform):
    """Principal Component Analysis as a transform.

    PCA finds an orthogonal set of directions (principal components) that successively
    capture the greatest possible variance of the (standardized) data. Projecting onto
    the leading components reduces dimensionality and decorrelates the features while
    retaining most of the information, which helps downstream models and visualization.

    References:
        Pearson, K. (1901). On lines and planes of closest fit to systems of points in
        space. Philosophical Magazine, 2(11):559-572.
        Hotelling, H. (1933). Analysis of a complex of statistical variables into
        principal components. Journal of Educational Psychology, 24:417-441.

    Parameters
    ----------
    attribute:
        Optional target column to exclude from PCA and reattach afterwards.
    components:
        Number of components (default: chosen automatically by the elbow rule).
    """

    def __init__(self, attribute: str | None = None, components: int | None = None) -> None:
        self.attribute = attribute
        self.components = components
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.feature_names: list | None = None

    def _features(self, data: pd.DataFrame) -> list:
        cols = [
            c
            for c in data.columns
            if c != self.attribute and pd.api.types.is_numeric_dtype(data[c])
        ]
        # drop constant (zero-variance) columns
        return [c for c in cols if data[c].min() != data[c].max()]

    def fit(self, data, **kwargs) -> DtPca:
        data = adjust_dataframe(data)
        self.feature_names = self._features(data)
        scaled = StandardScaler().fit(data[self.feature_names].to_numpy(dtype=float))
        self.scaler = scaled
        x = scaled.transform(data[self.feature_names].to_numpy(dtype=float))

        if self.components is None:
            full = PCA().fit(x)
            cumvar = np.cumsum(full.explained_variance_ratio_)
            self.components = int(FitCurvatureMin().transform(cumvar)["x"].iloc[0])

        self.components = max(1, min(self.components, len(self.feature_names)))
        self.pca = PCA(n_components=self.components).fit(x)
        return self

    def transform(self, data, **kwargs) -> pd.DataFrame:
        data = adjust_dataframe(data).copy()
        x = self.scaler.transform(data[self.feature_names].to_numpy(dtype=float))
        comps = self.pca.transform(x)
        cols = [f"PC{i + 1}" for i in range(self.components)]
        out = pd.DataFrame(comps, columns=cols, index=data.index)
        if self.attribute is not None and self.attribute in data.columns:
            out[self.attribute] = data[self.attribute].to_numpy()
        return out
