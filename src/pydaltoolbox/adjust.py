"""Data adjustment helpers.

Python counterparts of the R ``dal_adjust.R`` helpers. They coerce inputs into
the data structures the Experiment Line expects (``pandas``/``numpy``) so that
every component can assume a consistent input shape.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "adjust_class_label",
    "adjust_dataframe",
    "adjust_factor",
    "adjust_matrix",
]


def adjust_dataframe(data) -> pd.DataFrame:
    """Coerce ``data`` to a :class:`pandas.DataFrame` if needed.

    Mirrors R's ``adjust_data.frame``.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    return pd.DataFrame(data)


def adjust_matrix(data) -> np.ndarray:
    """Coerce ``data`` to a 2-D :class:`numpy.ndarray`.

    Mirrors R's ``adjust_matrix``.
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return np.asarray(data)


def adjust_factor(value, slevels: Sequence) -> pd.Categorical:
    """Convert ``value`` to an ordered categorical using ``slevels`` as levels.

    Mirrors R's ``adjust_factor``. Numeric codes (1-based, as in R) are mapped to
    the corresponding label in ``slevels``; existing labels are kept.
    """
    slevels = list(slevels)
    arr = np.asarray(value)

    if np.issubdtype(arr.dtype, np.number):
        # R uses 1-based factor codes; map i -> slevels[i - 1].
        labels = [slevels[int(i) - 1] for i in arr]
        return pd.Categorical(labels, categories=slevels)

    return pd.Categorical(arr, categories=slevels)


def adjust_class_label(x, val_true: float = 1, val_false: float = 0) -> pd.DataFrame:
    """One-hot encode a label vector into an indicator DataFrame.

    Mirrors R's ``adjust_class_label``: columns are the sorted distinct levels of
    ``x`` and each row has ``val_true`` in the column of its label.
    """
    cat = x if isinstance(x, pd.Categorical) else pd.Categorical(np.asarray(x))
    levels = list(cat.categories)
    codes = cat.codes
    res = np.full((len(codes), len(levels)), val_false, dtype=float)
    valid = codes >= 0
    res[np.arange(len(codes))[valid], codes[valid]] = val_true
    return pd.DataFrame(res, columns=levels)
