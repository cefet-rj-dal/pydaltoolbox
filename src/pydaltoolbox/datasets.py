"""Reference datasets used in examples and tutorials.

The R package ships ``iris`` (via datasets) and ``Boston``. Here we expose them as
tidy :class:`pandas.DataFrame` objects so the Python tutorials mirror the R ones.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris

__all__ = ["boston", "iris"]


def iris() -> pd.DataFrame:
    """Return the iris dataset with a categorical ``Species`` target column."""
    data = load_iris(as_frame=True)
    df = data.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    species = pd.Categorical.from_codes(data.target, data.target_names)
    df = df.drop(columns="target")
    df["Species"] = species
    return df


def boston() -> pd.DataFrame:
    """Return the Boston Housing dataset with ``medv`` as the target column."""
    raw = fetch_openml(name="boston", version=1, as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "medv" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "medv"})
    return df.apply(pd.to_numeric, errors="ignore")
