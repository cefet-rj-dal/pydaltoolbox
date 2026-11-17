"""Reference datasets used in examples and tutorials.

Exposed as tidy :class:`pandas.DataFrame` objects: ``iris`` (classification) and
``diabetes`` (regression) come bundled with scikit-learn (offline). ``boston`` is
kept for compatibility but is fetched from OpenML on demand (needs network) and its
use is discouraged for ethical reasons -- prefer ``diabetes`` for regression.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml, load_diabetes, load_iris

__all__ = ["boston", "diabetes", "iris"]


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


def diabetes() -> pd.DataFrame:
    """Return the diabetes regression dataset with a numeric ``target`` column.

    Bundled with scikit-learn (no download). Features are already mean-centered and
    scaled, which is convenient for distance- and gradient-based learners.
    """
    data = load_diabetes(as_frame=True)
    return data.frame.copy()


def boston() -> pd.DataFrame:
    """Return the Boston Housing dataset with ``medv`` as the target column."""
    raw = fetch_openml(name="boston", version=1, as_frame=True)
    df = raw.frame.copy()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "medv" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "medv"})
    return df.apply(pd.to_numeric, errors="ignore")
