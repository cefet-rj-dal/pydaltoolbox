"""Unit tests for cleaning transforms (NA removal, imputation, outliers)."""
import numpy as np
import pandas as pd

from pydaltoolbox import (
    ImputationSimple,
    NaRemoval,
    OutliersBoxplot,
    OutliersGaussian,
    datasets,
)


def _iris_with_na():
    iris = datasets.iris().copy()
    iris.loc[[1, 10, 25], "sepal_length"] = np.nan
    return iris


def test_na_removal_drops_rows_with_missing():
    iris = _iris_with_na()
    out = NaRemoval().transform(iris)
    assert len(out) == len(iris) - 3
    assert out["sepal_length"].notna().all()


def test_imputation_median_fills_numeric():
    iris = _iris_with_na()
    clean = datasets.iris()["sepal_length"].median()
    imp = ImputationSimple(method="median").fit(iris)
    out = imp.transform(iris)
    assert out["sepal_length"].notna().all()
    assert out.loc[1, "sepal_length"] == imp.values["sepal_length"]
    # imputed value equals median of observed values
    assert imp.values["sepal_length"] == pd.Series(iris["sepal_length"]).median()
    assert abs(imp.values["sepal_length"] - clean) < 0.2


def test_imputation_mode_fills_categorical():
    iris = datasets.iris().copy()
    iris["Species"] = iris["Species"].astype(object)
    iris.loc[[3, 15], "Species"] = np.nan
    out = ImputationSimple().fit(iris).transform(iris)
    assert out["Species"].notna().all()


def test_outliers_boxplot_removes_extreme_rows():
    iris = datasets.iris().copy()
    iris.loc[0, "sepal_width"] = 100.0  # gross outlier
    model = OutliersBoxplot().fit(iris)
    out = model.transform(iris)
    assert len(out) < len(iris)
    assert out["sepal_width"].max() < 100.0


def test_outliers_gaussian_removes_extreme_rows():
    iris = datasets.iris().copy()
    iris.loc[0, "sepal_width"] = 100.0
    out = OutliersGaussian(alpha=3.0).fit(iris).transform(iris)
    assert out["sepal_width"].max() < 100.0


def test_outliers_fit_is_chainable():
    iris = datasets.iris()
    model = OutliersBoxplot()
    assert model.fit(iris) is model
