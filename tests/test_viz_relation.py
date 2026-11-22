"""Tests for the step-2.4 viz plot_* family (relation, temporal, structure).

matplotlib is required; the module is skipped without it. The Agg backend is
forced so no window opens. Each function returns a matplotlib Axes (plot_pair
returns the scatter-matrix array of Axes).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

import pydaltoolbox as dal


@pytest.fixture
def iris():
    rng = np.random.default_rng(7)
    n = 30
    return pd.DataFrame({
        "sepal_length": rng.normal(5.8, 0.8, n),
        "sepal_width": rng.normal(3.0, 0.4, n),
        "petal_length": rng.normal(3.7, 1.7, n),
        "petal_width": rng.normal(1.2, 0.7, n),
        "Species": np.repeat(["a", "b", "c"], n // 3),
    })


def test_plot_scatter_categorical(iris):
    long = pd.DataFrame({"x": iris["sepal_length"], "value": iris["sepal_width"],
                         "variable": iris["Species"]})
    ax = dal.plot_scatter(long, colors=["red", "green", "blue"])
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 3  # one scatter per class
    assert ax.get_legend() is not None


def test_plot_scatter_numeric_group(iris):
    long = pd.DataFrame({"x": iris["sepal_length"], "value": iris["sepal_width"],
                         "variable": iris["petal_length"]})
    ax = dal.plot_scatter(long)
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 1  # single mappable scatter + colorbar


def test_plot_points(iris):
    x = np.arange(0, 10, 0.5)
    frame = pd.DataFrame({"x": x, "sin": np.sin(x), "cos": np.cos(x)})
    ax = dal.plot_points(frame, colors=["red", "green"])
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 2


def test_plot_correlation(iris):
    ax = dal.plot_correlation(iris.drop(columns="Species"))
    assert isinstance(ax, Axes)
    assert ax.images  # imshow present
    # 4x4 matrix -> 16 annotations with full triangle + diagonal
    assert len(ax.texts) == 16


def test_plot_correlation_upper_hclust(iris):
    ax = dal.plot_correlation(iris.drop(columns="Species"), triangle="upper",
                              reorder="hclust", show_diag=False)
    assert isinstance(ax, Axes)
    # strict upper triangle of 4x4 = 6 cells annotated
    assert len(ax.texts) == 6


def test_plot_correlation_needs_two_numeric():
    with pytest.raises(ValueError):
        dal.plot_correlation(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


def test_plot_pair_returns_axes_grid(iris):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    arr = dal.plot_pair(iris, cnames=cols, clabel="Species")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)
    assert isinstance(arr.flat[0], Axes)


def test_plot_pair_bad_columns(iris):
    with pytest.raises(ValueError):
        dal.plot_pair(iris, cnames=["nope"])


def test_plot_parallel(iris):
    ax = dal.plot_parallel(iris, columns=[0, 1, 2, 3], group="Species")
    assert isinstance(ax, Axes)
    assert ax.get_legend() is not None


def test_plot_pixel(iris):
    ax = dal.plot_pixel(iris.drop(columns="Species"), title="iris")
    assert isinstance(ax, Axes)
    assert ax.images
    assert [t.get_text() for t in ax.get_yticklabels()][:1] == ["sepal_length"]


def test_plot_series():
    x = np.arange(0, 10, 0.25)
    frame = pd.DataFrame({"x": x, "sin": np.sin(x)})
    ax = dal.plot_series(frame, colors=["red"])
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1


def test_plot_ts():
    x = np.arange(0, 10, 0.25)
    ax = dal.plot_ts(x=x, y=np.sin(x), color="red")
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1


def test_plot_ts_default_index():
    ax = dal.plot_ts(y=[1.0, 2.0, 3.0, 2.0])
    assert isinstance(ax, Axes)
    assert list(ax.lines[0].get_xdata()) == [1, 2, 3, 4]


def test_plot_ts_pred():
    x = np.arange(0, 10, 0.25)
    y = np.sin(x)
    ax = dal.plot_ts_pred(y=y, yadj=np.sin(x[:35]), ypred=np.sin(x[35:]))
    assert isinstance(ax, Axes)
    # observed + adjusted + prediction
    assert len(ax.lines) == 3


def test_plot_ts_pred_no_prediction():
    x = np.arange(0, 5, 0.25)
    y = np.sin(x)
    ax = dal.plot_ts_pred(y=y, yadj=np.sin(x))
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 2  # observed + adjusted only


def test_plot_dendrogram():
    pytest.importorskip("scipy")
    from scipy.cluster.hierarchy import linkage
    rng = np.random.default_rng(1)
    z = linkage(rng.normal(size=(15, 3)), method="ward")
    ax = dal.plot_dendrogram(z, labels=False)
    assert isinstance(ax, Axes)
    # scipy draws the dendrogram as LineCollections
    assert len(ax.collections) > 0 or len(ax.lines) > 0
