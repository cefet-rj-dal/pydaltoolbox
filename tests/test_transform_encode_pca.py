"""Unit tests for encoding, PCA and curvature transforms."""
import numpy as np

from pydaltoolbox import CategMapping, DtPca, FitCurvatureMin, datasets


def test_categ_mapping_one_hot_keeps_other_columns():
    iris = datasets.iris()
    out = CategMapping("Species").fit(iris).transform(iris)
    # original column replaced by one indicator per level
    assert "Species" not in out.columns
    for level in ["setosa", "versicolor", "virginica"]:
        assert f"Species_{level}" in out.columns
    # numeric feature columns are preserved
    assert "sepal_length" in out.columns
    # each row is one-hot (exactly one 1 across the indicators)
    ind = out[[f"Species_{lv}" for lv in ["setosa", "versicolor", "virginica"]]]
    assert (ind.sum(axis=1) == 1).all()


def test_categ_mapping_consistent_columns_train_test():
    iris = datasets.iris()
    cm = CategMapping("Species").fit(iris)
    subset = iris[iris["Species"] == "setosa"]  # only one level present
    out = cm.transform(subset)
    # all three indicator columns still present (from fit), even if some are all zero
    for level in ["setosa", "versicolor", "virginica"]:
        assert f"Species_{level}" in out.columns


def test_pca_auto_components_and_reattaches_target():
    iris = datasets.iris()
    pca = DtPca("Species").fit(iris)
    out = pca.transform(iris)
    assert 1 <= pca.components <= 4
    assert "Species" in out.columns
    assert out.shape[0] == len(iris)
    assert out.shape[1] == pca.components + 1  # components + target


def test_pca_explicit_components():
    iris = datasets.iris()
    out = DtPca("Species", components=2).fit(iris).transform(iris)
    assert list(out.columns) == ["PC1", "PC2", "Species"]


def test_fit_curvature_min_finds_elbow():
    # concave increasing curve: elbow should be early
    y = np.array([0.0, 0.7, 0.9, 0.97, 0.99, 1.0])
    res = FitCurvatureMin().transform(y)
    assert 1 <= int(res["x"].iloc[0]) <= len(y)
