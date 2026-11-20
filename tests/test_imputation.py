"""Unit tests for predictive (tree-based) imputation."""
import numpy as np

from pydaltoolbox import ImputationTree, datasets


def _iris_with_na():
    iris = datasets.iris().copy()
    iris.loc[[2, 10, 25], "sepal_length"] = np.nan
    return iris


def test_tree_imputation_fills_numeric_target():
    iris = _iris_with_na()
    imp = ImputationTree("sepal_length", random_state=1).fit(iris)
    out = imp.transform(iris)
    assert out["sepal_length"].notna().all()
    # imputed values stay within the observed range
    obs = datasets.iris()["sepal_length"]
    assert out["sepal_length"].min() >= obs.min() - 1e-9
    assert out["sepal_length"].max() <= obs.max() + 1e-9


def test_tree_imputation_is_chainable():
    imp = ImputationTree("sepal_length", random_state=1)
    assert imp.fit(_iris_with_na()) is imp


def test_tree_imputation_only_touches_missing_rows():
    iris = _iris_with_na()
    before = iris["sepal_length"].copy()
    out = ImputationTree("sepal_length", random_state=1).fit(iris).transform(iris)
    observed = before.notna()
    assert np.allclose(out.loc[observed, "sepal_length"], before[observed])


def test_tree_imputation_categorical_target():
    iris = datasets.iris().copy()
    iris["Species"] = iris["Species"].astype(object)
    iris.loc[[3, 15, 40], "Species"] = np.nan
    out = ImputationTree("Species", random_state=1).fit(iris).transform(iris)
    assert out["Species"].notna().all()
    # imputed labels are valid species
    assert set(out["Species"]) <= {"setosa", "versicolor", "virginica"}


def test_explicit_sources_subset():
    iris = _iris_with_na()
    imp = ImputationTree(
        "sepal_length", sources=["petal_length", "petal_width"], random_state=1
    ).fit(iris)
    out = imp.transform(iris)
    assert out["sepal_length"].notna().all()
    assert imp.sources_ == ["petal_length", "petal_width"]
