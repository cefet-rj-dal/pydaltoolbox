"""Unit tests for feature-engineering transforms."""
import pandas as pd
import pytest

from pydaltoolbox import Aggregation, FeatureGeneration, HierarchyCut, datasets


def test_feature_generation_callable_and_string():
    iris = datasets.iris()
    gen = FeatureGeneration(
        sepal_area=lambda df: df["sepal_length"] * df["sepal_width"],
        ratio="sepal_length / sepal_width",
    )
    out = gen.transform(iris)
    assert "sepal_area" in out.columns and "ratio" in out.columns
    assert out["sepal_area"].iloc[0] == iris["sepal_length"].iloc[0] * iris["sepal_width"].iloc[0]
    assert out.shape[0] == len(iris)


def test_aggregation_tuple_and_callable():
    iris = datasets.iris()
    agg = Aggregation(
        "Species",
        mean_sepal=("sepal_length", "mean"),
        n=("sepal_length", "size"),
        max_petal=lambda g: g["petal_length"].max(),
    )
    out = agg.transform(iris)
    assert set(out["Species"]) == {"setosa", "versicolor", "virginica"}
    assert (out["n"] == 50).all()
    setosa_mean = iris[iris["Species"] == "setosa"]["sepal_length"].mean()
    assert out.loc[out["Species"] == "setosa", "mean_sepal"].iloc[0] == pytest.approx(setosa_mean)


def test_hierarchy_cut_creates_level_column():
    iris = datasets.iris()
    hc = HierarchyCut(
        "sepal_length",
        breaks=[-float("inf"), 5.5, 6.5, float("inf")],
        labels=["low", "mid", "high"],
    )
    out = hc.transform(iris)
    assert "sepal_length_Level" in out.columns
    assert set(out["sepal_length_Level"].dropna().unique()) <= {"low", "mid", "high"}
    assert out["sepal_length_Level"].notna().all()


def test_hierarchy_cut_custom_name():
    iris = datasets.iris()
    out = HierarchyCut("sepal_length", breaks=[-float("inf"), 6, float("inf")],
                       new_attribute="size").transform(iris)
    assert "size" in out.columns


def test_transforms_return_dataframe():
    iris = datasets.iris()
    assert isinstance(FeatureGeneration(x="sepal_length * 2").transform(iris), pd.DataFrame)
