"""Unit tests for the smoothing family."""
import numpy as np
import pytest

from pydaltoolbox import (
    SmoothingCluster,
    SmoothingFreq,
    SmoothingInter,
    SmoothingQuantization,
    datasets,
)

SIMPLE = [SmoothingInter, SmoothingFreq, SmoothingQuantization]


@pytest.mark.parametrize("factory", SIMPLE)
def test_fit_is_chainable(factory):
    x = datasets.iris()["sepal_length"]
    sm = factory(n=3)
    assert sm.fit(x) is sm


@pytest.mark.parametrize("factory", SIMPLE)
def test_transform_reduces_to_at_most_n_levels(factory):
    x = datasets.iris()["sepal_length"]
    sm = factory(n=3)
    sm = sm.fit(x) if factory is not SmoothingQuantization else factory(n=3, seed=1).fit(x)
    out = sm.transform(x)
    assert len(out) == len(x)
    assert len(np.unique(out)) <= 3  # replaced by bin means -> at most n distinct values


def test_smoothing_preserves_order_of_magnitude():
    x = datasets.iris()["sepal_length"]
    out = SmoothingFreq(n=4).fit(x).transform(x)
    # bin means stay within the data range
    assert out.min() >= x.min() - 1e-9 and out.max() <= x.max() + 1e-9


def test_tuning_picks_n_from_candidates():
    x = datasets.iris()["sepal_length"]
    sm = SmoothingFreq(n=range(2, 8)).fit(x)
    assert isinstance(sm.n, int) and 2 <= sm.n <= 7
    assert sm.interval is not None  # refit with the chosen n


def test_evaluate_entropy_against_class():
    iris = datasets.iris()
    sm = SmoothingFreq(n=3).fit(iris["sepal_length"])
    import pandas as pd

    bins = pd.cut(iris["sepal_length"], bins=np.unique(sm.interval_adj),
                  include_lowest=True, labels=False)
    ev = sm.evaluate(bins, iris["Species"])
    assert "entropy" in ev and ev["entropy"] >= 0


def test_smoothing_cluster_uses_class_label():
    iris = datasets.iris()
    data = iris[["sepal_length", "Species"]]
    sm = SmoothingCluster("Species", n=3, seed=1).fit(data)
    out = sm.transform(iris["sepal_length"])
    assert len(out) == len(iris)
    assert len(np.unique(out)) <= 3
