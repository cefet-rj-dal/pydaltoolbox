"""Unit tests for normalization transforms (mirror R trans_norm examples)."""
import numpy as np
import pandas as pd
import pytest

from pydaltoolbox import MinMax, ZScore, datasets

NUMERIC = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@pytest.mark.parametrize("factory", [MinMax, ZScore])
def test_fit_is_chainable(factory):
    iris = datasets.iris()
    trans = factory()
    assert trans.fit(iris) is trans


def test_minmax_scales_to_unit_range():
    iris = datasets.iris()
    trans = MinMax().fit(iris)
    out = trans.transform(iris)
    for col in NUMERIC:
        assert out[col].min() == pytest.approx(0.0)
        assert out[col].max() == pytest.approx(1.0)


def test_zscore_standardizes_columns():
    iris = datasets.iris()
    out = ZScore().fit(iris).transform(iris)
    for col in NUMERIC:
        assert out[col].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[col].std() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("factory", [MinMax, ZScore])
def test_inverse_transform_round_trips(factory):
    iris = datasets.iris()
    trans = factory().fit(iris)
    back = trans.inverse_transform(trans.transform(iris))
    for col in NUMERIC:
        assert np.allclose(back[col].to_numpy(), iris[col].to_numpy())


@pytest.mark.parametrize("factory", [MinMax, ZScore])
def test_non_numeric_columns_untouched(factory):
    iris = datasets.iris()
    out = factory().fit(iris).transform(iris)
    assert (out["Species"].to_numpy() == iris["Species"].to_numpy()).all()


@pytest.mark.parametrize("factory", [MinMax, ZScore])
def test_constant_column_is_handled(factory):
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
    out = factory().fit(df).transform(df)
    assert np.isfinite(out["a"].to_numpy()).all()  # no division-by-zero NaNs


def test_action_dispatches_to_transform():
    iris = datasets.iris()
    trans = MinMax().fit(iris)
    assert trans.action(iris).equals(trans.transform(iris))
