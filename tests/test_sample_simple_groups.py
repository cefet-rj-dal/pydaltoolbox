"""Unit tests for simple and group (cluster) sampling."""
import numpy as np

from pydaltoolbox import SampleGroups, SampleSimple, datasets


def test_sample_simple_without_replacement():
    iris = datasets.iris()
    out = SampleSimple(size=10, seed=1).transform(iris)
    assert len(out) == 10
    assert out.index.tolist() == list(range(10))  # reset index


def test_sample_simple_with_replacement_can_exceed_n():
    iris = datasets.iris()
    out = SampleSimple(size=300, replace=True, seed=1).transform(iris)
    assert len(out) == 300


def test_sample_simple_on_1d_array():
    out = SampleSimple(size=5, seed=1).transform(np.arange(100))
    assert len(out) == 5
    assert set(out).issubset(set(range(100)))


def test_sample_groups_keeps_whole_groups():
    iris = datasets.iris()
    out = SampleGroups("Species", n_groups=2, seed=1).transform(iris)
    species = set(out["Species"].unique())
    assert len(species) == 2
    # every selected group keeps all its 50 rows
    assert len(out) == 100


def test_sample_groups_errors_when_too_many_groups():
    iris = datasets.iris()
    try:
        SampleGroups("Species", n_groups=5, seed=1).transform(iris)
        raise AssertionError("expected an error")
    except ValueError:
        pass


def test_sample_simple_is_reproducible_with_seed():
    iris = datasets.iris()
    a = SampleSimple(size=10, seed=42).transform(iris)
    b = SampleSimple(size=10, seed=42).transform(iris)
    assert a.equals(b)
