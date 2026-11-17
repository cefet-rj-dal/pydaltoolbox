"""Unit tests for class-balancing transforms."""
import pandas as pd

from pydaltoolbox import BalOversampling, BalSubsampling, SampleBalance, datasets


def _imbalanced():
    iris = datasets.iris()
    return pd.concat(
        [
            iris[iris["Species"] == "setosa"].head(50),
            iris[iris["Species"] == "versicolor"].head(20),
            iris[iris["Species"] == "virginica"].head(10),
        ]
    ).reset_index(drop=True)


def test_sample_balance_down_reduces_to_minority():
    out = SampleBalance("Species", method="down", seed=1).transform(_imbalanced())
    counts = out["Species"].value_counts()
    counts = counts[counts > 0]
    assert counts.nunique() == 1 and counts.iloc[0] == 10


def test_sample_balance_up_grows_to_majority():
    out = SampleBalance("Species", method="up", seed=1).transform(_imbalanced())
    counts = out["Species"].value_counts()
    counts = counts[counts > 0]
    assert counts.nunique() == 1 and counts.iloc[0] == 50


def test_subsampling_reduces_to_minority():
    out = BalSubsampling("Species", seed=1).transform(_imbalanced())
    counts = out["Species"].value_counts()
    counts = counts[counts > 0]
    assert set(counts) == {10}


def test_random_oversampling_grows_to_majority():
    out = BalOversampling("Species", method="random", seed=1).transform(_imbalanced())
    counts = out["Species"].value_counts()
    counts = counts[counts > 0]
    assert set(counts) == {50}


def test_smote_oversampling_grows_and_creates_synthetic_rows():
    imb = _imbalanced()
    out = BalOversampling("Species", method="smote", k=5, seed=1).transform(imb)
    counts = out["Species"].value_counts()
    counts = counts[counts > 0]
    assert set(counts) == {50}
    # synthetic minority rows should include values not present in the original
    orig_virg = imb[imb["Species"] == "virginica"]["sepal_length"].tolist()
    new_virg = out[out["Species"] == "virginica"]["sepal_length"].tolist()
    assert len(new_virg) == 50
    assert any(v not in orig_virg for v in new_virg)
