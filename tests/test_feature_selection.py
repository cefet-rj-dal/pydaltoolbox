"""Unit tests for feature-selection transforms."""
import numpy as np
import pandas as pd

from pydaltoolbox import (
    FeatureSelectionCorr,
    FeatureSelectionFss,
    FeatureSelectionInfoGain,
    FeatureSelectionLasso,
    FeatureSelectionRelief,
    FeatureSelectionStepwise,
    datasets,
)


def test_corr_drops_a_correlated_feature_and_keeps_target():
    iris = datasets.iris()
    fs = FeatureSelectionCorr(cutoff=0.9).fit(iris)
    out = fs.transform(iris)
    # petal_length and petal_width are > 0.9 correlated -> one is dropped
    assert len(fs.selected) < 4
    assert "Species" in out.columns  # non-feature column kept


def test_lasso_selects_subset_and_keeps_target():
    diabetes = datasets.diabetes()
    model = FeatureSelectionLasso("target").fit(diabetes)
    out = model.transform(diabetes)
    assert "target" in out.columns
    assert len(model.selected) >= 1
    assert set(out.columns) - {"target"} == set(model.selected)


def test_info_gain_ranks_and_selects_top():
    iris = datasets.iris()
    fs = FeatureSelectionInfoGain("Species", top=2).fit(iris)
    out = fs.transform(iris)
    assert len(fs.selected) == 2
    assert "Species" in out.columns
    # petal features carry the most information about the species
    top_two = set(fs.ranking["feature"].head(2))
    assert top_two <= {"petal_length", "petal_width", "sepal_length", "sepal_width"}
    assert "petal_length" in top_two or "petal_width" in top_two


def test_fss_selects_subset_and_keeps_target():
    diabetes = datasets.diabetes()
    fs = FeatureSelectionFss("target").fit(diabetes)
    out = fs.transform(diabetes)
    assert "target" in out.columns
    assert 1 <= len(fs.selected) <= diabetes.shape[1] - 1
    # ranking is populated and ordered 1..k
    assert list(fs.ranking["score"]) == list(range(1, len(fs.selected) + 1))


def test_stepwise_gaussian_selects_subset_and_keeps_target():
    diabetes = datasets.diabetes()
    fs = FeatureSelectionStepwise("target", family="gaussian", direction="both").fit(diabetes)
    out = fs.transform(diabetes)
    assert "target" in out.columns
    assert 1 <= len(fs.selected) <= diabetes.shape[1] - 1


def test_stepwise_binomial_recovers_informative_features():
    iris = datasets.iris()
    binary = iris.copy()
    binary["is_versicolor"] = (binary["Species"] == "versicolor").astype(int)
    binary = binary.drop(columns=["Species"])
    fs = FeatureSelectionStepwise("is_versicolor", direction="forward").fit(binary)
    out = fs.transform(binary)
    assert "is_versicolor" in out.columns
    assert len(fs.selected) >= 1


def test_stepwise_rejects_non_binary_target_for_binomial():
    iris = datasets.iris()
    try:
        FeatureSelectionStepwise("Species").fit(iris)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-binary binomial target")


def test_relief_ranks_informative_features_above_noise():
    rng = np.random.default_rng(42)
    iris = datasets.iris()
    binary = iris.copy()
    binary["label"] = (binary["Species"] == "versicolor").astype(int)
    binary = binary.drop(columns=["Species"])
    # add a pure-noise feature that Relief should rank below the real ones
    binary["noise"] = rng.standard_normal(len(binary))
    fs = FeatureSelectionRelief("label", m=50).fit(binary)
    assert fs.ranking is not None
    # noise should not be the top-weighted feature
    assert fs.ranking.iloc[0]["feature"] != "noise"


def test_relief_top_and_transform():
    iris = datasets.iris()
    binary = iris.copy()
    binary["label"] = (binary["Species"] == "versicolor").astype(int)
    binary = binary.drop(columns=["Species"])
    fs = FeatureSelectionRelief("label", top=2).fit(binary)
    out = fs.transform(binary)
    assert len(fs.selected) == 2
    assert "label" in out.columns
    assert set(out.columns) - {"label"} == set(fs.selected)


def test_relief_requires_two_classes():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "label": [1, 1, 1]})
    try:
        FeatureSelectionRelief("label").fit(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError for single-class target")


def test_relieff_seed_is_reproducible():
    iris = datasets.iris()
    binary = iris.assign(label=(iris["Species"] == "versicolor").astype(int)).drop(columns=["Species"])
    r1 = FeatureSelectionRelief("label", m=50, k=10, seed=7).fit(binary).ranking
    r2 = FeatureSelectionRelief("label", m=50, k=10, seed=7).fit(binary).ranking
    assert list(r1["feature"]) == list(r2["feature"])
    assert r1["score"].tolist() == r2["score"].tolist()


def test_relieff_ranks_all_real_features_above_noise():
    """Full ReliefF (k neighbours) cleanly separates real iris features from noise."""
    rng = np.random.default_rng(0)
    iris = datasets.iris()
    binary = iris.assign(label=(iris["Species"] == "versicolor").astype(int)).drop(columns=["Species"])
    for j in range(3):
        binary[f"noise{j}"] = rng.standard_normal(len(binary))
    real = {"sepal_length", "sepal_width", "petal_length", "petal_width"}
    fs = FeatureSelectionRelief("label", m=50, k=10, seed=0).fit(binary)
    scores = fs.ranking.set_index("feature")["score"]
    worst_real = min(scores[f] for f in real)
    best_noise = max(scores[f] for f in scores.index if f.startswith("noise"))
    assert worst_real > best_noise


def test_relieff_multiclass_weighting_runs():
    """ReliefF handles a 3-class target (prior-weighted misses) and recovers petal feats."""
    iris = datasets.iris()
    fs = FeatureSelectionRelief("Species", m=50, k=10, seed=1, top=2).fit(iris)
    assert len(fs.selected) == 2
    assert any(f.startswith("petal") for f in fs.selected)


def test_fit_is_chainable():
    iris = datasets.iris()
    fs = FeatureSelectionCorr()
    assert fs.fit(iris) is fs
