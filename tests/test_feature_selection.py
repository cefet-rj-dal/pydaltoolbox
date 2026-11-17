"""Unit tests for feature-selection transforms."""
from pydaltoolbox import (
    FeatureSelectionCorr,
    FeatureSelectionInfoGain,
    FeatureSelectionLasso,
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


def test_stepwise_selects_subset_and_keeps_target():
    diabetes = datasets.diabetes()
    fs = FeatureSelectionStepwise("target").fit(diabetes)
    out = fs.transform(diabetes)
    assert "target" in out.columns
    assert 1 <= len(fs.selected) <= diabetes.shape[1] - 1


def test_fit_is_chainable():
    iris = datasets.iris()
    fs = FeatureSelectionCorr()
    assert fs.fit(iris) is fs
