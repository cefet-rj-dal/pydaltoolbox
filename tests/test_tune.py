"""Unit tests for hyperparameter tuning (ClaTune, RegTune, CluTune)."""
from pydaltoolbox import (
    ClaKnn,
    ClaTune,
    ClusterKmeans,
    CluTune,
    RegKnn,
    RegTune,
    SampleStratified,
    datasets,
)


def test_cla_tune_returns_fitted_model_that_predicts():
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    split = SampleStratified("Species", seed=42).train_test(iris)
    tune = ClaTune(
        ClaKnn("Species", slevels),
        folds=3,
        ranges={"k": [1, 3, 5, 7]},
        seed=42,
    )
    model = tune.fit(split["train"])
    # returns a fitted learner, not the tuner
    assert isinstance(model, ClaKnn)
    assert model.k in (1, 3, 5, 7)
    pred = model.predict(split["test"].drop(columns="Species"))
    acc = model.evaluate(split["test"]["Species"], pred)["metrics"].iloc[0]["accuracy"]
    assert acc > 0.8


def test_reg_tune_minimizes_error_and_returns_model():
    diabetes = datasets.diabetes()
    tune = RegTune(RegKnn("target"), folds=3, ranges={"k": [3, 5, 10, 20]}, seed=42)
    model = tune.fit(diabetes)
    assert isinstance(model, RegKnn)
    assert model.k in (3, 5, 10, 20)
    pred = model.predict(diabetes.drop(columns="target"))
    assert len(pred) == len(diabetes)


def test_clu_tune_selects_k_and_returns_fitted_clusterer():
    iris = datasets.iris().drop(columns="Species")
    tune = CluTune(ClusterKmeans(seed=42), ranges={"k": range(2, 7)})
    model = tune.fit(iris)
    assert isinstance(model, ClusterKmeans)
    assert 2 <= model.k <= 6
    labels = model.cluster(iris)
    assert len(labels) == len(iris)


def test_single_configuration_skips_cv():
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    model = ClaTune(ClaKnn("Species", slevels), ranges={"k": [5]}).fit(iris)
    assert model.k == 5
