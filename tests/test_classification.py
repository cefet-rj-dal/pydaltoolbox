"""Unit tests for the classification family (mirrors R examples / unit tests)."""
import warnings

import pandas as pd
import pytest

from pydaltoolbox import (
    ClaBagging,
    ClaBoosting,
    ClaDtree,
    ClaGlm,
    ClaGlmnet,
    ClaKnn,
    ClaMajority,
    ClaMlp,
    ClaMultinom,
    ClaNb,
    ClaRf,
    ClaRpart,
    ClaSvm,
    SampleStratified,
    datasets,
)

# (factory taking slevels, minimum acceptable test accuracy)
LEARNERS = {
    "dtree": (lambda s: ClaDtree("Species", s, random_state=1), 0.8),
    "rpart": (lambda s: ClaRpart("Species", s, random_state=1), 0.8),
    "knn": (lambda s: ClaKnn("Species", s), 0.8),
    "nb": (lambda s: ClaNb("Species", s), 0.8),
    "rf": (lambda s: ClaRf("Species", s, random_state=1), 0.8),
    "svm": (lambda s: ClaSvm("Species", s, random_state=1), 0.8),
    "mlp": (lambda s: ClaMlp("Species", s, random_state=1), 0.8),
    "glm": (lambda s: ClaGlm("Species", s, random_state=1), 0.8),
    "multinom": (lambda s: ClaMultinom("Species", s, random_state=1), 0.8),
    "glmnet": (lambda s: ClaGlmnet("Species", s, random_state=1), 0.8),
    "bagging": (lambda s: ClaBagging("Species", s, random_state=1), 0.8),
    "boosting": (lambda s: ClaBoosting("Species", s, random_state=1), 0.6),
    "majority": (lambda s: ClaMajority("Species", s), 0.0),
}


@pytest.fixture(scope="module")
def split():
    iris = datasets.iris()
    sr = SampleStratified("Species", seed=42)
    return sr.train_test(iris), list(iris["Species"].cat.categories)


@pytest.mark.parametrize("name", list(LEARNERS))
def test_classifier_fit_predict_evaluate(name, split):
    (tt, slevels) = split
    factory, min_acc = LEARNERS[name]
    model = factory(slevels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(tt["train"])
        pred = model.predict(tt["test"].drop(columns="Species"))
    # predict returns a class-score table with one column per level
    assert isinstance(pred, pd.DataFrame)
    assert list(pred.columns) == [str(s) for s in slevels]
    assert len(pred) == len(tt["test"])
    ev = model.evaluate(tt["test"]["Species"], pred)
    assert ev["metrics"].iloc[0]["accuracy"] >= min_acc


def test_dtree_predict_accepts_frame_with_or_without_target():
    """Contract: predict() selects its feature columns and ignores the target."""
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    model = ClaDtree("Species", slevels, random_state=1).fit(iris)
    with_target = model.predict(iris)
    without_target = model.predict(iris.drop(columns="Species"))
    assert with_target.equals(without_target)


def test_slevels_inferred_when_not_provided():
    iris = datasets.iris()
    model = ClaKnn("Species").fit(iris)
    assert set(model.slevels) == set(iris["Species"].cat.categories)


def test_majority_predicts_single_class():
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    model = ClaMajority("Species", slevels).fit(iris)
    pred = model.predict(iris.drop(columns="Species"))
    # every row maps to the same (most frequent) class
    hard = pred.to_numpy().argmax(axis=1)
    assert len(set(hard)) == 1
