"""Unit tests for the classification family (mirrors R examples / unit tests)."""
import pandas as pd

from pydaltoolbox import ClaDtree, datasets


def test_dtree_predict_returns_class_score_table():
    iris = datasets.iris()
    features = iris.drop(columns="Species")
    slevels = list(iris["Species"].cat.categories)
    model = ClaDtree("Species", slevels, random_state=1).fit(iris)
    pred = model.predict(features)
    assert isinstance(pred, pd.DataFrame)
    assert list(pred.columns) == [str(s) for s in slevels]
    assert len(pred) == len(iris)


def test_dtree_predict_accepts_frame_with_or_without_target():
    """Contract: predict() selects its feature columns and ignores the target."""
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    model = ClaDtree("Species", slevels, random_state=1).fit(iris)
    with_target = model.predict(iris)
    without_target = model.predict(iris.drop(columns="Species"))
    assert with_target.equals(without_target)


def test_dtree_fit_is_chainable():
    iris = datasets.iris()
    model = ClaDtree("Species", random_state=1)
    assert model.fit(iris) is model
    # slevels inferred when not provided
    assert set(model.slevels) == set(iris["Species"].cat.categories)


def test_evaluate_metrics_on_training_data():
    iris = datasets.iris()
    slevels = list(iris["Species"].cat.categories)
    model = ClaDtree("Species", slevels, random_state=1).fit(iris)
    pred = model.predict(iris.drop(columns="Species"))
    ev = model.evaluate(iris["Species"], pred)
    metrics = ev["metrics"].iloc[0]
    assert metrics["accuracy"] > 0.9
    for col in ["TP", "TN", "FP", "FN", "precision", "recall", "f1"]:
        assert col in ev["metrics"].columns
