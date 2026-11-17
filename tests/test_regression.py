"""Unit tests for the regression family (mirrors R examples / unit tests)."""
import warnings

import numpy as np
import pytest

from pydaltoolbox import (
    RegDtree,
    RegKnn,
    RegLm,
    RegMlp,
    RegRf,
    RegSvm,
    SampleRandom,
    datasets,
)

# factory taking the target attribute name
REGRESSORS = {
    "lm": lambda a: RegLm(a),
    "dtree": lambda a: RegDtree(a, random_state=1),
    "knn": lambda a: RegKnn(a),
    "rf": lambda a: RegRf(a, random_state=1),
    "svm": lambda a: RegSvm(a),
    "mlp": lambda a: RegMlp(a, random_state=1, maxit=2000),
}

TARGET = "target"


@pytest.fixture(scope="module")
def split():
    df = datasets.diabetes()
    return SampleRandom(seed=42).train_test(df)


@pytest.mark.parametrize("name", list(REGRESSORS))
def test_regressor_fit_predict_evaluate(name, split):
    model = REGRESSORS[name](TARGET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(split["train"])
        pred = model.predict(split["test"].drop(columns=TARGET))
    assert len(pred) == len(split["test"])
    assert np.isfinite(pred).all()
    ev = model.evaluate(split["test"][TARGET], pred)
    metrics = ev["metrics"].iloc[0]
    for col in ("mse", "smape", "R2"):
        assert col in ev["metrics"].columns
    assert np.isfinite(metrics["mse"]) and metrics["mse"] >= 0


def test_linear_regression_has_reasonable_fit(split):
    model = RegLm(TARGET).fit(split["train"])
    pred = model.predict(split["test"].drop(columns=TARGET))
    r2 = model.evaluate(split["test"][TARGET], pred)["metrics"].iloc[0]["R2"]
    assert r2 > 0.3  # diabetes linear baseline


def test_predict_accepts_frame_with_or_without_target():
    df = datasets.diabetes()
    model = RegDtree(TARGET, random_state=1).fit(df)
    with_target = model.predict(df)
    without_target = model.predict(df.drop(columns=TARGET))
    assert np.array_equal(with_target, without_target)
