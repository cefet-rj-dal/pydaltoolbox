"""Quality / R-parity guards for regression learners (step 2.x, 0.23.0).

These check that the ML regressors are not degenerate on the diabetes dataset -- in
particular that the decision tree generalizes (it used to overfit to a negative test
R2 before the ``mindev``-equivalent pre-pruning was added, while R ``tree::tree`` gave
~0.24 on the same split). Thresholds are loose so the test is robust across scikit-learn
versions; the point is to catch a learner collapsing to worse-than-mean behaviour.
"""
import warnings

import pydaltoolbox as dal

warnings.filterwarnings("ignore")


def _split():
    data = dal.datasets.diabetes()
    sp = dal.SampleRandom(seed=1).train_test(data)
    return sp["train"], sp["test"]


def _r2(model, train, test):
    model.fit(train)
    pred = model.predict(test.drop(columns="target"))
    return float(model.evaluate(test["target"], pred)["metrics"]["R2"].iloc[0])


def test_regdtree_generalizes_not_negative():
    """RegDtree must generalize (R tree::tree ~ 0.24 on this split), not overfit to ~0."""
    train, test = _split()
    r2 = _r2(dal.RegDtree("target", random_state=1), train, test)
    assert r2 > 0.15, f"RegDtree overfit: test R2={r2:.3f} (expected ~0.24, R parity)"


def test_regression_ml_not_degenerate():
    """Core ML regressors should beat a mean predictor (R2 > 0) on diabetes."""
    train, test = _split()
    for model in (
        dal.RegDtree("target", random_state=1),
        dal.RegKnn("target", k=5),
        dal.RegRf("target", ntree=30, mtry=7, random_state=1),
        dal.RegMlp("target", size=5, decay=0.54, random_state=1),
    ):
        r2 = _r2(model, train, test)
        assert r2 > 0.0, f"{type(model).__name__} degenerate: test R2={r2:.3f}"
