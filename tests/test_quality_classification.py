"""Quality / R-parity guards for classifiers (step 0.23.x).

Paired against R on the IDENTICAL split (see
``daltoolbox/develop/parity/classification_breast*``). The dataset is scikit-learn
``breast_cancer`` -- harder than iris (binary, but features span wildly different
scales), which exposes scale- and default-faithfulness problems that iris (all
features in cm) hides.

The gap that motivated these guards was in R, not Python: R ``cla_glm`` inverted its
predictions whenever the ``positive`` class was not the last factor level (fixed by
releveling in ``fit.cla_glm``). The Python ``ClaGlm`` was already correct; these tests
lock that in so a future adapter change cannot silently degrade it.

Thresholds are loose on purpose -- the point is to catch a learner collapsing toward
baseline/degenerate behaviour, not to pin exact scikit-learn numbers.
"""
import warnings

import pandas as pd
from sklearn.datasets import load_breast_cancer

import pydaltoolbox as dal

warnings.filterwarnings("ignore")

MAJORITY = 0.667  # most-frequent-class accuracy on this split


def _data():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df = df.drop(columns="target")
    df["target"] = pd.Categorical.from_codes(raw.target, raw.target_names)
    return df


def _split():
    df = _data()
    slevels = list(df["target"].cat.categories)
    sp = dal.SampleRandom(seed=1).train_test(df)
    return sp["train"], sp["test"], slevels


def _acc(model, train, test):
    model.fit(train)
    pred = model.predict(test.drop(columns="target"))
    ev = model.evaluate(test["target"], pred)
    return float(ev["metrics"]["accuracy"].iloc[0])


def test_claglm_not_inverted():
    """ClaGlm must track the true labels (R parity ~0.95), never collapse/invert."""
    train, test, slevels = _split()
    acc = _acc(dal.ClaGlm("target", slevels), train, test)
    assert acc > 0.90, f"ClaGlm degraded/inverted: accuracy={acc:.3f} (expected ~0.95)"


def test_classification_ml_beats_majority():
    """Core ML classifiers should clearly beat the majority baseline on breast_cancer."""
    train, test, slevels = _split()
    for model in (
        dal.ClaDtree("target", slevels),
        dal.ClaKnn("target", slevels, k=5),
        dal.ClaNb("target", slevels),
        dal.ClaGlm("target", slevels),
        dal.ClaRf("target", slevels, ntree=30, mtry=5),
        dal.ClaMlp("target", slevels, size=5, decay=0.1),
        dal.ClaSvm("target", slevels),
    ):
        acc = _acc(model, train, test)
        assert acc > MAJORITY + 0.15, (
            f"{type(model).__name__} near-baseline: accuracy={acc:.3f}"
        )
