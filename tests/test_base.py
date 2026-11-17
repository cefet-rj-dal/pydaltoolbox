"""Unit tests for the core Experiment Line contract."""
import pandas as pd

from pydaltoolbox import DalBase, DalTransform, adjust_class_label, adjust_factor


def test_set_params_only_assigns_existing_attributes():
    obj = DalBase()
    obj.alpha = 1
    obj.set_params({"alpha": 5, "ghost": 99})
    assert obj.alpha == 5
    assert not hasattr(obj, "ghost")


def test_fit_is_chainable_noop():
    obj = DalBase()
    assert obj.fit(None) is obj


def test_transform_action_dispatch_identity():
    t = DalTransform()
    data = pd.DataFrame({"a": [1, 2, 3]})
    assert t.action(data).equals(data)
    assert t.inverse_transform(data).equals(data)


def test_adjust_class_label_one_hot():
    oh = adjust_class_label(["a", "b", "a"])
    assert list(oh.columns) == ["a", "b"]
    assert oh.iloc[0]["a"] == 1 and oh.iloc[0]["b"] == 0


def test_adjust_factor_from_numeric_codes():
    cat = adjust_factor([1, 2, 1], ["setosa", "versicolor"])
    assert list(cat) == ["setosa", "versicolor", "setosa"]
