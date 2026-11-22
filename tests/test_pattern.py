"""Tests for the pattern mining family (PatApriori, PatUtils)."""
import pandas as pd
import pytest

pytest.importorskip("mlxtend")

from pydaltoolbox import PatApriori, PatternMiner, PatUtils


@pytest.fixture
def transactions():
    """Small one-hot transaction matrix with known frequent itemsets."""
    return pd.DataFrame(
        {
            "milk": [1, 1, 0, 1, 1],
            "bread": [1, 1, 1, 1, 0],
            "butter": [0, 1, 0, 1, 1],
            "beer": [0, 0, 1, 0, 0],
        }
    )


def _itemsets(transactions, **kwargs):
    miner = PatApriori(target="frequent itemsets", **kwargs).fit(transactions)
    return miner, miner.discover(transactions)


def test_known_itemsets_and_supports(transactions):
    _, itemsets = _itemsets(transactions, supp=0.4, minlen=1, maxlen=3)
    found = {frozenset(s): round(sup, 4) for s, sup in zip(itemsets["itemsets"], itemsets["support"])}
    expected = {
        frozenset(["milk"]): 0.8,
        frozenset(["bread"]): 0.8,
        frozenset(["butter"]): 0.6,
        frozenset(["milk", "bread"]): 0.6,
        frozenset(["milk", "butter"]): 0.6,
        frozenset(["bread", "butter"]): 0.4,
        frozenset(["milk", "bread", "butter"]): 0.4,
    }
    assert found == expected  # parity with R arules on the same data


def test_rules_have_interest_measures(transactions):
    ap = PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(transactions)
    rules = ap.action(transactions)  # action -> discover
    assert len(rules) > 0
    for col in ("antecedents", "consequents", "support", "confidence", "lift"):
        assert col in rules.columns
    assert (rules["confidence"] >= 0.6).all()
    assert (rules["support"] >= 0.4).all()


def test_support_threshold_is_respected(transactions):
    _, itemsets = _itemsets(transactions, supp=0.6, minlen=1, maxlen=3)
    assert (itemsets["support"] >= 0.6).all()
    assert len(itemsets) == 5  # milk, bread, butter, {milk,bread}, {milk,butter}


def test_minlen_maxlen_bounds(transactions):
    _, itemsets = _itemsets(transactions, supp=0.4, minlen=2, maxlen=2)
    lengths = itemsets["itemsets"].apply(len)
    assert lengths.eq(2).all()


def test_quality_filter_and_retained_ratio(transactions):
    u = PatUtils()
    ap = PatApriori(
        target="rules", supp=0.4, conf=0.6, minlen=2,
        quality_filter=u.quality_min(lift=1.1),
    ).fit(transactions)
    rules = ap.discover(transactions)
    assert (rules["lift"] >= 1.1).all()
    metrics = ap.evaluate(rules)["metrics"]
    retained = metrics.loc[metrics["metric"] == "retained_ratio", "value"].iloc[0]
    assert 0.0 < retained <= 1.0


def test_include_exclude_item_filter(transactions):
    _, inc = _itemsets(transactions, supp=0.4, maxlen=3, include=["milk", "bread"])
    for s in inc["itemsets"]:
        assert frozenset(s).issubset({"milk", "bread"})
    _, exc = _itemsets(transactions, supp=0.4, maxlen=3, exclude=["butter"])
    for s in exc["itemsets"]:
        assert "butter" not in s


def test_evaluate_metrics_structure(transactions):
    ap, itemsets = _itemsets(transactions, supp=0.4)
    metrics = ap.evaluate(itemsets)["metrics"]
    assert list(metrics.columns) == ["metric", "value", "type"]
    assert set(metrics["metric"]) == {"pattern_count", "mean_support", "mean_length", "retained_ratio"}


def test_rules_evaluate_includes_confidence_and_lift(transactions):
    ap = PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(transactions)
    metrics = ap.evaluate(ap.discover(transactions))["metrics"]
    assert {"mean_confidence", "mean_lift"}.issubset(set(metrics["metric"]))


def test_schema_guard(transactions):
    ap = PatApriori(supp=0.4).fit(transactions)
    with pytest.raises(ValueError):
        ap.discover(transactions.rename(columns={"milk": "MILK"}))


def test_requires_fit_before_discover(transactions):
    with pytest.raises(ValueError):
        PatApriori(supp=0.4).discover(transactions)


def test_rejects_non_onehot_input():
    bad = pd.DataFrame({"a": [1, 2, 0], "b": [0, 1, 1]})
    with pytest.raises(ValueError):
        PatApriori(supp=0.4).fit(bad)


def test_invalid_params():
    with pytest.raises(ValueError):
        PatApriori(target="nonsense", supp=0.4)
    with pytest.raises(ValueError):
        PatApriori(supp=0.0)


def test_action_dispatches_to_discover(transactions):
    ap = PatApriori(target="frequent itemsets", supp=0.4).fit(transactions)
    assert isinstance(ap, PatternMiner)
    pd.testing.assert_frame_equal(ap.action(transactions), ap.discover(transactions))
