"""Tests for threshold strategies and the interest rule filter (step 1.8b)."""
import pandas as pd
import pytest

pytest.importorskip("mlxtend")

from pydaltoolbox import (
    ConfidenceThreshold,
    PatApriori,
    RuleFilterInterest,
    SupportThreshold,
)


@pytest.fixture
def basket():
    return pd.DataFrame(
        {
            "milk": [1, 1, 0, 1, 1, 1],
            "bread": [1, 1, 1, 1, 0, 1],
            "butter": [0, 1, 0, 1, 1, 1],
            "beer": [0, 0, 1, 0, 0, 0],
        }
    )


# -- support threshold strategies -------------------------------------------------
@pytest.mark.parametrize("method", ["curvature", "min_count", "sqrt", "quantile", "density"])
def test_support_threshold_methods_in_bounds(basket, method):
    strat = SupportThreshold(method=method, min_support=0.01, max_support=0.9)
    s = strat.resolve(basket)
    assert 0.01 <= s <= 0.9


def test_support_threshold_min_count(basket):
    # min_count / n = 3 / 6 = 0.5
    assert SupportThreshold(method="min_count", min_count=3).resolve(basket) == pytest.approx(0.5)


def test_support_threshold_empty_raises():
    with pytest.raises(ValueError):
        SupportThreshold().resolve(pd.DataFrame({"a": []}))


def test_support_threshold_bad_method():
    with pytest.raises(ValueError):
        SupportThreshold(method="nope")


# -- confidence threshold strategies ----------------------------------------------
def test_confidence_fixed():
    c = ConfidenceThreshold(method="fixed", value=0.7).resolve(pd.DataFrame({"y=s": [True]}))
    assert c == pytest.approx(0.7)


def test_confidence_rhs_baseline(basket):
    # milk frequency = 5/6 ~= 0.8333; + margin 0.1 = 0.9333 (within [0.5, 0.95])
    c = ConfidenceThreshold(method="rhs_baseline", margin=0.1, max_confidence=0.95)
    assert c.resolve(basket, rhs=["milk"]) == pytest.approx(5 / 6 + 0.1)


def test_confidence_support_adaptive():
    c = ConfidenceThreshold(method="support_adaptive", min_confidence=0.0, max_confidence=1.0)
    assert c.resolve(pd.DataFrame({"a": [True]}), support=0.3) == pytest.approx(0.7)


def test_confidence_bad_method():
    with pytest.raises(ValueError):
        ConfidenceThreshold(method="nope")


# -- auto-estimation in PatApriori ------------------------------------------------
def test_patapriori_auto_estimates_thresholds(basket):
    pm = PatApriori(target="rules", supp=0, conf=0).fit(basket)
    assert pm._supp_resolved is not None and pm._supp_resolved > 0
    assert pm._conf_resolved is not None and pm._conf_resolved > 0
    rules = pm.discover(basket)
    assert (rules["support"] >= pm._supp_resolved - 1e-9).all()
    assert (rules["confidence"] >= pm._conf_resolved - 1e-9).all()


def test_patapriori_explicit_supp_wins(basket):
    pm = PatApriori(target="frequent itemsets", supp=0.5).fit(basket)
    assert pm._supp_resolved == pytest.approx(0.5)


def test_patapriori_negative_supp_raises():
    with pytest.raises(ValueError):
        PatApriori(supp=-0.1)


# -- interest rule filter ---------------------------------------------------------
def test_interest_filter_measures_match_r(basket):
    # lift/kulc/ir verified identical to arules::interestMeasure on this data
    pm = PatApriori(target="rules", supp=0.4, conf=0.5, minlen=2).fit(basket)
    flt = RuleFilterInterest(lift_min=1.1, kulc_min=0.6, ir_max=0.5)
    kept = flt.transform(pm.discover(basket))
    assert len(kept) > 0
    assert (kept["lift"] >= 1.1).all()
    assert (kept["kulc"] >= 0.6).all()
    assert (kept["ir"] <= 0.5).all()


def test_interest_filter_count_needs_n(basket):
    pm = PatApriori(target="rules", supp=0.4, conf=0.5, minlen=2).fit(basket)
    rules = pm.discover(basket)  # carries attrs['n_transactions']
    kept = RuleFilterInterest(lift_min=None, count_min=4).transform(rules)
    assert (kept["count"] >= 4).all()


def test_interest_filter_missing_measure_raises():
    rules = pd.DataFrame({"antecedents": [frozenset(["a"])], "consequents": [frozenset(["b"])]})
    with pytest.raises(ValueError):
        RuleFilterInterest(lift_min=1.0).transform(rules)


def test_interest_filter_integrated(basket):
    flt = RuleFilterInterest(lift_min=1.1)
    pm = PatApriori(target="rules", supp=0.4, conf=0.5, minlen=2, rule_filter=flt).fit(basket)
    kept = pm.discover(basket)
    assert (kept["lift"] >= 1.1).all()
    assert kept.attrs["filtered_from"] >= len(kept)
