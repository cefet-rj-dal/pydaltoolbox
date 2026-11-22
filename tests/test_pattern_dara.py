"""Tests for DARA (divergent association rules) and the rule-filter framework."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlxtend")

from pydaltoolbox import (
    PatApriori,
    RuleFilter,
    RuleFilterDara,
    RuleFilterNone,
    dara,
    dara_score,
)
from pydaltoolbox.pattern.dara import _dara_counter, parse_items


def test_counter_matches_paper_table1():
    # Baroni et al. (2020), Table 1: white(100,10), black(200,20), gray(300,5) -> +1,+1,-2
    counter = _dara_counter(np.array([100.0, 200.0, 300.0]), np.array([10.0, 20.0, 5.0]))
    assert list(counter) == [1.0, 1.0, -2.0]


def test_counter_is_order_independent():
    ds = np.array([100.0, 200.0, 300.0])
    rl = np.array([10.0, 20.0, 5.0])
    base = dict(zip(["w", "b", "g"], _dara_counter(ds, rl)))
    perm = [2, 0, 1]
    shuffled = dict(zip([["w", "b", "g"][i] for i in perm], _dara_counter(ds[perm], rl[perm])))
    assert base == shuffled


def test_parse_items():
    assert parse_items(frozenset(["a=x", "b=y"])) == {"a": "x", "b": "y"}
    assert parse_items(frozenset(["noeq"])) == {}


@pytest.fixture
def divergent():
    """col=A frequent in data but rare in rules; col=B the opposite -> B positive."""
    raw = pd.DataFrame({"col": list("AAAAAB"), "y": ["s"] * 6})
    rules = pd.DataFrame(
        {
            "antecedents": [
                frozenset(["col=A"]),
                frozenset(["col=B"]),
                frozenset(["col=B"]),
                frozenset(["col=B"]),
            ],
            "consequents": [frozenset(["y=s"])] * 4,
        }
    )
    return raw, rules


def test_dara_ranking_matches_r(divergent):
    raw, rules = divergent
    table = dara(raw, rules, rhs_attr="y", rhs_value="s")["col"]
    got = {r.value: (r.dataset, r.rules, r.c) for r in table.itertuples()}
    assert got == {"A": (5.0, 1.0, -1.0), "B": (1.0, 3.0, 1.0)}  # parity with R pat_dara


def test_dara_excludes_rhs_attr(divergent):
    raw, rules = divergent
    assert "y" not in dara(raw, rules, rhs_attr="y", rhs_value="s")


def test_dara_rule_weight(divergent):
    raw, rules = divergent
    rules = rules.assign(support=[0.5, 0.1, 0.1, 0.1])
    table = dara(raw, rules, rhs_attr="y", rule_weight="support")["col"]
    freqs = {r.value: r.rules for r in table.itertuples()}
    assert freqs["A"] == pytest.approx(0.5)
    assert freqs["B"] == pytest.approx(0.3)


def test_dara_score_aggregations():
    ranking = {
        "a": pd.DataFrame({"value": ["x", "y"], "c": [2.0, -1.0]}),
        "b": pd.DataFrame({"value": ["p"], "c": [-4.0]}),
    }
    ante = frozenset(["a=x", "b=p"])
    assert dara_score(ranking, ante, how="sum") == pytest.approx(-2.0)
    assert dara_score(ranking, ante, how="mean") == pytest.approx(-1.0)
    assert dara_score(ranking, ante, how="max") == pytest.approx(2.0)
    assert dara_score(ranking, ante, how="max", use_abs=True) == pytest.approx(4.0)
    assert dara_score(ranking, frozenset(["z=1"]), how="sum") == 0.0


def test_rule_filter_dara_keeps_high_score(divergent):
    raw, rules = divergent
    flt = RuleFilterDara(raw, rhs_attr="y", rhs_value="s", min_score=1)
    out = flt.transform(rules)
    assert (out["dara_score"] >= 1).all()
    # only the three col=B rules score +1
    assert len(out) == 3
    assert out.attrs["filtered_from"] == 4
    assert "dara_ranking" in out.attrs


def test_rule_filter_dara_top(divergent):
    raw, rules = divergent
    flt = RuleFilterDara(raw, rhs_attr="y", rhs_value="s", min_score=-10, top=2)
    assert len(flt.transform(rules)) == 2


def test_rule_filter_none_is_noop(divergent):
    _, rules = divergent
    out = RuleFilterNone().transform(rules)
    assert len(out) == len(rules)
    assert out.attrs["filtered_from"] == len(rules)


def test_rule_filter_is_daltransform(divergent):
    from pydaltoolbox import DalTransform

    raw, rules = divergent
    flt = RuleFilterDara(raw, rhs_attr="y", rhs_value="s", min_score=1)
    assert isinstance(flt, DalTransform)
    assert flt.fit(rules) is flt  # fit is a chainable no-op
    # action dispatches to transform
    pd.testing.assert_frame_equal(flt.action(rules), flt.transform(rules))


def test_patapriori_end_to_end_with_dara():
    raw = pd.DataFrame(
        {
            "col": ["A", "A", "A", "A", "B", "B"],
            "extra": ["m", "n", "m", "n", "m", "n"],
            "y": ["s", "s", "s", "s", "s", "s"],
        }
    )
    oh = pd.get_dummies(raw, prefix_sep="=")
    flt = RuleFilterDara(raw, rhs_attr="y", rhs_value="s", min_score=-100)
    pm = PatApriori(target="rules", supp=0.1, conf=0.1, minlen=2, rhs=["y=s"], rule_filter=flt)
    pm.fit(oh)
    rules = pm.discover(oh)
    assert "dara_score" in rules.columns
    assert "dara_ranking" in rules.attrs
    assert isinstance(flt, RuleFilter)


def test_rule_filter_rejected_on_itemsets():
    raw = pd.DataFrame({"col": ["A", "B"], "y": ["s", "t"]})
    flt = RuleFilterDara(raw, rhs_attr="y")
    with pytest.raises(ValueError):
        PatApriori(target="frequent itemsets", supp=0.1, rule_filter=flt)


def test_rule_filter_dara_requires_rhs_attr():
    with pytest.raises(ValueError):
        RuleFilterDara(pd.DataFrame({"y": ["s"]}), rhs_attr=None)


def test_rule_filter_dara_bad_score():
    with pytest.raises(ValueError):
        RuleFilterDara(pd.DataFrame({"y": ["s"]}), rhs_attr="y", score="median")
