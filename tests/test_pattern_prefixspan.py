"""Tests for PatPrefixSpan sequential pattern mining (step 1.9).

Expected patterns are the classic Zaki example, verified identical to
``arulesSequences::cspade`` in R.
"""
import pandas as pd
import pytest

from pydaltoolbox import PatPrefixSpan, PatternMiner, PatUtils

ZAKI = [
    [(10, ["C", "D"]), (15, ["A", "B", "C"]), (20, ["A", "B", "F"]), (25, ["A", "C", "D", "F"])],
    [(15, ["A", "B", "F"]), (20, ["E"])],
    [(10, ["A", "B", "F"])],
    [(10, ["D", "G", "H"]), (20, ["B", "F"]), (25, ["A", "G", "H"])],
]


def _as_dict(result):
    return {r.sequence: round(r.support, 2) for r in result.itertuples()}


def test_zaki_support_matches_cspade():
    result = PatPrefixSpan(support=0.5).fit(ZAKI).discover()
    expected = {
        "<{A}>": 1.0, "<{B,F}>": 1.0, "<{B}>": 1.0, "<{F}>": 1.0,
        "<{A,B,F}>": 0.75, "<{A,B}>": 0.75, "<{A,F}>": 0.75,
        "<{B,F},{A}>": 0.5, "<{B},{A}>": 0.5, "<{D},{A}>": 0.5,
        "<{D},{B,F},{A}>": 0.5, "<{D},{B,F}>": 0.5, "<{D},{B},{A}>": 0.5,
        "<{D},{B}>": 0.5, "<{D},{F},{A}>": 0.5, "<{D},{F}>": 0.5,
        "<{D}>": 0.5, "<{F},{A}>": 0.5,
    }
    assert _as_dict(result) == expected  # parity with R cSPADE


def test_dataframe_input_matches_list():
    rows = [
        {"sequenceID": si, "eventID": t, "item": it}
        for si, seq in enumerate(ZAKI)
        for t, items in seq
        for it in items
    ]
    df_result = PatPrefixSpan(support=0.5).fit(pd.DataFrame(rows)).discover()
    list_result = PatPrefixSpan(support=0.5).fit(ZAKI).discover()
    assert _as_dict(df_result) == _as_dict(list_result)


def test_maxgap_matches_cspade():
    assert len(PatPrefixSpan(support=0.5, maxgap=5).fit(ZAKI).discover()) == 11
    assert len(PatPrefixSpan(support=0.5, maxgap=10).fit(ZAKI).discover()) == 17


def test_mingap_matches_cspade():
    assert len(PatPrefixSpan(support=0.5, mingap=10).fit(ZAKI).discover()) == 12


def test_maxlen_bounds_elements():
    result = PatPrefixSpan(support=0.5, maxlen=1).fit(ZAKI).discover()
    assert (result["length"] == 1).all()
    assert len(result) == 8  # the frequent single-element patterns


def test_maxsize_bounds_items_per_element():
    result = PatPrefixSpan(support=0.5, maxsize=1).fit(ZAKI).discover()
    assert not any("{B,F}" in s for s in result["sequence"])


def test_support_auto_estimation():
    pm = PatPrefixSpan(support=0).fit(ZAKI)  # default min_count=2 -> 0.5 on 4 sequences
    assert pm._support_resolved == pytest.approx(0.5)
    assert len(pm.discover()) == 18


def test_quality_filter_and_retained_ratio():
    u = PatUtils()
    pm = PatPrefixSpan(support=0.5, quality_filter=u.quality_min(support=0.75)).fit(ZAKI)
    result = pm.discover()
    assert (result["support"] >= 0.75).all()
    metrics = pm.evaluate(result)["metrics"]
    retained = metrics.loc[metrics["metric"] == "retained_ratio", "value"].iloc[0]
    assert 0.0 < retained < 1.0


def test_evaluate_metrics_structure():
    pm = PatPrefixSpan(support=0.5).fit(ZAKI)
    metrics = pm.evaluate(pm.discover())["metrics"]
    assert set(metrics["metric"]) == {"pattern_count", "mean_support", "mean_length", "retained_ratio"}


def test_is_pattern_miner_and_action_dispatches():
    pm = PatPrefixSpan(support=0.5).fit(ZAKI)
    assert isinstance(pm, PatternMiner)
    pd.testing.assert_frame_equal(pm.action(), pm.discover())


def test_requires_fit():
    with pytest.raises(ValueError):
        PatPrefixSpan(support=0.5).discover()


def test_negative_support_raises():
    with pytest.raises(ValueError):
        PatPrefixSpan(support=-0.1)
