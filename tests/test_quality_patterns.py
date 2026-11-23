"""Quality / R-parity guards for pattern mining (step 0.23.x).

Paired against R arules on the identical transactions (see
``daltoolbox/develop/parity/patterns_apriori*``).

- Frequent itemsets and their supports are IDENTICAL to R arules.
- Single-item-consequent rules match R arules exactly (support/confidence/lift).
- The only difference is that mlxtend additionally emits multi-item-consequent rules
  (e.g. {butter} => {bread, milk}), which arules::apriori does not generate by default
  -- a documented structural property of the backend, not a bug. We assert every extra
  rule has |consequent| > 1.

PrefixSpan (sequences) and DARA were validated identical to cSPADE and to the
paper/R in earlier work; those live in test_pattern_prefixspan.py / test_pattern_dara.py.
"""
import warnings

import pandas as pd
import pytest

from pydaltoolbox import PatApriori

warnings.filterwarnings("ignore")


def _basket():
    return pd.DataFrame(
        {
            "milk": [1, 1, 0, 1, 1],
            "bread": [1, 1, 1, 1, 0],
            "butter": [0, 1, 0, 1, 1],
            "beer": [0, 0, 1, 0, 0],
        }
    )


# R arules single-consequent rules on this basket (patterns_apriori_samesplit.R):
# (lhs, rhs) -> (support, confidence, lift)
R_RULES = {
    (("butter",), ("milk",)): (0.6, 1.0, 1.25),
    (("milk",), ("butter",)): (0.6, 0.75, 1.25),
    (("bread",), ("milk",)): (0.6, 0.75, 0.9375),
    (("milk",), ("bread",)): (0.6, 0.75, 0.9375),
    (("bread", "butter"), ("milk",)): (0.4, 1.0, 1.25),
    (("butter",), ("bread",)): (0.4, 0.667, 0.833),
    (("butter", "milk"), ("bread",)): (0.4, 0.667, 0.833),
    (("bread", "milk"), ("butter",)): (0.4, 0.667, 1.111),
}


def test_apriori_itemsets_match_arules():
    miner = PatApriori(target="frequent itemsets", supp=0.4, minlen=1, maxlen=3).fit(_basket())
    found = {
        frozenset(s): round(float(sup), 3)
        for s, sup in zip(miner.discover(_basket())["itemsets"], miner.discover(_basket())["support"])
    }
    expected = {
        frozenset(["milk"]): 0.8, frozenset(["bread"]): 0.8, frozenset(["butter"]): 0.6,
        frozenset(["milk", "bread"]): 0.6, frozenset(["milk", "butter"]): 0.6,
        frozenset(["bread", "butter"]): 0.4, frozenset(["milk", "bread", "butter"]): 0.4,
    }
    assert found == expected


def test_apriori_single_consequent_rules_match_arules():
    rules = PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(_basket()).discover(_basket())
    seen = {}
    extra_multi = 0
    for _, r in rules.iterrows():
        lhs, rhs = tuple(sorted(r["antecedents"])), tuple(sorted(r["consequents"]))
        if len(rhs) == 1:
            seen[(lhs, rhs)] = (
                round(float(r["support"]), 3),
                round(float(r["confidence"]), 3),
                round(float(r["lift"]), 3),
            )
        else:
            extra_multi += 1  # mlxtend-only multi-consequent rule

    # every single-consequent rule matches arules (tolerant on rounding)
    assert set(seen) == set(R_RULES)
    for key, (s, c, lift) in R_RULES.items():
        gs, gc, gl = seen[key]
        assert gs == pytest.approx(s, abs=1e-3)
        assert gc == pytest.approx(c, abs=1e-3)
        assert gl == pytest.approx(lift, abs=1e-3)

    # the only divergence from arules is multi-item consequents (documented, not a bug)
    assert extra_multi >= 1
