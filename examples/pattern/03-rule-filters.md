# Interest Filters and Automatic Thresholds

Association-rule mining often produces many rules, most of them trivial. Two tools help:
`RuleFilterInterest` keeps only rules that pass classical interest measures, and the support
/ confidence **threshold strategies** estimate sensible thresholds from the data when you
leave `supp` (or `conf`) at `0`.

**Reference:** Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*
(3rd ed.). Morgan Kaufmann.

> Pattern mining needs the optional `mlxtend` dependency: `pip install pydaltoolbox[patterns]`.


```python
import pandas as pd

import pydaltoolbox as dal

transactions = pd.DataFrame(
    {
        "milk":   [1, 1, 0, 1, 1, 1],
        "bread":  [1, 1, 1, 1, 0, 1],
        "butter": [0, 1, 0, 1, 1, 1],
        "beer":   [0, 0, 1, 0, 0, 0],
    }
)
```

## Automatic thresholds

With `supp = 0` (and, for rules, `conf = 0`), `PatApriori` estimates the thresholds during
`fit`. The default support strategy finds the knee of the item-frequency curve; the default
confidence strategy uses the right-hand-side baseline. The estimated values are kept on the
fitted miner.


```python
miner = dal.PatApriori(target="rules", supp=0, conf=0)
miner.fit(transactions)
print("estimated support:", round(miner._supp_resolved, 3))
print("estimated confidence:", round(miner._conf_resolved, 3))
```

    estimated support: 0.5
    estimated confidence: 0.5
    

You can also choose a strategy explicitly:


```python
miner = dal.PatApriori(
    target="rules",
    support_strategy=dal.SupportThreshold("min_count", min_count=3),
    confidence_strategy=dal.ConfidenceThreshold("fixed", value=0.6),
)
miner.fit(transactions)
round(miner._supp_resolved, 3), round(miner._conf_resolved, 3)
```




    (0.5, 0.6)



## Filtering by interest measures

`RuleFilterInterest` keeps rules that pass every configured bound: minimum `lift`,
Kulczynski (`kulc`), support, confidence and count, and maximum imbalance ratio (`ir`). The
`kulc`, `ir` and `count` columns are derived from the measures `mlxtend` reports (these
values match `arules::interestMeasure`).


```python
interest = dal.RuleFilterInterest(lift_min=1.1, kulc_min=0.6, ir_max=0.5)
miner = dal.PatApriori(
    target="rules", supp=0.4, conf=0.5, minlen=2, rule_filter=interest
)
miner.fit(transactions)
rules = miner.discover(transactions)
rules[["antecedents", "consequents", "support", "confidence", "lift", "kulc", "ir"]]
```




           antecedents      consequents   support  confidence   lift  kulc   ir
    0         (butter)           (milk)  0.666667        1.00  1.200  0.90  0.2
    1           (milk)         (butter)  0.666667        0.80  1.200  0.90  0.2
    2  (butter, bread)           (milk)  0.500000        1.00  1.200  0.80  0.4
    3    (milk, bread)         (butter)  0.500000        0.75  1.125  0.75  0.0
    4         (butter)    (milk, bread)  0.500000        0.75  1.125  0.75  0.0
    5           (milk)  (butter, bread)  0.500000        0.60  1.200  0.80  0.4


