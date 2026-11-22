# Apriori Frequent Itemsets and Rules (PatApriori)

Apriori mines frequent itemsets with a level-wise search that prunes any candidate with
an infrequent subset (the downward-closure property), then derives association rules whose
confidence and lift meet a threshold. Transactions are given as a one-hot encoded
`DataFrame`: one row per transaction, one boolean column per item.

**Reference:** Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association
Rules in Large Databases.* VLDB.

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
transactions
```




       milk  bread  butter  beer
    0     1      1       0     0
    1     1      1       1     0
    2     0      1       0     1
    3     1      1       1     0
    4     1      0       1     0
    5     1      1       1     0



With `target="frequent itemsets"`, `discover` returns the itemsets whose support reaches
the threshold:


```python
miner = dal.PatApriori(target="frequent itemsets", supp=0.4, minlen=1, maxlen=3).fit(transactions)
itemsets = miner.discover(transactions)
itemsets
```




        support               itemsets
    0  0.833333                 (milk)
    1  0.833333                (bread)
    2  0.666667               (butter)
    3  0.666667          (bread, milk)
    4  0.666667         (butter, milk)
    5  0.500000        (bread, butter)
    6  0.500000  (bread, butter, milk)



With `target="rules"`, the same support floor is used to build rules, keeping those above
the confidence threshold. Each rule carries its support, confidence and lift:


```python
rule_miner = dal.PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(transactions)
rules = rule_miner.discover(transactions)
rules[["antecedents", "consequents", "support", "confidence", "lift"]]
```




            antecedents      consequents   support  confidence   lift
    0           (bread)           (milk)  0.666667        0.80  0.960
    1            (milk)          (bread)  0.666667        0.80  0.960
    2          (butter)           (milk)  0.666667        1.00  1.200
    3            (milk)         (butter)  0.666667        0.80  1.200
    4           (bread)         (butter)  0.500000        0.60  0.900
    5          (butter)          (bread)  0.500000        0.75  0.900
    6   (bread, butter)           (milk)  0.500000        1.00  1.200
    7     (bread, milk)         (butter)  0.500000        0.75  1.125
    8    (butter, milk)          (bread)  0.500000        0.75  0.900
    9           (bread)   (butter, milk)  0.500000        0.60  0.900
    10         (butter)    (bread, milk)  0.500000        0.75  1.125
    11           (milk)  (bread, butter)  0.500000        0.60  1.200



A quality filter (from `PatUtils`) narrows the result to the most interesting rules; the
`retained_ratio` metric reports the fraction kept after filtering:


```python
utils = dal.PatUtils()
filtered = dal.PatApriori(
    target="rules", supp=0.4, conf=0.6, minlen=2,
    quality_filter=utils.quality_min(lift=1.1),
).fit(transactions)
kept = filtered.discover(transactions)
filtered.evaluate(kept)["metrics"]
```




                metric     value       type
    0    pattern_count  6.000000  intrinsic
    1     mean_support  0.555556  intrinsic
    2  mean_confidence  0.816667  intrinsic
    3        mean_lift  1.175000  intrinsic
    4      mean_length  2.666667  intrinsic
    5   retained_ratio  0.500000     filter


