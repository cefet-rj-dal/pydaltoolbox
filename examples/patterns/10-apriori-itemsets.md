# Apriori Frequent Itemsets (PatApriori)

## About the method
- `PatApriori` with `target="frequent itemsets"`: identify groups of items that often appear together. Itemsets capture co-occurrence, not implication.

## Didactic goal
Show item-level filtering explicitly: the support threshold defines the candidate pool, `include`/`exclude` restrict the item vocabulary, and the quality filter keeps only the strongest surviving itemsets.

> Pattern mining needs the optional `mlxtend` dependency: `pip install pydaltoolbox[patterns]`.

## Transactional data (one-hot)


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



## Mine frequent itemsets with item and quality filters
`include`/`exclude` act as semantic filters over which items may appear in the result; the quality filter is a second stage that keeps only the itemsets that are still strong after the structural filtering.


```python
utils = dal.PatUtils()
miner = dal.PatApriori(
    target="frequent itemsets",
    supp=0.4,
    maxlen=3,
    include=["milk", "bread", "butter"],
    quality_filter=utils.quality_min(support=0.5),
).fit(transactions)
itemsets = miner.discover(transactions)
itemsets
```




        support               itemsets
    0  0.833333                 (milk)
    1  0.833333                (bread)
    2  0.666667               (butter)
    3  0.666667          (bread, milk)
    4  0.666667         (butter, milk)
    5  0.500000        (butter, bread)
    6  0.500000  (butter, bread, milk)



## Evaluate the discovered itemsets


```python
miner.evaluate(itemsets)["metrics"]
```




               metric     value       type
    0   pattern_count  7.000000  intrinsic
    1    mean_support  0.666667  intrinsic
    2     mean_length  1.714286  intrinsic
    3  retained_ratio  1.000000     filter



## What to observe
- `include` and `exclude` restrict the item vocabulary before the quality filter runs.
- The quality filter keeps only the itemsets that remain strong after structural filtering.
- Itemsets are not directional rules: they capture co-occurrence, not implication.

## References
- Agrawal, R., and Srikant, R. (1994). *Fast Algorithms for Mining Association Rules in Large Databases.* VLDB.
