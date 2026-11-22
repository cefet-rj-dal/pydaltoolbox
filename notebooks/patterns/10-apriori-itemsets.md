---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

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

## Evaluate the discovered itemsets
```python
miner.evaluate(itemsets)["metrics"]
```

## What to observe
- `include` and `exclude` restrict the item vocabulary before the quality filter runs.
- The quality filter keeps only the itemsets that remain strong after structural filtering.
- Itemsets are not directional rules: they capture co-occurrence, not implication.

## References
- Agrawal, R., and Srikant, R. (1994). *Fast Algorithms for Mining Association Rules in Large Databases.* VLDB.
