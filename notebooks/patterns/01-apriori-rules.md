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

With `target="frequent itemsets"`, `discover` returns the itemsets whose support reaches
the threshold:

```python
miner = dal.PatApriori(target="frequent itemsets", supp=0.4, minlen=1, maxlen=3).fit(transactions)
itemsets = miner.discover(transactions)
itemsets
```

With `target="rules"`, the same support floor is used to build rules, keeping those above
the confidence threshold. Each rule carries its support, confidence and lift:

```python
rule_miner = dal.PatApriori(target="rules", supp=0.4, conf=0.6, minlen=2).fit(transactions)
rules = rule_miner.discover(transactions)
rules[["antecedents", "consequents", "support", "confidence", "lift"]]
```

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
