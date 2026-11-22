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

# Sequential Pattern Mining (PatPrefixSpan)

PrefixSpan discovers frequent **sequential patterns** — ordered lists of itemsets (events)
that occur, in order, in a fraction of the input sequences at or above a support threshold.
It grows patterns one prefix at a time and mines the projected database of suffixes, adding
either an item to the last itemset (*i-extension*) or a new itemset (*s-extension*).

**Reference:** Pei, J., Han, J., Mortazavi-Asl, B., Wang, J., Pinto, H., Chen, Q., Dayal, U.,
& Hsu, M. (2001). *PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected
Pattern Growth.* ICDE.

A sequence is a list of events; each event is an itemset (and, optionally, a time). The
classic Zaki example below has four sequences of timed events:

```python
import pydaltoolbox as dal

sequences = [
    [(10, ["C", "D"]), (15, ["A", "B", "C"]), (20, ["A", "B", "F"]), (25, ["A", "C", "D", "F"])],
    [(15, ["A", "B", "F"]), (20, ["E"])],
    [(10, ["A", "B", "F"])],
    [(10, ["D", "G", "H"]), (20, ["B", "F"]), (25, ["A", "G", "H"])],
]

miner = dal.PatPrefixSpan(support=0.5)
miner.fit(sequences)
patterns = miner.discover()
patterns[["sequence", "support", "length"]]
```

## Time-gap constraints

`maxgap` and `mingap` bound the time difference between consecutive events of a pattern.
With `maxgap=5`, only patterns whose matched events are at most 5 time units apart survive:

```python
dal.PatPrefixSpan(support=0.5, maxgap=5).fit(sequences).discover()[["sequence", "support"]]
```

## Long-format input

Sequences can also be given as a long `DataFrame` with `sequenceID`, `eventID` and `item`
columns (the shape produced by `arulesSequences::read_baskets`):

```python
import pandas as pd

rows = [
    {"sequenceID": si, "eventID": t, "item": it}
    for si, seq in enumerate(sequences)
    for t, items in seq
    for it in items
]
dal.PatPrefixSpan(support=0.5).fit(pd.DataFrame(rows)).discover().shape[0]
```

`evaluate` summarizes the discovered set:

```python
miner.evaluate(patterns)["metrics"]
```
