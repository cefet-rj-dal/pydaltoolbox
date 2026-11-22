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




               sequence  support  length
    0             <{A}>     1.00       1
    1           <{B,F}>     1.00       1
    2             <{B}>     1.00       1
    3             <{F}>     1.00       1
    4         <{A,B,F}>     0.75       1
    5           <{A,B}>     0.75       1
    6           <{A,F}>     0.75       1
    7       <{B,F},{A}>     0.50       2
    8         <{B},{A}>     0.50       2
    9         <{D},{A}>     0.50       2
    10  <{D},{B,F},{A}>     0.50       3
    11      <{D},{B,F}>     0.50       2
    12    <{D},{B},{A}>     0.50       3
    13        <{D},{B}>     0.50       2
    14    <{D},{F},{A}>     0.50       3
    15        <{D},{F}>     0.50       2
    16            <{D}>     0.50       1
    17        <{F},{A}>     0.50       2



## Time-gap constraints

`maxgap` and `mingap` bound the time difference between consecutive events of a pattern.
With `maxgap=5`, only patterns whose matched events are at most 5 time units apart survive:


```python
dal.PatPrefixSpan(support=0.5, maxgap=5).fit(sequences).discover()[["sequence", "support"]]
```




           sequence  support
    0         <{A}>     1.00
    1       <{B,F}>     1.00
    2         <{B}>     1.00
    3         <{F}>     1.00
    4     <{A,B,F}>     0.75
    5       <{A,B}>     0.75
    6       <{A,F}>     0.75
    7   <{B,F},{A}>     0.50
    8     <{B},{A}>     0.50
    9         <{D}>     0.50
    10    <{F},{A}>     0.50



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




    18



`evaluate` summarizes the discovered set:


```python
miner.evaluate(patterns)["metrics"]
```




               metric      value       type
    0   pattern_count  18.000000  intrinsic
    1    mean_support   0.652778  intrinsic
    2     mean_length   1.722222  intrinsic
    3  retained_ratio   1.000000     filter


