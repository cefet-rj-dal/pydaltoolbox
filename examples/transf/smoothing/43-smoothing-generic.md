# Smoothing — base family (Smoothing)

## About the transformation
- `Smoothing`: base family behind the concrete strategies `SmoothingInter`, `SmoothingFreq`, `SmoothingQuantization`, and `SmoothingCluster`. The base constructor holds the generic tuning logic the concrete classes inherit.

## Didactic goal
Expose the role of the base object: the concrete classes inherit the same fit/transform/tuning contract, differing only in how the bin edges are computed.

## Environment setup


```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Use a concrete strategy with several candidate bin counts; the base tuning logic chooses one


```python
iris = dal.datasets.iris()
obj = dal.SmoothingInter(n=[2, 3, 4, 5]).fit(iris["sepal_length"])
bins = obj.transform(iris["sepal_length"])
print("chosen n:", obj.n)
pd.Series(bins).value_counts().sort_index()
```

    chosen n: 3
    




    5.018644    59
    6.136620    71
    7.235000    20
    Name: count, dtype: int64


