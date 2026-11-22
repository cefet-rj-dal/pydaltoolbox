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

# Smoothing by Frequency (SmoothingFreq)

## About the technique
- `SmoothingFreq`: discretization/smoothing by frequency (quantiles), producing bins with similar counts.

## Didactic goal
Discretization turns a continuous variable into a small set of ordered bins; smoothing reduces noise by summarizing values into those bins. Choosing the number of bins is the key decision.

## Environment setup
```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Apply the discretization and inspect the bins
```python
iris = dal.datasets.iris()
x = iris["sepal_length"]
sm = dal.SmoothingFreq(n=2).fit(x)
bins = sm.transform(x)
pd.Series(bins).value_counts().sort_index()
```

## Optimize the number of bins and refit
```python
tuned = dal.SmoothingFreq(n=range(1, 21)).fit(iris["sepal_length"])
tuned.n
```

## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Discretization)
