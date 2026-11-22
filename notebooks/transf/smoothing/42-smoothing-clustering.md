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

# Smoothing by Clustering (SmoothingCluster)

## About the technique
- `SmoothingCluster`: discretization/smoothing by class-aware clustering, so the grouping is influenced by the target class.

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
sm = dal.SmoothingCluster("Species", n=3).fit(iris[["sepal_length", "Species"]])
bins = sm.transform(iris["sepal_length"])
pd.Series(bins).value_counts().sort_index()
```

## Optimize the number of bins and refit
```python
tuned = dal.SmoothingCluster("Species", n=range(1, 9)).fit(iris[["sepal_length", "Species"]])
tuned.n
```

## References
- Han, J., Kamber, M., and Pei, J. (2011). *Data Mining: Concepts and Techniques.* (Discretization)
