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



# Fuzzy c-means (ClusterCmeans)


## About the method
- `ClusterCmeans`: fuzzy c-means clustering, which keeps membership degrees instead of only hard assignments.

## Didactic goal
Keep the same clustering line of experiment and change only the notion of assignment from hard clusters to soft memberships.

## Environment setup
```python
import warnings

import pandas as pd

import pydaltoolbox as dal

warnings.filterwarnings("ignore")
```

## Load data; keep reference labels only for interpretation
```python
iris = dal.datasets.iris()
x = iris.drop(columns="Species")
ref = iris["Species"]
x.head()
```

## Model configuration
```python
model = dal.ClusterCmeans(centers=3, m=2, seed=42)
```

## Fit the model and obtain cluster labels
```python
model = model.fit(x)
clu = model.cluster(x)
pd.Series(clu).value_counts().sort_index()
```

## Evaluate the partition
```python
model.evaluate(clu, ref)["metrics"]
```

## Inspect the membership matrix attached to the result
```python
pd.DataFrame(model.membership_).head()
```

## References
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms.*
