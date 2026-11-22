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



# Gaussian Mixture Model (ClusterGmm)


## About the method
- `ClusterGmm`: Gaussian mixture model clustering.

## Didactic goal
Keep the same clustering line of experiment and change only the clustering family to a probabilistic mixture model.

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
model = dal.ClusterGmm(g=3, seed=42)
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

## References
- Fraley, C., and Raftery, A. E. (2002). *Model-Based Clustering, Discriminant Analysis, and Density Estimation.*
