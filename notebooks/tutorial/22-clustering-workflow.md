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

# Tutorial 22 - Clustering Workflow

Clustering has no explicit target during training, but the workflow is still structured: choose
a method, fit it, obtain assignments, and inspect the result. Preprocessing can change
unsupervised results substantially.

## K-means on the original scale
```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterKmeans(k=3, seed=1).fit(x)
clu = model.cluster(x)
print(pd.Series(clu).value_counts().sort_index().to_dict())
model.evaluate(clu, iris["Species"])["metrics"]
```

## Normalize and repeat the same clustering
```python
norm = dal.MinMax().fit(iris)
x_norm = norm.transform(iris).drop(columns="Species")

model_norm = dal.ClusterKmeans(k=3, seed=1).fit(x_norm)
clu_norm = model_norm.cluster(x_norm)
print(pd.Series(clu_norm).value_counts().sort_index().to_dict())
model_norm.evaluate(clu_norm, iris["Species"])["metrics"]
```

In unsupervised learning, the data representation can matter as much as the algorithm.
