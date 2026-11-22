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



# Hierarchical Clustering (ClusterHclust)


## About the method
- `ClusterHclust`: agglomerative hierarchical clustering.

## Didactic goal
Keep the same clustering line of experiment and add a hierarchy that can be inspected before or after cutting the tree into clusters.

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
model = dal.ClusterHclust(k=3, method="ward")
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

## Inspect the hierarchy graphically (needs the optional `[viz]` extra)
```python
from scipy.cluster.hierarchy import linkage

ax = dal.plot_dendrogram(
    linkage(x.to_numpy(), method="ward"),
    labels=False,
    title="Hierarchical clustering of iris",
)
ax.figure
```

## References
- Johnson, S. C. (1967). *Hierarchical Clustering Schemes.*
