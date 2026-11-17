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

Agglomerative hierarchical clustering builds a dendrogram by repeatedly merging the
closest groups; a flat partition is obtained by cutting the tree at `k` clusters (or at
a distance threshold `h`). The linkage (`method`) and distance (`dist`) control how
groups are compared, and features are standardized first by default.

**Reference:** Johnson, S. C. (1967). *Hierarchical Clustering Schemes.* Psychometrika.

```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterHclust(k=3, method="ward").fit(x)
labels = model.cluster(x)
pd.crosstab(labels, iris["Species"])
```

```python
model.evaluate(labels, iris["Species"])["metrics"]
```
