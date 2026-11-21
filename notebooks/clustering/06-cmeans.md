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

# Fuzzy c-means Clustering (ClusterCmeans)

Fuzzy c-means is a *soft* clustering method: instead of assigning each point to a single
cluster, it gives every point a degree of membership in each of the `centers` clusters.
The fuzziness parameter `m` (m > 1) controls how soft the partition is. `cluster`
returns the hard assignment (the cluster of highest membership); the full membership
matrix is kept in the `membership_` attribute.

**Reference:** Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function
Algorithms.*

```python
import pandas as pd

import pydaltoolbox as dal

iris = dal.datasets.iris()
x = iris.drop(columns="Species")

model = dal.ClusterCmeans(centers=3, m=2, seed=42).fit(x)
labels = model.cluster(x)
```

Cross-tabulating the hard assignments against the true species:

```python
pd.crosstab(labels, iris["Species"])
```

The membership matrix holds, for each observation, how strongly it belongs to each
cluster (rows sum to 1):

```python
pd.DataFrame(model.membership_).head()
```

`evaluate` reports internal quality (silhouette and the within-cluster fuzzy error) and,
when a reference label is given, external agreement (entropy, purity, adjusted Rand
index).

```python
model.evaluate(labels, iris["Species"])["metrics"]
```
